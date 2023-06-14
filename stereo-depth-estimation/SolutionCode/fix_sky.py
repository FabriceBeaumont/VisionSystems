# WandB setup first
import sys
import os
import wandb
import torch
import warnings
from asset.utils import *
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from asset.transforms import DisparityTransformCompose, DisparityTransform, RandomCrop, Interpolate, CutMix
from asset.datasets import DisparityDataset
from asset.criterion import NonZeroWrapper, IntermediateSupervisionWrapper, PE
from asset.model import BaselineModel, CustomModel
from asset.trainer import TrainerKitti, TrainerSky
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

if not sys.warnoptions:
    warnings.simplefilter("ignore", category=UserWarning)

            
#os.environ['WANDB_MODE'] = 'dryrun' #uncomment to prevent wandb logging - 'offline mode'

hyperparameter_defaults = dict(
    encoder_depth= 4,
    encoder_backbone='efficientnet-b3', #mobilenet_v2
    height_train=224,
    width_train=288,
    n_epochs=1000,
    patience=24,
    batch_size_train=256,
    batch_size_valid=352,
    lr=1e-3,
    lr_decay=0.992,
    pretrained=True,
    augment_data=False
)

wandb.init(config=hyperparameter_defaults, project="CudaLab_SS21_Sky_Fix")
print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Cuda Version:", torch.version.cuda)
print("CUDNN Version:", torch.backends.cudnn.version())

config=wandb.config

pretrained_title = config.encoder_backbone + f'_encoder_depth_{config.encoder_depth}'+'_pretrained'
pretrain_str = '_pretrain' if config.pretrained else '_nopretrain'
augment_str = '_augment' if config.augment_data else '_noaugment'
experiment_title = config.encoder_backbone + pretrain_str + augment_str +'_eval'
print('Experiment title:', experiment_title)
pretty_print(config)

def get_frames_skyfinder(skyfinder_root ='/home/data/Skyfinder'):
    """Return the paths of all left camera images in the Driving subset of the SceneFlow dataset."""
    
    imgs_paths, mask_paths = [], []
    for key in os.listdir(skyfinder_root):
        if key != 'Masks':
            for img in os.listdir(os.path.join(skyfinder_root, key)):
                imgs_paths.append(os.path.join(skyfinder_root, key, img))
                mask_paths.append(os.path.join(skyfinder_root, 'Masks', key+'.png'))
    return imgs_paths, mask_paths

from torch.utils.data import Dataset
class SkyFinder(Dataset):
    
    def __init__(self, skyfinder_root = '/home/data/Skyfinder', transform=None):
        self.imgs_paths, self.masks_paths = get_frames_skyfinder(skyfinder_root=skyfinder_root)
        self.broken_inds = np.load(os.path.join(skyfinder_root, 'Masks', 'broken_inds.npy'))
        for i in self.broken_inds:
            del self.imgs_paths[i]
            del self.masks_paths[i]
        self.transform = transform
        return
    
    def load_image(self, img_path):
        return torch.tensor(np.asarray(Image.open(img_path).convert('RGB'))).float().permute(2,0,1)/255.
    
    def load_mask(self, mask_path):
        return torch.tensor(np.asarray(Image.open(mask_path).convert('L'))).float().unsqueeze(0)/255.
    
    def __len__(self):
        """Return the number of file paths."""
        
        return len(self.imgs_paths)

    def __getitem__(self, i):
        img_path, mask_path = self.imgs_paths[i], self.masks_paths[i]
        
        img, mask = self.load_image(img_path), self.load_mask(mask_path)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask
    
model = CustomModel(encoder_depth=config.encoder_depth, backbone=config.encoder_backbone)
savepath = 'models/'+experiment_title+'_best.pt'
checkpoint = torch.load(savepath)
model.load_state_dict(checkpoint['model_state_dict'])
feats = model.features.cpu()

class RandomCropSky(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, mask):
        h, w = img.shape[-2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[:, top: top + new_h, left: left + new_w]
        mask = mask[:, top: top + new_h, left: left + new_w]
        
        return img, mask
    
class EncodeSky(object):
    def __init__(self, features):
        self.features = features
        self.features.requires_grad_(False)
        self.features.eval()

    def __call__(self, img, mask):
        with torch.no_grad():
            img = self.features(img.unsqueeze(0)).squeeze(0)
        
        return img, mask
    
class SkyTransformCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        
        return img, mask
    
class SkyTransform(object):
    def __init__(self, transform, apply_to_mask=True):
        self.transform = transform
        self.apply_to_mask = apply_to_mask

    def __call__(self, img, mask):

        img = self.transform(img)
        if self.apply_to_mask:
            mask = self.transform(mask)
            
        return img, mask
    
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform_sky = SkyTransformCompose([SkyTransform(transforms.Normalize(MEAN, STD), apply_to_mask=False),
                                    RandomCropSky(output_size=(config.height_train,config.width_train))])
sky_dataset = SkyFinder(transform=transform_sky)

from asset.model import BasicBlock

# skynet = nn.Sequential(
#             BasicBlock(inplanes= 480, planes= 64, stride=1),
#             nn.Upsample(scale_factor=2),
#             BasicBlock(inplanes= 64, planes=32, stride=1),
#             nn.Upsample(scale_factor=2),
#             BasicBlock(inplanes=32, planes=16, stride=1),
#             BasicBlock(inplanes=16, planes=16, stride=1),
#             BasicBlock(inplanes=16, planes=1, stride=1),
#         )

skynet = smp.Unet(
    encoder_name="timm-mobilenetv3_large_minimal_100",        
    encoder_weights="imagenet",    
    in_channels=3,                  
    classes=1
)

n_params = count_parameters(skynet)
wandb.run.summary['Parameter Number'] = n_params
print(f'The model has {n_params} trainable parameters.')

train_share = 0.9
N_samples = len(sky_dataset)
#N_samples=400
indices = list(range(N_samples))
split = int(np.ceil(train_share * N_samples))
np.random.seed(42)
np.random.shuffle(indices)

train_idx, valid_idx = indices[:split], indices[split:]
print('Number of training : validation samples -', len(train_idx), ':', len(valid_idx))
trainset_sky = torch.utils.data.Subset(sky_dataset, train_idx)
validset_sky = torch.utils.data.Subset(sky_dataset, valid_idx)

trainloader_sky = torch.utils.data.DataLoader(trainset_sky, batch_size=config.batch_size_train, shuffle=True, num_workers=10)
validloader_sky = torch.utils.data.DataLoader(validset_sky, batch_size=config.batch_size_valid, shuffle=False, num_workers=10)

criterion_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')

class FocalLoss(nn.Module):
    def __init__(self, alpha=.8, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

    def forward(self, inputs, targets):
        ps = F.sigmoid(inputs)
        p_t = torch.where(targets==1, ps, 1 - ps)
        bce = torch.where(targets==1, -F.logsigmoid(inputs), -F.logsigmoid(-inputs))
        alpha_factor = torch.ones_like(targets) * self.alpha
        alpha_t = torch.where(targets==1, alpha_factor, 1 - alpha_factor)
        F_loss = alpha_t * (1 - p_t)**self.gamma * bce
        return torch.mean(F_loss)

criterion_focal = FocalLoss()
metrics = {
  "BCE": criterion_bce,
}

title='sky_detection'
test_trainer = TrainerSky(skynet, criterion_focal, trainloader_sky, validloader_sky, features=feats,eval_metrics=metrics, es_mode='min', patience=config.patience, lr=config.lr, n_epochs=config.n_epochs, lr_decay=config.lr_decay, description=title)

test_trainer.fit()

wandb.save('models/'+title+'_best.pt')
wandb.save('trainer_logs/'+title+'.npy')