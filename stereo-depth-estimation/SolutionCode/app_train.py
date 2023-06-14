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
from asset.trainer import TrainerKitti, TrainerBaselineKitti

if not sys.warnoptions:
    warnings.simplefilter("ignore", category=UserWarning)

            
#os.environ['WANDB_MODE'] = 'dryrun' #uncomment to prevent wandb logging - 'offline mode'

            
hyperparameter_defaults = dict(
    encoder_depth= 4,
    encoder_backbone= 'efficientnet-b3', #'efficientnet-b3', mobilenet_v2
    height_train=256,
    width_train=512,
    height_valid=368,
    width_valid=1248,
    n_epochs=240,
    patience=24,
    batch_size_train=8,
    batch_size_valid=1,
    lr=1e-3,
    lr_decay=0.992,
    pretrained=False,
    augment_data=True
)

wandb.init(config=hyperparameter_defaults, project="CudaLab_SS21_Train")
print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Cuda Version:", torch.version.cuda)
print("CUDNN Version:", torch.backends.cudnn.version())

config=wandb.config

pretrained_title = config.encoder_backbone + f'_encoder_depth_{config.encoder_depth}'+'_pretrained'
#pretrained_title = 'baseline_model_pretrained'
pretrain_str = '_pretrain' if config.pretrained else '_nopretrain'
augment_str = '_augment' if config.augment_data else '_noaugment'
experiment_title = config.encoder_backbone + pretrain_str + augment_str + '_eval'
#experiment_title = 'baseline_model' + pretrain_str + augment_str
print('Experiment title:', experiment_title)
pretty_print(config)

# Dataset and Dataloader creation
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

transform = DisparityTransformCompose([DisparityTransform(transforms.Normalize(MEAN, STD), apply_to_disparity=False),
                                       RandomCrop(output_size=(config.height_train,config.width_train))])

transform_large = DisparityTransformCompose([DisparityTransform(transforms.Normalize(MEAN, STD), apply_to_disparity=False)])

inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(MEAN, STD)],
   std= [1/s for s in STD]
)


kitti_dataset = DisparityDataset(get_left_frames_kitti(), left_to_right_kitti, left_to_disparity_kitti, load_image_kitti, load_disparity_kitti, transform)
kitti_dataset_large = DisparityDataset(get_left_frames_kitti(), left_to_right_kitti, left_to_disparity_kitti, load_image_kitti, load_disparity_kitti, transform_large)
print('Number of samples in the KITTI dataset:', len(kitti_dataset))

train_share = 0.75
N_samples = len(kitti_dataset)
#N_samples = 40 ### REMOVE THIS LINE AFTER DEBUGGING!!!
indices = list(range(N_samples))
split = int(np.ceil(train_share * N_samples))
np.random.seed(42)
np.random.shuffle(indices)

train_idx, valid_idx = indices[:split], indices[split:]
print('Number of training : validation samples -', len(train_idx), ':', len(valid_idx))
trainset = torch.utils.data.Subset(kitti_dataset, train_idx)
validset = torch.utils.data.Subset(kitti_dataset_large, valid_idx)

if config.augment_data:
    transform = DisparityTransformCompose([transform,
                                       CutMix(trainset)])
    kitti_dataset = DisparityDataset(get_left_frames_kitti(), left_to_right_kitti, left_to_disparity_kitti, load_image_kitti, load_disparity_kitti, transform)
    trainset = torch.utils.data.Subset(kitti_dataset, train_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size_train, shuffle=True, num_workers=10)
validloader = torch.utils.data.DataLoader(validset, batch_size=config.batch_size_valid, shuffle=False, num_workers=10)

# Setup metrics
criterion_smoothl1 = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)
wrapped_criterion_smoothl1 = NonZeroWrapper(torch.nn.SmoothL1Loss(reduction='none', beta=1.0), max_disp=192)
wrapped_criterion_smoothl1 = IntermediateSupervisionWrapper(wrapped_criterion_smoothl1)
wrapped_criterion_1PE = NonZeroWrapper(PE(reduction='none', threshold=1))
wrapped_criterion_3PE = NonZeroWrapper(PE(reduction='none'))
wrapped_criterion_5PE = NonZeroWrapper(PE(reduction='none', threshold=5))
metrics = {
  "1PE": wrapped_criterion_1PE,
  "3PE": wrapped_criterion_3PE,
  "5PE": wrapped_criterion_5PE
}

# Init model
model_custom = CustomModel(encoder_depth=config.encoder_depth, backbone=config.encoder_backbone)
#model_custom = BaselineModel()

n_params = count_parameters(model_custom)
wandb.run.summary['Parameter Number'] = n_params
print(f'The model has {n_params} trainable parameters.')

# Load pretrained if required
if config.pretrained:
    print('Loading pretrained model!')
    title = pretrained_title
    pretrain_loader = TrainerKitti(model_custom, wrapped_criterion_smoothl1, trainloader, validloader, eval_metrics=metrics, es_mode='min', description=title, patience=32, lr=config.lr, n_epochs=config.n_epochs, lr_decay=config.lr_decay, size_y=config.height_valid, size_x=config.width_valid)
    pretrain_loader.load_model()
    model_custom = pretrain_loader.model
    
# Train
title = experiment_title
custom_trainer = TrainerKitti(model_custom, wrapped_criterion_smoothl1, trainloader, validloader, eval_metrics=metrics, es_mode='min', description=title, patience=config.patience, lr=config.lr, n_epochs=config.n_epochs, lr_decay=config.lr_decay, size_y=config.height_valid, size_x=config.width_valid)
custom_trainer.fit()

# Save parameters and training logs to WandB
wandb.save('models/'+experiment_title+'_best.pt')
wandb.save('trainer_logs/'+experiment_title+'.npy')
