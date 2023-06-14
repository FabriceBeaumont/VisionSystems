# WandB setup first
import sys
import os
import wandb
import torch
import warnings
from asset.utils import *
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from asset.transforms import DisparityTransformCompose, DisparityTransform, RandomCrop, Interpolate
from asset.datasets import DisparityDataset
from asset.criterion import NonZeroWrapper, IntermediateSupervisionWrapper, PE
from asset.model import BaselineModel, CustomModel
from asset.trainer import Trainer, TrainerBaseline

if not sys.warnoptions:
    warnings.simplefilter("ignore", category=UserWarning)

            
#os.environ['WANDB_MODE'] = 'dryrun' #uncomment to prevent wandb logging - 'offline mode'

hyperparameter_defaults = dict(
    encoder_depth=4, #3, -1
    encoder_backbone='efficientnet-b3', #'mobilenet_v2', 'NA'
    height_warmup=512,
    width_warmup=928,
    height=528,
    width=944,
    n_epochs_warmup=3,
    n_epochs=8,
    batch_size_train_warmup=8,
    batch_size_valid_warmup=32,
    batch_size_train=2,
    batch_size_valid=8,
    lr=1e-3,
    lr_decay_warmup=0.6, 
    lr_decay=0.75
)

wandb.init(config=hyperparameter_defaults, project="CudaLab_SS21_Pretraining")
print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Cuda Version:", torch.version.cuda)
print("CUDNN Version:", torch.backends.cudnn.version())

config=wandb.config

experiment_title = config.encoder_backbone + f'_encoder_depth_{config.encoder_depth}'+'_pretrained_long' # 'baseline_model_pretrained'
print('Experiment title:', experiment_title)
pretty_print(config)

# Dataset creation
transform_warmup = DisparityTransformCompose([DisparityTransform(transforms.Normalize(mean_imagenet, std_imagenet), apply_to_disparity=False),
                                              RandomCrop(output_size=(config.height_warmup,config.width_warmup)),
                                              Interpolate(scale_factor=0.5)])

transform = DisparityTransformCompose([DisparityTransform(transforms.Normalize(mean_imagenet, std_imagenet), apply_to_disparity=False),
                                       RandomCrop(output_size=(config.height,config.width))])

monkaa_dataset_warmup = DisparityDataset(get_left_frames_monkaa(), left_to_right_monkaa, left_to_disparity_monkaa, load_image_sceneflow, load_disparity_sceneflow, transform_warmup)
driving_dataset_warmup = DisparityDataset(get_left_frames_driving(), left_to_right_driving, left_to_disparity_driving, load_image_sceneflow, load_disparity_sceneflow, transform_warmup)
flyingthings3d_dataset_warmup = DisparityDataset(get_left_frames_flyingthings3d(use_difficult_examples=False), left_to_right_flyingthings3d, left_to_disparity_flyingthings3d, load_image_sceneflow, load_disparity_sceneflow, transform_warmup)
concat_dataset_warmup = ConcatDataset([monkaa_dataset_warmup, driving_dataset_warmup, flyingthings3d_dataset_warmup])


monkaa_dataset = DisparityDataset(get_left_frames_monkaa(), left_to_right_monkaa, left_to_disparity_monkaa, load_image_sceneflow, load_disparity_sceneflow, transform)
driving_dataset = DisparityDataset(get_left_frames_driving(), left_to_right_driving, left_to_disparity_driving, load_image_sceneflow, load_disparity_sceneflow, transform)
flyingthings3d_dataset = DisparityDataset(get_left_frames_flyingthings3d(use_difficult_examples=False), left_to_right_flyingthings3d, left_to_disparity_flyingthings3d, load_image_sceneflow, load_disparity_sceneflow, transform)
print('Number of samples in the Monkaa dataset:', len(monkaa_dataset))
print('Number of samples in the Driving dataset:', len(driving_dataset))
print('Number of samples in the Flyingthings3d dataset:', len(flyingthings3d_dataset))
concat_dataset = ConcatDataset([monkaa_dataset, driving_dataset, flyingthings3d_dataset])
print('Number of samples in the concatenated Sceneflow dataset:', len(concat_dataset))

# Create dataloaders
train_share_concat = 0.8
N_samples_concat = len(concat_dataset)
#N_samples_concat = 400 ### REMOVE THIS LINE AFTER DEBUGGING!!!
indices_concat = list(range(N_samples_concat))
split_concat = int(np.ceil(train_share_concat * N_samples_concat))
np.random.seed(42)
np.random.shuffle(indices_concat)

BATCHSIZE_CONCAT_WARMUP = config.batch_size_train_warmup
BATCHSIZE_CONCAT_VALID_WARMUP = config.batch_size_valid_warmup

BATCHSIZE_CONCAT = config.batch_size_train
BATCHSIZE_CONCAT_VALID = config.batch_size_valid

train_idx_concat, valid_idx_concat = indices_concat[:split_concat], indices_concat[split_concat:]
print('Number of training : validation samples -', len(train_idx_concat), ':', len(valid_idx_concat))

trainset_concat_warmup = torch.utils.data.Subset(concat_dataset_warmup, train_idx_concat)
validset_concat_warmup = torch.utils.data.Subset(concat_dataset_warmup, valid_idx_concat)

trainset_concat = torch.utils.data.Subset(concat_dataset, train_idx_concat)
validset_concat = torch.utils.data.Subset(concat_dataset, valid_idx_concat)

trainloader_concat_warmup = torch.utils.data.DataLoader(trainset_concat_warmup, batch_size=BATCHSIZE_CONCAT_WARMUP, shuffle=True, num_workers=10)
validloader_concat_warmup = torch.utils.data.DataLoader(validset_concat_warmup, batch_size=BATCHSIZE_CONCAT_VALID_WARMUP, shuffle=False, num_workers=10)

trainloader_concat = torch.utils.data.DataLoader(trainset_concat, batch_size=BATCHSIZE_CONCAT, shuffle=True, num_workers=10)
validloader_concat = torch.utils.data.DataLoader(validset_concat, batch_size=BATCHSIZE_CONCAT_VALID, shuffle=False, num_workers=10)

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

# Complete warmup
title = experiment_title
custom_trainer = Trainer(model_custom, wrapped_criterion_smoothl1, trainloader_concat_warmup, validloader_concat_warmup, eval_metrics=metrics, es_mode='min', description=title, patience=32, lr=config.lr, n_epochs=config.n_epochs_warmup, lr_decay=config.lr_decay_warmup)
print('Starting warmup...')
custom_trainer.fit() 

# Train at full resolution
model_custom = custom_trainer.model
custom_trainer = Trainer(model_custom, wrapped_criterion_smoothl1, trainloader_concat, validloader_concat, eval_metrics=metrics, es_mode='min', description=title, patience=32, lr=config.lr, n_epochs=config.n_epochs, lr_decay=config.lr_decay)
print()
print('Starting training...')
custom_trainer.fit()
print()
# Upload model weights
wandb.save('models/'+experiment_title+'_best.pt')
wandb.save('trainer_logs/'+experiment_title+'.npy')
print('Run completed!')
