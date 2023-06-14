import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import random as rand
import PIL

# Class which handles CutMix transform, should be used as transform and target_transform
# For usage example on common datasets see CutMix_Demo.ipynb
class CutMix:
  # implemented according to pseudocode from https://arxiv.org/pdf/1905.04899.pdf
    def __init__(self, dataset: Dataset, seed: int, num_classes: int = 0):
        self.rng = np.random.default_rng(seed)
        self.dataset = dataset
        self.ToTensor = transforms.ToTensor()
        self.ToPIL = transforms.ToPILImage()
        self.len = len(self.dataset)
        if num_classes <= 0:
            self.num_classes = len(self.dataset.classes)
        else:
            self.num_classes = num_classes
        test_image, _ = dataset.__getitem__(0)
        self.W, self.H = test_image.size

    def __call__(self, sample):
        # shared non-determinism between image and label CutMix
        lam = self.rng.random()
        r_x = self.rng.random()*self.W
        r_y = self.rng.random()*self.H
        # the following two lines correct a typo in the paper pseudocode
        r_w = np.sqrt(1-lam)*self.W
        r_h = np.sqrt(1-lam)*self.H

        x_1 = int(np.round(np.max([r_x-r_w/2,0])))
        x_2 = int(np.round(np.min([r_x+r_w/2,self.W])))
        y_1 = int(np.round(np.max([r_y-r_h/2,0])))
        y_2 = int(np.round(np.min([r_y+r_h/2,self.H])))
        # since we apply CutMix element-wise, replace batch shuffle by this
        mix_ind = self.rng.integers(self.len, endpoint=False)
        image_s, label_s = self.dataset.__getitem__(mix_ind)

        # what to do when the sample is an image
        if isinstance(sample, PIL.Image.Image):
            image = self.ToTensor(sample)
            image_s = self.ToTensor(image_s)
            image[:, y_1:y_2, x_1:x_2] = image_s[:, y_1:y_2, x_1:x_2]
            image = self.ToPIL(image)
            return image

        # what to do when the sample is an index
        else:
            label = sample
            label = torch.Tensor([label]).long()
            label = F.one_hot(label, num_classes=self.num_classes).squeeze()
            label_s = torch.Tensor([label_s]).long()
            label_s = F.one_hot(label_s, num_classes=self.num_classes).squeeze()
            lam_true = 1 - (x_2 - x_1) * (y_2 - y_1) / (self.H * self.W)
            return lam_true * label + (1 - lam_true) * label_s

# target transform - one hot encodes integer labels
class OneHot:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    def __call__(self, label):
        label = torch.Tensor([label]).long()
        return F.one_hot(label, num_classes=self.num_classes).squeeze().float()

# function to undo normalization and then display images
def show_grid(data, labels=None, pred_labels=None, class_names=['P','R'], mean=None, std=None, description=None):
    """Imshow for Tensor."""
    data = data.numpy().transpose((0, 2, 3, 1))
    if mean is not None and std is not None:
        data = np.array(std) * data + np.array(mean)
        data = np.clip(data, 0, 1)
    
    fig = plt.figure(figsize=(8*2, 4*2))
    if description is not None:
        fig.suptitle(description, fontsize=30)
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(data[i])
        plt.axis("off")
        if labels is not None:
            label = labels[i]
            sorted_args = label.argsort().squeeze()
            title = f'GT  : {label[sorted_args[-1]] :.2f} {class_names[sorted_args[-1]]}, {label[sorted_args[-2]] :.2f} {class_names[sorted_args[-2]]}'
            if pred_labels is not None:
                pred_label = pred_labels[i]
                sorted_args = pred_label.argsort().squeeze()
                title += f'\n Out : {pred_label[sorted_args[-1]] :.2f} {class_names[sorted_args[-1]]}, {pred_label[sorted_args[-2]] :.2f} {class_names[sorted_args[-2]]}'
            plt.title(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# Can be used to trigger early stopping in a training process
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=7):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False

#given some losses and some batch size counts, return weighted epoch average loss
def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    return weighted_losses.sum()/counts_np.sum()

def reject_randomness(manualSeed):
    np.random.seed(manualSeed)
    rand.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None

def plot_history(trainer):
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,5)
    fig.suptitle(trainer.description, fontsize=18)
    ax[0].plot(trainer.history['train loss'], label='train loss', c='r', lw=3)
    ax[0].plot(trainer.history['valid loss'], label='valid loss', c='b', lw=3)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].set_title('Training and Validation Losses')

    acc_array = np.array(trainer.history['accuracy'])
    ax[1].plot(trainer.history['accuracy'], label='accuracy', c='r', lw=3)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend(loc="lower right")
    ax[1].set_title(f'Validation Accuracy\nBest Accuracy: {acc_array.max()} @ Epoch {acc_array.argmax()+1}')
    plt.show()


def plot_samples(trainer):
    softmax = nn.Softmax(dim=-1)
    inputs, classes, net_out = trainer.get_sample(mode='valid')
    net_out = softmax(net_out)
    labels = [label for label in classes]
    show_grid(inputs, labels=labels, pred_labels=net_out, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], description=trainer.description + ' Validation Samples')
    
    inputs, classes, net_out = trainer.get_sample(mode='train')
    net_out = softmax(net_out)
    labels = [label for label in classes]
    show_grid(inputs, labels=labels, pred_labels=net_out, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], description=trainer.description + ' Training Samples')
    
def get_best_accuracy(trainer):
    acc_array = np.array(trainer.history['accuracy'])
    return acc_array.max(), acc_array.argmax()+1