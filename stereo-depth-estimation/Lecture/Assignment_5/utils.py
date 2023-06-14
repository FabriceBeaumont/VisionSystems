import numpy as np
import matplotlib.pyplot as plt
import random as rand
import torch

# Utils
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


def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    return weighted_losses.sum()/counts_np.sum()


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
