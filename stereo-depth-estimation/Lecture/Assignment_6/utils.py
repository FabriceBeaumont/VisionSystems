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
    fig, ax = plt.subplots(1,5)
    fig.set_size_inches(33,8)
    fig.suptitle(trainer.description, fontsize=15)
    ax[0].plot(trainer.history['train loss'], label='train loss', c='lightcoral', lw=3)
    ax[0].plot(trainer.history['valid loss'], label='valid loss', c='cornflowerblue', lw=3)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].set_title('Training and Validation Losses')

    loss_array = np.array(trainer.history['Reconstruction Loss'])
    ax[1].plot(trainer.history['Reconstruction Loss'], label='Reconstruction Loss', c='teal', lw=3)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Reconstruction Loss")
    ax[1].legend(loc="lower right")
    ax[1].set_title(f'Reconstruction Loss\nBest Value: {loss_array.min()} @ Epoch {loss_array.argmin()+1}')
    
    loss_array = np.array(trainer.history['KL Loss'])
    ax[2].plot(trainer.history['KL Loss'], label='KL Loss', c='green', lw=3)
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("KL Loss")
    ax[2].legend(loc="lower right")
    ax[2].set_title(f'KL Loss\nBest Value: {loss_array.min()} @ Epoch {loss_array.argmin()+1}')
    
    loss_array = np.array(trainer.history['Perceptual Loss'])
    ax[3].plot(trainer.history['Perceptual Loss'], label='Perceptual Loss', c='darkseagreen', lw=3)
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Perceptual Loss")
    ax[3].legend(loc="lower right")
    ax[3].set_title(f'Perceptual Loss\nBest Value: {loss_array.min()} @ Epoch {loss_array.argmin()+1}')
    
    loss_array = np.array(trainer.history["DSSIM"])
    ax[4].plot(trainer.history["DSSIM"], label="DSSIM", c='greenyellow', lw=3)
    ax[4].set_xlabel("Epoch")
    ax[4].set_ylabel("DSSIM")
    ax[4].legend(loc="lower right")
    ax[4].set_title(f'DSSIM\nBest Value: {loss_array.min()} @ Epoch {loss_array.argmin()+1}')
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


@torch.no_grad()
def sample_interpolation(model, p1, p2, N=15):
    """ Sampling N points from the line that connects p1 and p2 """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.stack([p1 * t + p2 * (1-t) for t in torch.linspace(0, 1, N).to(device)])
    decoded = model.decoder(z)
    decoded_imgs = decoded.cpu().view(-1,3,32,32)
    return decoded_imgs

COLORS = ['r', 'b', 'g', 'y', 'purple', 'orange', 'k', 'brown', 'grey',
          'c', "gold", "fuchsia", "lime", "darkred", "tomato", "navy"]

def display_projections(points, labels, ax=None, legend=None):
    """ Displaying low-dimensional data projections """
    
    legend = [f"Class {l}" for l in np.unique(labels)] if legend is None else legend
    if(ax is None):
        _, ax = plt.subplots(1,1,figsize=(12,6))
    
    for i,l in enumerate(np.unique(labels)):
        idx = np.where(l==labels)

        ax.scatter(points[idx, 0], points[idx, 1], label=legend[int(l)], c=COLORS[i])
    ax.legend(loc="best")