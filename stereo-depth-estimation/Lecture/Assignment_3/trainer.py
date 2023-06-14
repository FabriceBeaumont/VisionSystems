import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random as rand
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import joblib
import optuna
from optuna.trial import TrialState
import time

# reject randomness (as much as possible)
manualSeed = 2021

np.random.seed(manualSeed)
rand.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

def train(model, lr, train_loader, valid_loader, regularize):
    alpha = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    start = time.time()
    es = EarlyStopping(mode='max', patience=10)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    N_EPOCHS = 250
    UPDATE_EVERY = 1
    train_loss_hist, train_acc_hist, valid_loss_hist, valid_acc_hist  = [], [], [], []
    progress_bar = tqdm(range(N_EPOCHS), total=N_EPOCHS, position=0, leave=True)
    for epoch in progress_bar:
        train_loss_list, train_acc_list, batch_sizes_train = [], [], []
        valid_loss_list, valid_acc_list, batch_sizes_valid = [], [], []

        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            labels = labels.to(device)
            pred_labels, _ = model(imgs.to(device))
            loss = criterion(pred_labels, labels)
            loss = loss + alpha*regularize(model)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            batch_sizes_train.append(labels.shape[0])
            pred_label_inds = pred_labels.argmax(dim=-1)
            acc = (pred_label_inds == labels).float().mean() * 100
            train_acc_list.append(acc.item())

        avg_train_loss = epoch_average(train_loss_list, batch_sizes_train)
        avg_train_acc = epoch_average(train_acc_list, batch_sizes_train)
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(avg_train_acc)

        model.eval()
        for imgs, labels in valid_loader:
            with torch.no_grad():
                labels = labels.to(device)
                pred_labels, _ = model(imgs.to(device))
                loss = criterion(pred_labels, labels)

                valid_loss_list.append(loss.item())
                batch_sizes_valid.append(imgs.shape[0])
                pred_label_inds = pred_labels.argmax(dim=-1)
                acc = (pred_label_inds == labels).float().mean() * 100
                valid_acc_list.append(acc.item())

        avg_valid_loss = epoch_average(valid_loss_list, batch_sizes_valid)
        avg_valid_acc = epoch_average(valid_acc_list, batch_sizes_valid)
        valid_loss_hist.append(avg_valid_loss)
        valid_acc_hist.append(avg_valid_acc)

        if(epoch % UPDATE_EVERY == 0 or epoch == N_EPOCHS-1):
            progress_bar.set_description(f"Epoch {epoch+1} - Train: loss {train_loss_hist[-1]:.4f} | acc {train_acc_hist[-1]:.2f} - Valid: loss {valid_loss_hist[-1]:.4f} | acc {valid_acc_hist[-1]:.2f}. ")

        scheduler.step(avg_valid_loss)
        if(es.step(avg_valid_acc)):
            break
    
    return avg_valid_acc, epoch, time.time()-start, train_acc_hist, valid_acc_hist