import os
import torch
import torch.nn as nn
import time
from tqdm.notebook import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from utils import EarlyStopping, epoch_average

class Trainer():
    def __init__(self, model, criterion, train_loader, valid_loader, eval_metrics=None, lr=5e-4,
                 patience=5, es_mode='min', description='untitled', n_epochs=250):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.description = description
        self.n_epochs = n_epochs
        self.model = model.to(self.device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr = lr
        self.patience = patience
        self.eval_metrics = eval_metrics
        self.es_mode = es_mode
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience)
        self.es = EarlyStopping(mode=self.es_mode, patience=2*self.patience)
        self.history = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        
        
    def inference_step(self, x):
        return self.model(x.to(self.device))
    
    def save_hist(self):
        if(not os.path.exists("trainer_logs")):
            os.makedirs("trainer_logs")
        savepath = f"trainer_logs/{self.description}.npy"
        np.save(savepath, self.history)
        return
    
    def save_model(self):
        if(not os.path.exists("models")):
            os.makedirs("models")
        if(not os.path.exists("trainer_logs")):
            os.makedirs("trainer_logs")
        savepath = f"models/{self.description}_best.pt"
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, savepath)
        self.save_hist()
        return
    
    def load_model(self):
        savepath = f"models/{self.description}_best.pt"
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = f"trainer_logs/{self.description}.npy"
        self.history = np.load(savepath,allow_pickle='TRUE').item()
        return
    
    def train_epoch(self):
        loss_list, batch_sizes = [], []
        for data, target in self.train_loader:
            target = data.clone()
            self.optimizer.zero_grad()
            net_out = self.inference_step(data)
            loss = self.criterion(net_out, target.to(self.device))
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
            batch_sizes.append(data.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        return average_loss
    
    def get_sample(self, mode='valid'):
        if mode == 'valid':
            self.model.eval()
            data, target = next(iter(self.valid_loader))
            net_out = self.inference_step(data) 
            self.model.train()
        else:
            data, target = next(iter(self.train_loader))
            net_out = self.inference_step(data)
        x_hat, (z, mu, log_var) = net_out
        return data.cpu(), target.cpu(), (x_hat.cpu(), (z.cpu(), mu.cpu(), log_var.cpu()))
    
    @torch.no_grad()
    def eval_epoch(self):
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for data, target in self.valid_loader:
            target = data.clone()
            net_out = self.inference_step(data)
            target = target.to(self.device)
            loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(data.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(metric(net_out,target).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
        for epoch in progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()
            
            self.model.eval()
            valid_loss = self.eval_epoch()
            self.scheduler.step(valid_loss)
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            progress_bar.set_description("".join(epoch_summary))
            es_metric = list(self.history.values())[1][-1]
            if self.es_mode == 'min':
                if es_metric < best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            else:
                if es_metric > best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            if(self.es.step(es_metric)):
                print('Early stopping triggered!')
                break
        self.training_time = time.time() - self.training_time
        self.save_hist()
        self.load_model()