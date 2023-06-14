import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import autocast, GradScaler

from asset.utils import *

class Trainer():
    def __init__(self, model, criterion, train_loader, valid_loader, eval_metrics=None, lr=5e-4,
                 patience=5, es_mode='min', description='untitled', n_epochs=250, lr_decay=None):
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
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.5, verbose=True)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.992, last_epoch=-1) if lr_decay is None else ExponentialLR(self.optimizer, gamma=lr_decay, last_epoch=-1)
        self.es = EarlyStopping(mode=self.es_mode, patience=self.patience)
        self.history = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        self.scaler = GradScaler()
        self.progress_bar = None
        
    def inference_step(self, x_left, x_right):
        return self.model(x_left.to(self.device), x_right.to(self.device))
    
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
        i = 0
        n = len(self.train_loader)
        for (img_left, img_right), target in self.train_loader:
            i += 1
            self.progress_bar.set_postfix({f'T of [{n}]' : i}, refresh=True)
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(img_left, img_right)
                loss = self.criterion(net_out, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        return average_loss
    
    @torch.no_grad()
    def get_sample(self, mode='valid'):
        if mode == 'valid':
            self.model.eval()
            (img_left, img_right), target = next(iter(self.valid_loader))
            with autocast():
                net_out = self.inference_step(img_left, img_right) 
            self.model.train()
        else:
            (img_left, img_right), target = next(iter(self.train_loader))
            with autocast():
                (_, _, net_out) = self.inference_step(img_left, img_right)
        return (img_left, img_right), target, net_out.detach().cpu().float()
    
    def show_results(self):
        (img_left, img_right), target, net_out = self.get_sample(mode='valid')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_valid, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(inv_normalize(img_left), axes[0], title='Left Image')
        show_image(inv_normalize(img_right), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        (img_left, img_right), target, net_out = self.get_sample(mode='train')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_train, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(inv_normalize(img_left), axes[0], title='Left Image')
        show_image(inv_normalize(img_right), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        return fig_valid, fig_train

    @torch.no_grad()
    def eval_epoch(self):
        i = 0
        n = len(self.valid_loader)
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for (img_left, img_right), target in self.valid_loader:
            i += 1
            self.progress_bar.set_postfix({f'V of [{n}]' : i}, refresh=True)
            target = target.to(self.device)
            with autocast():
                net_out = self.inference_step(img_left, img_right)
                loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    with autocast():
                        epoch_metrics[key].append(metric(net_out,target).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss        
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        self.progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
        for epoch in self.progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()
            
            self.model.eval()
            valid_loss = self.eval_epoch()
            #self.scheduler.step(valid_loss)
            self.scheduler.step()
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            #self.progress_bar.set_description("".join(epoch_summary))
            
            x_data=np.linspace(0,20,num=60)
            y_data = x_data**2
            fig, ax = plt.subplots(1,1)
            ax.plot(x_data,y_data)
            plt.close('all')
            summary_dict = {key: v[-1] for (key,v) in self.history.items()}
            fig_valid, fig_train = self.show_results()
            summary_dict['training sample'] = wandb.Image(fig_train)
            summary_dict['validation sample'] = wandb.Image(fig_valid)
            wandb.log(summary_dict)
#             wandb.log({'test_fig': wandb.Image(fig), 'epoch': epoch+1})
            
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

#Trainer for the baseline model - no intermediate supervision
class TrainerBaseline():
    def __init__(self, model, criterion, train_loader, valid_loader, eval_metrics=None, lr=5e-4,
                 patience=5, es_mode='min', description='untitled', n_epochs=250, lr_decay=None):
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
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.5, verbose=True)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.992, last_epoch=-1) if lr_decay is None else ExponentialLR(self.optimizer, gamma=lr_decay, last_epoch=-1)
        self.es = EarlyStopping(mode=self.es_mode, patience=self.patience)
        self.history = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        self.scaler = GradScaler()
        self.progress_bar = None

    def inference_step(self, x_left, x_right):
        return self.model(x_left.to(self.device), x_right.to(self.device))

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
        i = 0
        n = len(self.train_loader)
        for (img_left, img_right), target in self.train_loader:
            i += 1
            self.progress_bar.set_postfix({f'T of [{n}]' : i}, refresh=True)
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(img_left, img_right)
                loss = self.criterion(net_out, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        return average_loss

    @torch.no_grad()
    def get_sample(self, mode='valid'):
        if mode == 'valid':
            self.model.eval()
            (img_left, img_right), target = next(iter(self.valid_loader))
            with autocast():
                net_out = self.inference_step(img_left, img_right) 
            self.model.train()
        else:
            (img_left, img_right), target = next(iter(self.train_loader))
            with autocast():
                net_out = self.inference_step(img_left, img_right)
        return (img_left, img_right), target, net_out.detach().cpu().float()

    def show_results(self):
        (img_left, img_right), target, net_out = self.get_sample(mode='valid')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_valid, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(inv_normalize(img_left), axes[0], title='Left Image')
        show_image(inv_normalize(img_right), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        (img_left, img_right), target, net_out = self.get_sample(mode='train')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_train, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(inv_normalize(img_left), axes[0], title='Left Image')
        show_image(inv_normalize(img_right), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        return fig_valid, fig_train

    @torch.no_grad()
    def eval_epoch(self):
        i = 0
        n = len(self.valid_loader)
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for (img_left, img_right), target in self.valid_loader:
            i += 1
            self.progress_bar.set_postfix({f'V of [{n}]' : i}, refresh=True)
            target = target.to(self.device)
            with autocast():
                net_out = self.inference_step(img_left, img_right)
                loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    with autocast():
                        epoch_metrics[key].append(metric(net_out,target).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss        

    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        self.progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
        for epoch in self.progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()

            self.model.eval()
            valid_loss = self.eval_epoch()
            #self.scheduler.step(valid_loss)
            self.scheduler.step()

            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            #self.progress_bar.set_description("".join(epoch_summary))

            x_data=np.linspace(0,20,num=60)
            y_data = x_data**2
            fig, ax = plt.subplots(1,1)
            ax.plot(x_data,y_data)
            plt.close('all')
            summary_dict = {key: v[-1] for (key,v) in self.history.items()}
            fig_valid, fig_train = self.show_results()
            summary_dict['training sample'] = wandb.Image(fig_train)
            summary_dict['validation sample'] = wandb.Image(fig_valid)
            wandb.log(summary_dict)
#             wandb.log({'test_fig': wandb.Image(fig), 'epoch': epoch+1})

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
        
#Trainer for the kitti dataset - different handling of validation steps due to necessary resizing
class TrainerKitti():
    def __init__(self, model, criterion, train_loader, valid_loader, size_y, size_x, eval_metrics=None, lr=5e-4,
                 patience=5, es_mode='min', description='untitled', n_epochs=250, lr_decay=None):
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
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.5, verbose=True)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.992, last_epoch=-1) if lr_decay is None else ExponentialLR(self.optimizer, gamma=lr_decay, last_epoch=-1)
        self.es = EarlyStopping(mode=self.es_mode, patience=self.patience)
        self.history = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        self.scaler = GradScaler()
        self.progress_bar = None
        self.size_y = size_y
        self.size_x = size_x
        
    def inference_step(self, x_left, x_right):
        return self.model(x_left.to(self.device), x_right.to(self.device))
    
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
        i = 0
        n = len(self.train_loader)
        for (img_left, img_right), target in self.train_loader:
            i += 1
            self.progress_bar.set_postfix({f'T of [{n}]' : i}, refresh=True)
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(img_left, img_right)
                loss = self.criterion(net_out, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        return average_loss
    
    @torch.no_grad()
    def get_sample(self, mode='valid'):
        if mode == 'valid':
            self.model.eval()
            (img_left, img_right), target = next(iter(self.valid_loader))
            with autocast():
                img_left, img_right = F.interpolate(img_left, size=(self.size_y,self.size_x), mode='bicubic'), F.interpolate(img_right, size=(self.size_y,self.size_x), mode='bicubic')
                net_out = self.inference_step(img_left, img_right) 
                net_out = F.interpolate(net_out, size=tuple(target.shape[-2:]), mode='bicubic')
                net_out *= target.shape[-1]/self.size_x
            self.model.train()
        else:
            (img_left, img_right), target = next(iter(self.train_loader))
            with autocast():
                (_, _, net_out) = self.inference_step(img_left, img_right)
        return (img_left, img_right), target, net_out.detach().cpu().float()
    
    def show_results(self):
        (img_left, img_right), target, net_out = self.get_sample(mode='valid')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_valid, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(np.clip(inv_normalize(img_left),0,1), axes[0], title='Left Image')
        show_image(np.clip(inv_normalize(img_right),0,1), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        (img_left, img_right), target, net_out = self.get_sample(mode='train')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_train, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(inv_normalize(img_left), axes[0], title='Left Image')
        show_image(inv_normalize(img_right), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        return fig_valid, fig_train

    @torch.no_grad()
    def eval_epoch(self):
        i = 0
        n = len(self.valid_loader)
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for (img_left, img_right), target in self.valid_loader:
            i += 1
            self.progress_bar.set_postfix({f'V of [{n}]' : i}, refresh=True)
            target = target.to(self.device)
            with autocast():
                img_left, img_right = F.interpolate(img_left, size=(self.size_y,self.size_x), mode='bicubic'), F.interpolate(img_right, size=(self.size_y,self.size_x), mode='bicubic')
                net_out = self.inference_step(img_left, img_right)
                net_out = F.interpolate(net_out, size=tuple(target.shape[-2:]), mode='bicubic')
                net_out *= target.shape[-1]/self.size_x
                loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    with autocast():
                        epoch_metrics[key].append(metric(net_out,target).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss        
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        self.progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        wandb.run.summary['valid_loss'+'_zeroshot']=self.history['valid loss'][0]
        for key in self.eval_metrics:
            wandb.run.summary[key+'_zeroshot'] = self.history[key][0]
        self.training_time = time.time()
        for epoch in self.progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()
            
            self.model.eval()
            valid_loss = self.eval_epoch()
            #self.scheduler.step(valid_loss)
            self.scheduler.step()
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            #self.progress_bar.set_description("".join(epoch_summary))
            
            x_data=np.linspace(0,20,num=60)
            y_data = x_data**2
            fig, ax = plt.subplots(1,1)
            ax.plot(x_data,y_data)
            plt.close('all')
            summary_dict = {key: v[-1] for (key,v) in self.history.items()}
            fig_valid, fig_train = self.show_results()
            summary_dict['training sample'] = wandb.Image(fig_train)
            summary_dict['validation sample'] = wandb.Image(fig_valid)
            wandb.log(summary_dict)
#             wandb.log({'test_fig': wandb.Image(fig), 'epoch': epoch+1})
            
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
        
#Trainer for the kitti dataset - different handling of validation steps due to necessary resizing
class TrainerBaselineKitti():
    def __init__(self, model, criterion, train_loader, valid_loader, size_y, size_x, eval_metrics=None, lr=5e-4,
                 patience=5, es_mode='min', description='untitled', n_epochs=250, lr_decay=None):
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
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.5, verbose=True)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.992, last_epoch=-1) if lr_decay is None else ExponentialLR(self.optimizer, gamma=lr_decay, last_epoch=-1)
        self.es = EarlyStopping(mode=self.es_mode, patience=self.patience)
        self.history = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        self.scaler = GradScaler()
        self.progress_bar = None
        self.size_y = size_y
        self.size_x = size_x
        
    def inference_step(self, x_left, x_right):
        return self.model(x_left.to(self.device), x_right.to(self.device))
    
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
        i = 0
        n = len(self.train_loader)
        for (img_left, img_right), target in self.train_loader:
            i += 1
            self.progress_bar.set_postfix({f'T of [{n}]' : i}, refresh=True)
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(img_left, img_right)
                loss = self.criterion(net_out, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        return average_loss
    
    @torch.no_grad()
    def get_sample(self, mode='valid'):
        if mode == 'valid':
            self.model.eval()
            (img_left, img_right), target = next(iter(self.valid_loader))
            with autocast():
                img_left, img_right = F.interpolate(img_left, size=(self.size_y,self.size_x), mode='bicubic'), F.interpolate(img_right, size=(self.size_y,self.size_x), mode='bicubic')
                net_out = self.inference_step(img_left, img_right) 
                net_out = F.interpolate(net_out, size=tuple(target.shape[-2:]), mode='bicubic')
                net_out *= target.shape[-1]/self.size_x
            self.model.train()
        else:
            (img_left, img_right), target = next(iter(self.train_loader))
            with autocast():
                net_out = self.inference_step(img_left, img_right)
        return (img_left, img_right), target, net_out.detach().cpu().float()
    
    def show_results(self):
        (img_left, img_right), target, net_out = self.get_sample(mode='valid')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_valid, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(np.clip(inv_normalize(img_left),0,1), axes[0], title='Left Image')
        show_image(np.clip(inv_normalize(img_right),0,1), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        (img_left, img_right), target, net_out = self.get_sample(mode='train')
        img_left, img_right, target, net_out = img_left[0], img_right[0], target[0], net_out[0]
        fig_train, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_image(inv_normalize(img_left), axes[0], title='Left Image')
        show_image(inv_normalize(img_right), axes[1], title='Right Image')
        show_disparity(target, axes[2], title='Disparity GT')
        show_disparity(net_out, axes[3], title='Disparity Pred')
        plt.close('all')
        return fig_valid, fig_train

    @torch.no_grad()
    def eval_epoch(self):
        i = 0
        n = len(self.valid_loader)
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for (img_left, img_right), target in self.valid_loader:
            i += 1
            self.progress_bar.set_postfix({f'V of [{n}]' : i}, refresh=True)
            target = target.to(self.device)
            with autocast():
                img_left, img_right = F.interpolate(img_left, size=(self.size_y,self.size_x), mode='bicubic'), F.interpolate(img_right, size=(self.size_y,self.size_x), mode='bicubic')
                net_out = self.inference_step(img_left, img_right)
                net_out = F.interpolate(net_out, size=tuple(target.shape[-2:]), mode='bicubic')
                net_out *= target.shape[-1]/self.size_x
                loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    with autocast():
                        epoch_metrics[key].append(metric(net_out,target).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss        
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        self.progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        wandb.run.summary['valid_loss'+'_zeroshot']=self.history['valid loss'][0]
        for key in self.eval_metrics:
            wandb.run.summary[key+'_zeroshot'] = self.history[key][0]
        self.training_time = time.time()
        for epoch in self.progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()
            
            self.model.eval()
            valid_loss = self.eval_epoch()
            #self.scheduler.step(valid_loss)
            self.scheduler.step()
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            #self.progress_bar.set_description("".join(epoch_summary))
            
            x_data=np.linspace(0,20,num=60)
            y_data = x_data**2
            fig, ax = plt.subplots(1,1)
            ax.plot(x_data,y_data)
            plt.close('all')
            summary_dict = {key: v[-1] for (key,v) in self.history.items()}
            fig_valid, fig_train = self.show_results()
            summary_dict['training sample'] = wandb.Image(fig_train)
            summary_dict['validation sample'] = wandb.Image(fig_valid)
            wandb.log(summary_dict)
#             wandb.log({'test_fig': wandb.Image(fig), 'epoch': epoch+1})
            
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
        
#Trainer for sky segmentation
class TrainerSky():
    def __init__(self, model, criterion, train_loader, valid_loader, features, eval_metrics=None, lr=5e-4,
                 patience=5, es_mode='min', description='untitled', n_epochs=250, lr_decay=None):
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
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.5, verbose=True)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.992, last_epoch=-1) if lr_decay is None else ExponentialLR(self.optimizer, gamma=lr_decay, last_epoch=-1)
        self.es = EarlyStopping(mode=self.es_mode, patience=self.patience)
        self.history = {'train loss': [], 'valid loss' : []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        self.scaler = GradScaler()
        self.progress_bar = None
        self.features = features.to(self.device)
        self.features.requires_grad_(False)
        self.features.eval()

    def inference_step(self, x):
        #enc = self.features()
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
        i = 0
        n = len(self.train_loader)
        for img, target in self.train_loader:
            i += 1
            self.progress_bar.set_postfix({f'T of [{n}]' : i}, refresh=True)
            self.optimizer.zero_grad()
            with autocast():
                net_out = self.inference_step(img)
                loss = self.criterion(net_out, target.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        return average_loss

    @torch.no_grad()
    def get_sample(self, mode='valid'):
        if mode == 'valid':
            self.model.eval()
            img, target = next(iter(self.valid_loader))
            with autocast():
                net_out = self.inference_step(img) 
            self.model.train()
        else:
            img, target = next(iter(self.train_loader))
            with autocast():
                net_out = self.inference_step(img)
        return img, target, net_out.detach().cpu().float()

    def show_results(self):
        img, target, net_out = self.get_sample(mode='valid')
        net_out = F.sigmoid(net_out)
        rand_ind = np.random.randint(0, img.shape[0])
        img, target, net_out = img[rand_ind], target[rand_ind], net_out[rand_ind]
        fig_valid, axes = plt.subplots(1, 3, figsize=(15, 5))
        show_image(np.clip(inv_normalize(img),0,1), axes[0], title='Image')
        axes[1].imshow(target.squeeze(), cmap='Greys', vmin=0., vmax=1.)
        axes[1].set_title('Mask GT')
        axes[1].axis('off')
        axes[2].imshow(net_out.squeeze(), cmap='Greys', vmin=0., vmax=1.)
        axes[2].set_title('Mask Pred')
        axes[2].axis('off')
        plt.close('all')
        img, target, net_out = self.get_sample(mode='train')
        net_out = F.sigmoid(net_out)
        img, target, net_out = img[0], target[0], net_out[0]
        fig_train, axes = plt.subplots(1, 3, figsize=(15, 5))
        show_image(np.clip(inv_normalize(img),0,1), axes[0], title='Image')
        axes[1].imshow(target.squeeze(), cmap='Greys', vmin=0., vmax=1.)
        axes[1].set_title('Mask GT')
        axes[1].axis('off')
        axes[2].imshow(net_out.squeeze(), cmap='Greys', vmin=0., vmax=1.)
        axes[2].set_title('Mask Pred')
        axes[2].axis('off')
        plt.close('all')
        return fig_valid, fig_train

    @torch.no_grad()
    def eval_epoch(self):
        i = 0
        n = len(self.valid_loader)
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for img, target in self.valid_loader:
            i += 1
            self.progress_bar.set_postfix({f'V of [{n}]' : i}, refresh=True)
            target = target.to(self.device)
            with autocast():
                net_out = self.inference_step(img)
                loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    with autocast():
                        epoch_metrics[key].append(metric(net_out,target).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss        

    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        self.progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()
        for epoch in self.progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()

            self.model.eval()
            valid_loss = self.eval_epoch()
            #self.scheduler.step(valid_loss)
            self.scheduler.step()

            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            #self.progress_bar.set_description("".join(epoch_summary))

            x_data=np.linspace(0,20,num=60)
            y_data = x_data**2
            fig, ax = plt.subplots(1,1)
            ax.plot(x_data,y_data)
            plt.close('all')
            summary_dict = {key: v[-1] for (key,v) in self.history.items()}
            fig_valid, fig_train = self.show_results()
            summary_dict['training sample'] = wandb.Image(fig_train)
            summary_dict['validation sample'] = wandb.Image(fig_valid)
            wandb.log(summary_dict)
#             wandb.log({'test_fig': wandb.Image(fig), 'epoch': epoch+1})

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