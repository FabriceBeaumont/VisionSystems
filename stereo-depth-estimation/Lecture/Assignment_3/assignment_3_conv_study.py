# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random as rand
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import joblib

N_EPOCHS = 250
NUM_STUDY_TRIALS = 1000  # try 1000 configs
STUDY_TIMEOUT = int(86400 * 0.8)  # or optimize for x days
STUDY_NAME = "SVHN_CONV.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {DEVICE}")


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


def get_data():
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         # legends say that these are the true values for SVHN
         transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
    transform_valid = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])

    trainset = dsets.SVHN('data', split='train', download=True, transform=transform_train)
    testset = dsets.SVHN('data', split='test', download=True, transform=transform_valid)
    BATCH_SIZE = 4096
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False)
    return train_loader, valid_loader


class Conv_Model(nn.Module):
  def __init__(self, input_channels, channels_1, kernel_1, channels_2, kernel_2, fc_1_neurons, fc_2_neurons,pool_mode, output_dim, p_drop_in, p_drop_conv, p_drop_fc):
      super().__init__()
      self.drop_in = nn.Dropout2d(p=p_drop_in)
      self.conv1 = nn.Conv2d(input_channels, channels_1, kernel_1, padding=kernel_1//2)
      self.drop_conv = nn.Dropout2d(p=p_drop_conv)
      if pool_mode == 'max':
        self.pool = nn.MaxPool2d(2, 2)
        self.out_size = 8
      elif pool_mode == 'avg':
        self.pool = nn.AvgPool2d(2,2)
        self.out_size = 8
      else:
        self.pool = lambda x: x
        self.out_size = 32
      self.conv2 = nn.Conv2d(channels_1, channels_2, kernel_2, padding=kernel_2//2)
      self.fc1 = nn.Linear(channels_2 * self.out_size * self.out_size, fc_1_neurons)
      self.drop_fc = nn.Dropout(p=p_drop_fc)
      self.fc2 = nn.Linear(fc_1_neurons, fc_2_neurons)
      self.fc3 = nn.Linear(fc_2_neurons, output_dim)

  def forward(self, x):
      x = self.drop_in(x)
      x = self.pool(F.relu(self.conv1(x)))
      x = self.drop_conv(x)
      x = self.pool(F.relu(self.conv2(x)))
      x = self.drop_conv(x)
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = self.drop_fc(x)
      x = F.relu(self.fc2(x))
      x = self.drop_fc(x)
      x = self.fc3(x)
      return x


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
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
    return weighted_losses.sum() / counts_np.sum()


def define_model(trial):
    INPUT_CHANNELS = 3  # Immutable
    CHANNELS_1 = trial.suggest_int('channels_1', 4, 10, step=2)
    KERNEL_1 = trial.suggest_int('kernel_1', 3, 11, step=2)
    CHANNELS_2 = trial.suggest_int('channels_2', 8, 20, step=2)
    KERNEL_2 = trial.suggest_int('kernel_2', 3, 11, step=2)
    FC_1_NEURONS = trial.suggest_int('fc_1_neurons', 92, 160, step=4)
    FC_2_NEURONS = trial.suggest_int('fc_2_neurons', 24, 84, step=4)
    POOL_MODE = trial.suggest_categorical('pool_mode', ['max', 'avg', 'none'])
    P_DROP_IN = trial.suggest_float("p_drop_in", 0.0, 0.6, step=0.3)
    P_DROP_CONV = trial.suggest_float("p_drop_conv", 0.0, 0.8, step=0.4)
    P_DROP_FC = trial.suggest_float("p_drop_fc", 0.0, 0.8, step=0.4)
    OUTPUT_DIM = 10  # Immutable

    return Conv_Model(input_channels=INPUT_CHANNELS, channels_1=CHANNELS_1, kernel_1=KERNEL_1, channels_2=CHANNELS_2,
                      kernel_2=KERNEL_2, fc_1_neurons=FC_1_NEURONS, fc_2_neurons=FC_2_NEURONS,pool_mode=POOL_MODE,
                      output_dim=OUTPUT_DIM, p_drop_in=P_DROP_IN, p_drop_conv=P_DROP_CONV, p_drop_fc=P_DROP_FC)


criterion = nn.CrossEntropyLoss().to(DEVICE)


def objective(trial):
    reject_randomness(2021)
    model = define_model(trial).to(DEVICE)
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data - param.data.mean()
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)  # 1e-1, 1e-2, 1e-3, 1e-4 - 4 opts
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)
    es = EarlyStopping(mode='max', patience=10)
    train_loader, valid_loader = get_data()

    for epoch in range(N_EPOCHS):
        valid_loss_list, valid_acc_list, batch_sizes_valid = [], [], []

        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            labels = labels.to(DEVICE)
            pred_logits = model(imgs.to(DEVICE))
            loss = criterion(pred_logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        for imgs, labels in valid_loader:
            with torch.no_grad():
                labels = labels.to(DEVICE)
                pred_logits = model(imgs.to(DEVICE))

                batch_sizes_valid.append(imgs.shape[0])
                pred_label_inds = pred_logits.argmax(dim=-1)
                acc = (pred_label_inds == labels).float().mean() * 100
                valid_acc_list.append(acc.item())

        avg_valid_acc = epoch_average(valid_acc_list, batch_sizes_valid)
        trial.report(avg_valid_acc, epoch)
        scheduler.step(avg_valid_acc)
        if (es.step(avg_valid_acc)):
            break

    joblib.dump(trial.study, STUDY_NAME)
    return avg_valid_acc


study = optuna.create_study(study_name="SVHN_CONV", direction="maximize",
                            sampler=TPESampler(n_startup_trials=5, multivariate=True))
study.optimize(objective, n_trials=NUM_STUDY_TRIALS, timeout=STUDY_TIMEOUT)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
