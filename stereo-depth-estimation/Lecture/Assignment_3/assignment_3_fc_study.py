# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
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
STUDY_NAME = "SVHN_FC.pkl"
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


# network architecture stolen from: https://openreview.net/pdf/1WvovwjA7UMnPB1oinBL.pdf

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class FC_Model(nn.Module):
    def __init__(self, input_dim, z_wide_dim, z_narrow_dim, output_dim, p_drop_in, p_drop_hidden, sigma_in,
                 sigma_hidden, num_z_layers):
        super().__init__()
        self.input_dim = input_dim
        layers = [GaussianNoise(sigma=sigma_in),
                  nn.Dropout(p=p_drop_in),
                  nn.Linear(input_dim, z_wide_dim),
                  GaussianNoise(sigma=sigma_hidden),
                  nn.Dropout(p=p_drop_hidden)]
        for i in range(num_z_layers):
            layers += [nn.Linear(z_wide_dim, z_narrow_dim, bias=False),
                       nn.ReLU(),
                       GaussianNoise(sigma=sigma_hidden),
                       nn.Dropout(p=p_drop_hidden),
                       nn.Linear(z_narrow_dim, z_wide_dim),
                       GaussianNoise(sigma=sigma_hidden),
                       nn.Dropout(p=p_drop_hidden)]
        layers += [nn.Linear(z_wide_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)
        out = self.model(x_flat)
        return out


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
    INPUT_DIM = 3072  # Immutable
    Z_WIDE_DIM = trial.suggest_int('z_wide_dim', 1000, 4000, step=1000)  # 1000 - 4000 | 1000 steps - 4 opts
    Z_NARROW_DIM = trial.suggest_int('z_narrow_dim', 100, 1000, step=300)  # 100-1000 | 300 steps - 4 opts
    OUTPUT_DIM = 10  # Immutable
    P_DROP_IN = trial.suggest_float("p_drop_in", 0.0, 0.6, step=0.3)  # 0.0 - 0.6 | 0.3 steps - 3 opts
    P_DROP_HIDDEN = trial.suggest_float("p_drop_hidden", 0.0, 0.8, step=0.4)  # 0.0 - 0.8 | 0.4 steps - 3 opts
    SIGMA_IN = trial.suggest_float("sigma_in", 0.0, 0.5, step=0.25)  # 0.0 - 0.5 | 0.25 steps - 3 opts
    SIGMA_HIDDEN = trial.suggest_float("sigma_hidden", 0.0, 1., step=0.5)  # 0.0 - 1.0 | 0.5 steps - 3 opts
    NUM_Z_LAYERS = trial.suggest_int('num_z_layers', 1, 3)

    return FC_Model(input_dim=INPUT_DIM, z_wide_dim=Z_WIDE_DIM, z_narrow_dim=Z_NARROW_DIM, output_dim=OUTPUT_DIM,
                    p_drop_in=P_DROP_IN, p_drop_hidden=P_DROP_HIDDEN, sigma_in=SIGMA_IN, sigma_hidden=SIGMA_HIDDEN,
                    num_z_layers=NUM_Z_LAYERS)


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


study = optuna.create_study(study_name="SVHN_FC", direction="maximize",
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