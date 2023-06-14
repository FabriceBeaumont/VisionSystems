import torch
import torch.nn as nn

class IntermediateSupervisionWrapper(nn.Module):
    def __init__(self, criterion, alphas=torch.tensor([.5,.7,1.])):
        super(IntermediateSupervisionWrapper, self).__init__()
        self.criterion = criterion
        self.alphas = alphas/torch.norm(alphas)
    def forward(self, x, target):
        if type(x) is tuple:
            (c1, c2, c3) = x
            loss = self.alphas[0] * self.criterion(c1, target) + self.alphas[1] * self.criterion(c2, target) + self.alphas[2] * self.criterion(c2, target)
        else:
            loss = self.criterion(x, target)
        return loss

class NonZeroWrapper(nn.Module):
    def __init__(self, criterion, eps=0, max_disp=None):
        super(NonZeroWrapper, self).__init__()
        self.criterion = criterion
        self.eps = eps
        self.max_disp = max_disp
    def forward(self, x, target):
        mask = target > self.eps
        if self.max_disp is not None:
            mask_max = torch.le(target, self.max_disp)
            mask = torch.logical_and(mask, mask_max) 
        loss = self.criterion(x, target)
        return loss[mask].mean()

class PE(nn.Module):
    def __init__(self, threshold=3, reduction='mean'):
        super(PE, self).__init__()
        self.diff = nn.L1Loss(reduction='none')
        self.threshold = threshold
        self.reduction = reduction
    def forward(self, x, target):
        err = self.diff(x, target)
        mask_err = torch.ge(err, self.threshold)
        mask_ratio = torch.ge(err, 0.05*target)
        mask = torch.logical_and(mask_err, mask_ratio)
        loss = torch.where(mask, torch.ones_like(target), torch.zeros_like(target))
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss