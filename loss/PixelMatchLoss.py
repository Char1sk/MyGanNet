import torch
import torch.nn as nn


class PixelMatchLoss(nn.Module):
    def __init__(self, device, cri) -> None:
        super(PixelMatchLoss, self).__init__()
        if cri == 'L1':
            self.criterion = nn.L1Loss()
        elif cri == 'L2':
            self.criterion = nn.MSELoss()
    
    def forward(self, pred, label):
        lossGlobal = self.criterion(pred, label)
        return lossGlobal


class CompositionalLoss(nn.Module):
    def __init__(self, alpha, device):
        super(CompositionalLoss, self).__init__()
        self.alpha = alpha
        self.criterionGlobal = GlobalLoss()
        self.criterionLocal  = LocalLoss(device)
    
    def forward(self, pred, label, mats):
        lossGlobal = self.criterionGlobal(pred, label)
        lossLocal  = self.criterionLocal (pred, label, mats)
        loss = self.alpha*lossGlobal + (1-self.alpha)*lossLocal
        return loss


class GlobalLoss(nn.Module):
    def __init__(self):
        super(GlobalLoss, self).__init__()
        self.criterionL1 = torch.nn.L1Loss()
    
    def forward(self, pred, label):
        loss = self.criterionL1(pred, label)
        return loss


class LocalLoss(nn.Module):
    def __init__(self, device):
        super(LocalLoss, self).__init__()
        self.criterionL1 = torch.nn.L1Loss()
        self.device = device
    
    def forward(self, pred, label, mats):
        loss = torch.zeros((), device=self.device)
        for i in range(mats.size(1)):
            mat = mats[:,i,:,:].unsqueeze(1)
            mp, ml = self.dot_product(pred, label, mat)
            loss += self.criterionL1(mp, ml)
        return loss
    
    def dot_product(self, pred, label, mat):
        mpred  = torch.mul(pred,  mat)
        mlabel = torch.mul(label, mat)
        return mpred, mlabel
