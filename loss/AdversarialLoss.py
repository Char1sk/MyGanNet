import torch
import torch.nn as nn
from torch import Tensor


class AdversarialLoss(nn.Module):
    
    def __init__(self, device:str, useBCE:bool) -> None:
        super(AdversarialLoss, self).__init__()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None
        if useBCE:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.MSELoss()
        
        self.device = device
    
    def forward(self, input:Tensor, target_is_real:bool) -> Tensor:
        target_tensor = self.get_target_tensor(input, target_is_real)
        target_tensor = target_tensor.to(self.device)
        return self.loss(input, target_tensor)
    
    def get_target_tensor(self, input:Tensor, target_is_real:bool) -> Tensor:
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = torch.FloatTensor(input.size()).fill_(self.real_label)
                # real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = torch.FloatTensor(input.size()).fill_(self.fake_label)
                # fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor
