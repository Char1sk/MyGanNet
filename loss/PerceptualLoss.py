import torch
import torch.nn as nn
from torch import Tensor


class PerceptualLoss(nn.Module):
    
    def __init__(self, vgg_model:nn.Module, n_layers:int):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model.features
        
        # use relu_1_1, 2_1, 3_1, 4_1
        if n_layers == 3:
            self.use_layer = set(['2', '25', '29'])
        elif n_layers == 2:
            self.use_layer = set(['2', '25'])
        # self.use_layer = set(['2', '9', '16', '29'])
        self.mse = torch.nn.MSELoss()
    
    def forward(self, pred:Tensor, label:Tensor):
        loss = 0
        
        for name, module in self.vgg_layers._modules.items():
            pred, label = module(pred), module(label)
            if name in self.use_layer:
                # s = Variable(s.data, requires_grad=False)
                loss += self.mse(pred, label)
        
        return loss
