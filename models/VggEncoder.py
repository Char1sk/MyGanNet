import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models


class MyVggEncoder(nn.Module):
    '''
    VGGEncoder
    
    part of VGG19 (through relu_4_1)
    
    ref:
    https://arxiv.org/pdf/1703.06868.pdf (sec. 6)
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    '''

    def __init__(self, model_file:str, batch_norm:bool=True) -> None:
        super(MyVggEncoder, self).__init__()
        
        VGG_TYPE = 'vgg19_bn' if batch_norm else 'vgg19'
        conf = models.vgg.cfgs['E'][:12]  # VGG through relu_4_1
        self.features = models.vgg.make_layers(conf, batch_norm=batch_norm)
        self.load_state_dict(torch.load(model_file))
    
    def forward(self, x:Tensor) -> Tensor:
        return self.features(x)


# def makeVGGEncoder(model_file, batch_norm=True):
    
#     VGG_TYPE = 'vgg19_bn' if batch_norm else 'vgg19'
    
#     enc = VGGEncoder(batch_norm)
#     enc.load_state_dict(torch.load(model_file))
    
#     return enc
