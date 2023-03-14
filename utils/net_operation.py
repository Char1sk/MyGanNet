import torch.nn as nn
from typing import Union, List


def weights_init(m:nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_requires_grad(nets:Union[nn.Module, List[nn.Module]], requires:bool) -> None:
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires


def set_models_eval(nets:Union[nn.Module, List[nn.Module]], is_train:bool) -> None:
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            net.train(is_train)
