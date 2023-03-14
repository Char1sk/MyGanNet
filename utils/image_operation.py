import torch
from torch import Tensor
from typing import Tuple


def partition_image(img:Tensor, h_ratio:float, w_ratio:float) -> Tuple[Tensor, Tensor, Tensor]:
    # only works for C*H*W or B*C*H*W tensors
    h, w = img.shape[-2:]
    ht, wl = int(h*h_ratio), int(w*w_ratio)
    if img.dim() == 3:
        img_tl = img[:, :ht, :wl]
        img_tr = img[:, :ht, wl:]
        img_d = img[:, ht:, :]
    elif img.dim() == 4:
        img_tl = img[:, :, :ht, :wl]
        img_tr = img[:, :, :ht, wl:]
        img_d = img[:, :, ht:, :]
    else:
        print("DIM NOT CORRECT")
    return (img_tl, img_tr, img_d)


def concat_image(img_tl:Tensor, img_tr:Tensor, img_d:Tensor) -> Tensor:
    # only works for B*C*H*W tl,tr,d tensors
    img_t = torch.cat([img_tl, img_tr], dim=3)
    img = torch.cat([img_t, img_d], dim=2)
    return img