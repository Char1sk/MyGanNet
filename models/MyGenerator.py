import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple


class MyGenerator(nn.Module):
    def __init__(self, in_nc:int, out_nc:int, num_downs:int=8, g_nc:int=64, norm_layer:nn.Module=nn.BatchNorm2d, architecture:str='SE') -> None:
        super(MyGenerator, self).__init__()
        increasing_times = 3 if num_downs-1>=3 else num_downs-1
        nc = 2**increasing_times * g_nc
        # innermost
        model = None
        # nc not increasing
        for _ in range(num_downs-increasing_times-1):
            model = get_unet_block(nc, nc, nc, is_inner_most=(model is None),
                                sub_module=model, norm_layer=norm_layer, architecture=architecture)
        # nc increasing
        for _ in range(increasing_times):
            model = get_unet_block(nc//2, nc, nc//2, is_inner_most=(model is None),
                                sub_module=model, norm_layer=norm_layer, architecture=architecture)
            nc //= 2
        # outermost
        model = get_unet_block(in_nc, g_nc, out_nc, is_outer_most=True,
                            sub_module=model, norm_layer=norm_layer, architecture=architecture)
        
        self.model = model
    
    def forward(self, *x:Tuple[Tensor]) -> Tensor:
        return self.model(*x)


def get_unet_block(in_nc:int, sub_in_nc:int, out_nc:int,
                    is_outer_most:bool=False, is_inner_most:bool=False,
                    sub_module:nn.Module=None, norm_layer:nn.Module=nn.BatchNorm2d,
                    architecture:str='SE') -> nn.Module:
    if architecture == 'SE':
        return MyUnetBlock(in_nc, sub_in_nc, out_nc, is_outer_most, is_inner_most, sub_module, norm_layer)
    elif architecture == 'DE':
        return MyDEUnetBlock(in_nc, sub_in_nc, out_nc, is_outer_most, is_inner_most, sub_module, norm_layer)
    elif architecture == 'TE':
        return MyTEUnetBlock(in_nc, sub_in_nc, out_nc, is_outer_most, is_inner_most, sub_module, norm_layer)
    else:
        return None


class MyTEUnetBlock(nn.Module):
    def __init__(self, in_nc:int, sub_in_nc:int, out_nc:int,
                    is_outer_most:bool=False, is_inner_most:bool=False,
                    sub_module:nn.Module=None, norm_layer:nn.Module=nn.BatchNorm2d) -> None:
        super(MyTEUnetBlock, self).__init__()
        self.is_outer_most = is_outer_most
        self.is_inner_most = is_inner_most
        if is_outer_most:
            # [C] - SUB - [R C T]
            self.down_tl = nn.Sequential(
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.down_tr = nn.Sequential(
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.down_d  = nn.Sequential(
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2*sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
            self.sub_module = sub_module
            # self.model = nn.Sequential(down, sub_module, up)
        elif is_inner_most:
            # [R C] - [R C N]
            self.down_tl = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.down_tr = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.down_d  = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(out_nc)
            )
            # self.model = nn.Sequential(down, up)
        else:
            # [R C N] - SUB - [R C N (D)]
            self.down_tl = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(sub_in_nc)
            )
            self.down_tr = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(sub_in_nc)
            )
            self.down_d  = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(sub_in_nc)
            )
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2*sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(out_nc)
            )
            self.sub_module = sub_module
            # self.model = nn.Sequential(down, sub_module, up)
    
    def forward(self, xl:Tensor, xr:Tensor, xd:Tensor) -> Tensor:
        if self.is_outer_most:
            x = torch.cat([xl, xr], dim=3)
            x = torch.cat([x, xd], dim=2)
            
            xl = self.down_tl(xl)
            xr = self.down_tr(xr)
            xd = self.down_d(xd)
            
            xx = self.sub_module(xl, xr, xd)
            
            fx = self.up(xx)
            fx = (fx+1) / 2
            fx = shape_pad_around(x, fx)
            return fx
        
        elif self.is_inner_most:
            x = torch.cat([xl, xr], dim=3)
            x = shape_pad_dim(xd, x, 3)
            x = torch.cat([x, xd], dim=2)
            
            xl = self.down_tl(xl)
            xr = self.down_tr(xr)
            xd = self.down_d(xd)
            
            xx = torch.cat([xl, xr], dim=3)
            xx = torch.cat([xx, xd], dim=2)
            
            fx = self.up(xx)
            fx = shape_pad_around(x, fx)
            return torch.cat([x, fx], dim=1)
        
        else:
            x = torch.cat([xl, xr], dim=3)
            x = shape_pad_dim(xd, x, 3)
            x = torch.cat([x, xd], dim=2)
            
            xl = self.down_tl(xl)
            xr = self.down_tr(xr)
            xd = self.down_d(xd)
            
            xx = self.sub_module(xl, xr, xd)
            
            fx = self.up(xx)
            fx = shape_pad_around(x, fx)
            return torch.cat([x, fx], dim=1)


class MyDEUnetBlock(nn.Module):
    def __init__(self, in_nc:int, sub_in_nc:int, out_nc:int,
                    is_outer_most:bool=False, is_inner_most:bool=False,
                    sub_module:nn.Module=None, norm_layer:nn.Module=nn.BatchNorm2d) -> None:
        super(MyDEUnetBlock, self).__init__()
        self.is_outer_most = is_outer_most
        self.is_inner_most = is_inner_most
        if is_outer_most:
            # [C] - SUB - [R C T]
            self.down_l = nn.Sequential(
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.down_r = nn.Sequential(
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2*sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
            self.sub_module = sub_module
            # self.model = nn.Sequential(down, sub_module, up)
        elif is_inner_most:
            # [R C] - [R C N]
            self.down_l = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.down_r = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(out_nc)
            )
            # self.model = nn.Sequential(down, up)
        else:
            # [R C N] - SUB - [R C N (D)]
            self.down_l = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(sub_in_nc)
            )
            self.down_r = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(sub_in_nc)
            )
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2*sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(out_nc)
            )
            self.sub_module = sub_module
            # self.model = nn.Sequential(down, sub_module, up)
    
    def forward(self, xl:Tensor, xr:Tensor) -> Tensor:
        if self.is_outer_most:
            x = torch.cat([xl, xr], dim=3)
            
            xl = self.down_l(xl)
            xr = self.down_r(xr)
            
            xu = self.sub_module(xl, xr)
            
            fx = self.up(xu)
            fx = (fx+1) / 2
            fx = shape_pad_around(x, fx)
            return fx
        
        elif self.is_inner_most:
            x = torch.cat([xl, xr], dim=3)
            
            xl = self.down_l(xl)
            xr = self.down_r(xr)
            
            xu = torch.cat([xl, xr], dim=3)
            
            fx = self.up(xu)
            fx = shape_pad_around(x, fx)
            return torch.cat([x, fx], dim=1)
        
        else:
            x = torch.cat([xl, xr], dim=3)
            
            xl = self.down_l(xl)
            xr = self.down_r(xr)
            
            xu = self.sub_module(xl, xr)
            
            fx = self.up(xu)
            fx = shape_pad_around(x, fx)
            return torch.cat([x, fx], dim=1)


class MyUnetBlock(nn.Module):
    def __init__(self, in_nc:int, sub_in_nc:int, out_nc:int,
                    is_outer_most:bool=False, is_inner_most:bool=False,
                    sub_module:nn.Module=None, norm_layer:nn.Module=nn.BatchNorm2d) -> None:
        super(MyUnetBlock, self).__init__()
        self.is_outer_most = is_outer_most
        if is_outer_most:
            # [C] - SUB - [R C T]
            down = nn.Sequential(
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2*sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
            self.model = nn.Sequential(down, sub_module, up)
        elif is_inner_most:
            # [R C] - [R C N]
            down = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1)
            )
            up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(out_nc)
            )
            self.model = nn.Sequential(down, up)
        else:
            # [R C N] - SUB - [R C N (D)]
            down = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, sub_in_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(sub_in_nc)
            )
            up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(2*sub_in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                norm_layer(out_nc)
            )
            self.model = nn.Sequential(down, sub_module, up)
    
    def forward(self, x:Tensor) -> Tensor:
        if self.is_outer_most:
            fx = (self.model(x)+1) / 2
            fx = shape_pad_around(x, fx)
            return fx
        else:
            fx = self.model(x)
            fx = shape_pad_around(x, fx)
            return torch.cat([x, fx], dim=1)


class MyCombiner(nn.Module):
    def __init__(self, in_nc:int, out_nc:int, num_res:int=2, g_nc:int=64, norm_layer:nn.Module=nn.BatchNorm2d) -> None:
        super(MyCombiner, self).__init__()
        # Not RelectionPad
        
        first = nn.Sequential(
            nn.Conv2d(in_nc, g_nc, kernel_size=7, padding=3),
            norm_layer(g_nc),
            nn.ReLU(True)
        )
        
        mid = []
        for _ in range(num_res):
            mid += [MyResBlock(g_nc)]
        mid = nn.Sequential(*mid)
        
        last = nn.Sequential(
            nn.Conv2d(g_nc, out_nc, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        self.model = nn.Sequential(first, mid, last)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)


class MyResBlock(nn.Module):
    def __init__(self, nc:int, norm_layer:nn.Module=nn.BatchNorm2d) -> None:
        super(MyResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            norm_layer(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            norm_layer(nc)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return x + self.model(x)


def shape_pad_around(x:Tensor, fx:Tensor) -> Tensor:
    # assume that x>=fx in shape, x is BCHW
    if x.shape != fx.shape:
        pt = (x.size(2) - fx.size(2)) // 2
        pd = x.size(2) - fx.size(2) - pt
        pl = (x.size(3) - fx.size(3)) // 2
        pr = x.size(3) - fx.size(3) - pl
        return F.pad(fx, [pl, pr, pt, pd])
    else:
        return fx


def shape_pad_dim(x:Tensor, fx:Tensor, dim:int) -> Tensor:
    # assume that x>=fx in shape dim, x is BCHW
    if x.size(dim) != fx.size(dim):
        pl, pr, pt, pd = 0, 0, 0, 0
        if dim == 3:
            pl = (x.size(3) - fx.size(3)) // 2
            pr = x.size(3) - fx.size(3) - pl
        return F.pad(fx, [pl, pr, pt, pd])
    else:
        return fx
