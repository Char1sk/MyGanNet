import os
import argparse

import torch
from torch import Tensor

from models.MyGenerator import MyGenerator, MyCombiner
from models.MyDiscriminator import MyPatchDiscriminator
from models.VggEncoder import MyVggEncoder
from loss.AdversarialLoss import AdversarialLoss
from loss.PixelMatchLoss import PixelMatchLoss
from loss.PerceptualLoss import PerceptualLoss
from utils.LossRecord import LossRecord
from utils.net_operation import weights_init, set_requires_grad, set_models_eval
from utils.image_operation import partition_image, concat_image


# NOT a nn.Module, just for closure
class MyGanModel():
    
    def __init__(self, opt:argparse.Namespace, isTrain:bool, device:str) -> None:
        
        self.opt = opt
        self.isTrain = isTrain
        
        self.G_global   = MyGenerator(opt.input_nc, opt.output_nc, num_downs=7).to(device).apply(weights_init)
        self.G_local_tl = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
        self.G_local_tr = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
        self.G_local_d  = MyGenerator(opt.input_nc, opt.output_nc, num_downs=5).to(device).apply(weights_init)
        self.G_combiner = MyCombiner(2*opt.output_nc, opt.output_nc, 3).to(device).apply(weights_init)
        self.G_list = [self.G_global, self.G_local_tl, self.G_local_tr, self.G_local_d, self.G_combiner]
        
        self.D_global   = MyPatchDiscriminator(opt.input_nc+opt.output_nc).to(device).apply(weights_init)
        self.D_local_tl = MyPatchDiscriminator(opt.input_nc+opt.output_nc).to(device).apply(weights_init)
        self.D_local_tr = MyPatchDiscriminator(opt.input_nc+opt.output_nc).to(device).apply(weights_init)
        self.D_local_d  = MyPatchDiscriminator(opt.input_nc+opt.output_nc).to(device).apply(weights_init)
        self.D_list = [self.D_global, self.D_local_tl, self.D_local_tr, self.D_local_d]
        
        VGG = MyVggEncoder(opt.vgg_model).to(device)
        self.criterion_adv = AdversarialLoss(device, opt.BCE)
        self.criterion_pxl = PixelMatchLoss(device)
        self.criterion_per = PerceptualLoss(VGG, opt.vgg_layers)
        
        G_param_list = []
        for g in self.G_list:
            G_param_list += list(g.parameters())
        D_param_list = []
        for d in self.D_list:
            D_param_list += list(d.parameters())
        self.optim_G = torch.optim.Adam(G_param_list, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optim_D = torch.optim.Adam(D_param_list, lr=opt.lr, betas=(opt.beta1, opt.beta2))
    
    
    def do_forward(self, x:Tensor) -> Tensor:
        '''Do forward using G with input x, the results are stored in this instance'''
        
        x_tl, x_tr, x_d = partition_image(x, self.opt.h_ratio, self.opt.w_ratio)
        
        self.pred_global = self.G_global(x)
        self.pred_local_tl = self.G_local_tl(x_tl)
        self.pred_local_tr = self.G_local_tr(x_tr)
        self.pred_local_d  = self.G_local_d(x_d)
        self.pred_local  = concat_image(self.pred_local_tl, self.pred_local_tr, self.pred_local_d)
        self.pred = self.G_combiner(torch.cat([self.pred_global, self.pred_local], dim=1))
        
        return self.pred
    
    
    def backward_D(self, input:Tensor, label:Tensor, record:LossRecord) -> None:
        '''Use results of do_forward to calculate loss & gradient of D'''
        
        # NOTE adv_loss calculates final image and its parts
        input_parts = (input, *partition_image(input, self.opt.h_ratio, self.opt.w_ratio))
        real_parts  = (label, *partition_image(label, self.opt.h_ratio, self.opt.w_ratio))
        # fake_parts_final = (self.pred, *partition_image(self.pred, self.opt.h_ratio, self.opt.w_ratio))
        fake_parts_inter = (self.pred, self.pred_local_tl, self.pred_local_tr, self.pred_local_d)
        
        loss_D_fake = 0.0
        loss_D_real = 0.0
        for i in range(len(real_parts)):
            fake_pair = torch.cat([input_parts[i], fake_parts_inter[i]], dim=1)
            real_pair = torch.cat([input_parts[i], real_parts[i]], dim=1)
            fake_judge = self.D_list[i](fake_pair.detach())
            real_judge = self.D_list[i](real_pair)
            loss_D_fake += self.criterion_adv(fake_judge, False)
            loss_D_real += self.criterion_adv(real_judge, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        
        loss_D.backward()
        record.D += loss_D.item()
    
    
    def backward_G(self, input:Tensor, label:Tensor, record:LossRecord, do_back:bool=True) -> None:
        '''Use results of do_forward to calculate loss & gradient of G'''
        input_parts = (input, *partition_image(input, self.opt.h_ratio, self.opt.w_ratio))
        real_parts  = (label, *partition_image(label, self.opt.h_ratio, self.opt.w_ratio))
        fake_parts_inter = (self.pred, self.pred_local_tl, self.pred_local_tr, self.pred_local_d)
        fake_parts_final = (self.pred, *partition_image(self.pred, self.opt.h_ratio, self.opt.w_ratio))
        
        loss_G_adv = 0.0
        loss_G_pxl = 0.0
        loss_G_per = 0.0
        for i in range(len(real_parts)):
            # fake_pair = torch.cat([input_parts[i], fake_parts_final[i]], dim=1)
            fake_pair = torch.cat([input_parts[i], fake_parts_inter[i]], dim=1)
            fake_judge = self.D_list[i](fake_pair)
            loss_adv = self.opt.delta * self.criterion_adv(fake_judge, True)
            loss_pxl = self.opt.lamda * self.criterion_pxl(fake_parts_inter[i], real_parts[i])
            loss_per = self.opt.gamma * self.criterion_per(fake_parts_inter[i], real_parts[i])
            loss_G_adv += loss_adv
            loss_G_pxl += loss_pxl
            loss_G_per += loss_per
            record.Gparts[i].add(loss_adv.item(), loss_pxl.item(), loss_per.item())
        loss_G = loss_G_adv + loss_G_pxl + loss_G_per
        
        if do_back:
            loss_G.backward()
        record.Gtotal.add(loss_G_adv.item(), loss_G_pxl.item(), loss_G_per.item())
    
    
    def optimize_step(self, input:Tensor, label:Tensor, record:LossRecord) -> None:
        '''Use results of do_forward to optimize D and G. This will call backward_D/G'''
        # optimize D
        set_requires_grad([self.D_global, self.D_local_tl, self.D_local_tr, self.D_local_d], True)
        self.optim_D.zero_grad()
        self.backward_D(input, label, record)
        self.optim_D.step()
        # optimize G
        set_requires_grad([self.D_global, self.D_local_tl, self.D_local_tr, self.D_local_d], False)
        self.optim_G.zero_grad()
        self.backward_G(input, label, record)
        self.optim_G.step()
    
    
    def set_models_train(self, is_train:bool) -> None:
        set_models_eval(self.G_list+self.D_list, is_train)
    
    
    def save_models(self, folder:str, epoch:int) -> None:
        if not os.path.exists(self.opt.model_saves_folder):
            os.makedirs(self.opt.model_saves_folder)
        
        # Save G
        G_global_path   = os.path.join(folder, f'G_global_epoch_{epoch}.weight')
        G_local_tl_path = os.path.join(folder, f'G_local_tl_epoch_{epoch}.weight')
        G_local_tr_path = os.path.join(folder, f'G_local_tr_epoch_{epoch}.weight')
        G_local_d_path  = os.path.join(folder, f'G_local_d_epoch_{epoch}.weight')
        G_combiner_path = os.path.join(folder, f'G_combiner_epoch_{epoch}.weight')
        torch.save(self.G_global.state_dict(),   G_global_path)
        torch.save(self.G_local_tl.state_dict(), G_local_tl_path)
        torch.save(self.G_local_tr.state_dict(), G_local_tr_path)
        torch.save(self.G_local_d.state_dict(),  G_local_d_path)
        torch.save(self.G_combiner.state_dict(), G_combiner_path)
        
        # Save D
        D_global_path   = os.path.join(folder, f'D_global_epoch_{epoch}.weight')
        D_local_tl_path = os.path.join(folder, f'D_local_tl_epoch_{epoch}.weight')
        D_local_tr_path = os.path.join(folder, f'D_local_tr_epoch_{epoch}.weight')
        D_local_d_path  = os.path.join(folder, f'D_local_d_epoch_{epoch}.weight')
        torch.save(self.D_global.state_dict(),   D_global_path)
        torch.save(self.D_local_tl.state_dict(), D_local_tl_path)
        torch.save(self.D_local_tr.state_dict(), D_local_tr_path)
        torch.save(self.D_local_d.state_dict(),  D_local_d_path)
