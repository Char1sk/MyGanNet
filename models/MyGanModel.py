import os
import argparse

import torch
from torch import Tensor
import torch.nn as nn

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
        
        self.G_list = []
        if not self.opt.no_global:
            self.G_global = MyGenerator(opt.input_nc, opt.output_nc, num_downs=7).to(device).apply(weights_init)
            self.G_list += [self.G_global]
        if not self.opt.no_local:
            if self.opt.architecture == 'SE':
                self.G_local_tl = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
                self.G_local_tr = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
                self.G_local_d  = MyGenerator(opt.input_nc, opt.output_nc, num_downs=self.opt.ld_layer).to(device).apply(weights_init)
                self.G_list += [self.G_local_tl, self.G_local_tr, self.G_local_d]
            elif self.opt.architecture == 'DE':
                self.G_local_t = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4, architecture='DE').to(device).apply(weights_init)
                self.G_local_d = MyGenerator(opt.input_nc, opt.output_nc, num_downs=self.opt.ld_layer).to(device).apply(weights_init)
                self.G_list += [self.G_local_t, self.G_local_d]
            elif self.opt.architecture == 'TE':
                self.G_local = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4, architecture='TE').to(device).apply(weights_init)
                self.G_list += [self.G_local]
        if not self.opt.no_global and not self.opt.no_local:
            self.G_combiner = MyCombiner(2*opt.output_nc, opt.output_nc, num_res=self.opt.cb_layer).to(device).apply(weights_init)
            self.G_list += [self.G_combiner]
        
        
        if self.isTrain:
            self.D_list = []
            if not self.opt.no_global:
                self.D_global = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                self.D_list += [self.D_global]
            if not self.opt.no_local:
                if self.opt.architecture == 'SE' or self.opt.all_D or self.opt.final_parts:
                    self.D_local_tl = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                    self.D_local_tr = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                    self.D_local_d  = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                    self.D_list += [self.D_local_tl, self.D_local_tr, self.D_local_d]
                elif self.opt.architecture == 'DE':
                    self.D_local_t = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                    self.D_local_d = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                    self.D_list += [self.D_local_t, self.D_local_d]
                elif self.opt.architecture == 'TE':
                    self.D_local = MyPatchDiscriminator(opt.input_nc+opt.output_nc, use_sigmoid=opt.use_BCE).to(device).apply(weights_init)
                    self.D_list += [self.D_local]
            
            VGG = MyVggEncoder(opt.vgg_model).to(device)
            self.criterion_adv = AdversarialLoss(device, opt.use_BCE)
            self.criterion_pxl = PixelMatchLoss(device, opt.pxl_loss)
            if not self.opt.no_perceptual:
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
        
        if not self.opt.no_global:
            self.pred_global = self.G_global(x)
            self.pred = self.pred_global
        if not self.opt.no_local:
            if self.opt.architecture == 'SE':
                pred_local_tl = self.G_local_tl(x_tl)
                pred_local_tr = self.G_local_tr(x_tr)
                pred_local_d  = self.G_local_d(x_d)
                self.pred_local = concat_image(pred_local_tl, pred_local_tr, pred_local_d)
            elif self.opt.architecture == 'DE':
                pred_local_t  = self.G_local_t(x_tl, x_tr)
                pred_local_d  = self.G_local_d(x_d)
                self.pred_local = torch.cat([pred_local_t, pred_local_d], dim=2)
            elif self.opt.architecture == 'TE':
                self.pred_local = self.G_local(x_tl, x_tr, x_d)
            self.pred = self.pred_local
        if not self.opt.no_global and not self.opt.no_local:
            self.pred = self.G_combiner(torch.cat([self.pred_global, self.pred_local], dim=1))
        
        return self.pred
    
    
    def backward_D(self, input:Tensor, label:Tensor, record:LossRecord) -> None:
        '''Use results of do_forward to calculate loss & gradient of D'''
        input_parts, real_parts, fake_parts = [], [], []
        
        if not self.opt.no_global:
            input_parts += [input]
            real_parts  += [label]
            fake_parts  += [self.pred]
        if not self.opt.no_local:
            input_parts += partition_image(input, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
            real_parts  += partition_image(label, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
            if self.opt.final_parts:
                fake_parts += partition_image(self.pred, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
            else:
                fake_parts += partition_image(self.pred_local, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
        
        loss_D_fake = 0.0
        loss_D_real = 0.0
        for i in range(len(real_parts)):
            fake_pair = torch.cat([input_parts[i], fake_parts[i]], dim=1)
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
        input_parts, real_parts, fake_parts = [], [], []
        
        if not self.opt.no_global:
            input_parts += [input]
            real_parts  += [label]
            fake_parts  += [self.pred]
        if not self.opt.no_local:
            input_parts += partition_image(input, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
            real_parts  += partition_image(label, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
            if self.opt.final_parts:
                fake_parts += partition_image(self.pred, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
            else:
                fake_parts += partition_image(self.pred_local, self.opt.h_ratio, self.opt.w_ratio, self.opt.architecture, self.opt.all_D, self.opt.final_parts)
        
        loss_G_adv = 0.0
        loss_G_pxl = 0.0
        loss_G_per = 0.0
        for i in range(len(real_parts)):
            # fake_pair = torch.cat([input_parts[i], fake_parts_final[i]], dim=1)
            fake_pair = torch.cat([input_parts[i], fake_parts[i]], dim=1)
            fake_judge = self.D_list[i](fake_pair)
            loss_adv = self.opt.delta * self.criterion_adv(fake_judge, True)
            loss_pxl = self.opt.lamda * self.criterion_pxl(fake_parts[i], real_parts[i])
            loss_per = self.opt.gamma * self.criterion_per(fake_parts[i], real_parts[i]) if not self.opt.no_perceptual else torch.tensor(0.0)
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
        set_requires_grad(self.D_list, True)
        self.optim_D.zero_grad()
        self.backward_D(input, label, record)
        self.optim_D.step()
        # optimize G
        set_requires_grad(self.D_list, False)
        self.optim_G.zero_grad()
        self.backward_G(input, label, record)
        self.optim_G.step()
    
    
    def set_models_train(self, is_train:bool) -> None:
        set_models_eval(self.G_list, is_train)
        if self.isTrain:
            set_models_eval(self.D_list, is_train)
    
    
    def save_models(self, folder:str, epoch:int) -> None:
        folder = os.path.join(folder, str(epoch))
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Save G
        if not self.opt.no_global:
            G_global_path = os.path.join(folder, f'G_global.weight')
            torch.save(self.G_global.state_dict(),   G_global_path)
        if not self.opt.no_local:
            if self.opt.architecture == 'SE':
                G_local_tl_path = os.path.join(folder, f'G_local_tl.weight')
                G_local_tr_path = os.path.join(folder, f'G_local_tr.weight')
                G_local_d_path  = os.path.join(folder, f'G_local_d.weight')
                torch.save(self.G_local_tl.state_dict(), G_local_tl_path)
                torch.save(self.G_local_tr.state_dict(), G_local_tr_path)
                torch.save(self.G_local_d.state_dict(),  G_local_d_path)
            elif self.opt.architecture == 'DE':
                G_local_t_path = os.path.join(folder, f'G_local_t.weight')
                G_local_d_path = os.path.join(folder, f'G_local_d.weight')
                torch.save(self.G_local_t.state_dict(), G_local_t_path)
                torch.save(self.G_local_d.state_dict(), G_local_d_path)
            elif self.opt.architecture == 'TE':
                G_local_path  = os.path.join(folder, f'G_local.weight')
                torch.save(self.G_local.state_dict(),  G_local_path)
        if not self.opt.no_global and not self.opt.no_local:
            G_combiner_path = os.path.join(folder, f'G_combiner.weight')
            torch.save(self.G_combiner.state_dict(), G_combiner_path)
        
        # Save D
        if not self.opt.no_global:
            D_global_path = os.path.join(folder, f'D_global.weight')
            torch.save(self.D_global.state_dict(),   D_global_path)
        if not self.opt.no_local:
            if self.opt.architecture == 'SE' or self.opt.all_D or self.opt.final_parts:
                D_local_tl_path = os.path.join(folder, f'D_local_tl.weight')
                D_local_tr_path = os.path.join(folder, f'D_local_tr.weight')
                D_local_d_path  = os.path.join(folder, f'D_local_d.weight')
                torch.save(self.D_local_tl.state_dict(), D_local_tl_path)
                torch.save(self.D_local_tr.state_dict(), D_local_tr_path)
                torch.save(self.D_local_d.state_dict(),  D_local_d_path)
            elif self.opt.architecture == 'DE':
                D_local_t_path = os.path.join(folder, f'D_local_t.weight')
                D_local_d_path = os.path.join(folder, f'D_local_d.weight')
                torch.save(self.D_local_t.state_dict(), D_local_t_path)
                torch.save(self.D_local_d.state_dict(), D_local_d_path)
            elif self.opt.architecture == 'TE':
                D_local_path  = os.path.join(folder, f'D_local.weight')
                torch.save(self.D_local.state_dict(), D_local_path)
    
    
    def load_models(self, folder:str) -> None:
        # Load G
        if not self.opt.no_global:
            G_global_path = os.path.join(folder, f'G_global.weight')
            self.G_global.load_state_dict(torch.load(G_global_path))
        if not self.opt.no_local:
            if self.opt.architecture == 'SE':
                G_local_tl_path = os.path.join(folder, f'G_local_tl.weight')
                G_local_tr_path = os.path.join(folder, f'G_local_tr.weight')
                G_local_d_path  = os.path.join(folder, f'G_local_d.weight')
                self.G_local_tl.load_state_dict(torch.load(G_local_tl_path))
                self.G_local_tr.load_state_dict(torch.load(G_local_tr_path))
                self.G_local_d.load_state_dict( torch.load(G_local_d_path))
            elif self.opt.architecture == 'DE':
                G_local_t_path = os.path.join(folder, f'G_local_t.weight')
                G_local_d_path = os.path.join(folder, f'G_local_d.weight')
                self.G_local_t.load_state_dict(torch.load(G_local_t_path))
                self.G_local_d.load_state_dict(torch.load(G_local_d_path))
            elif self.opt.architecture == 'TE':
                G_local_path = os.path.join(folder, f'G_local.weight')
                self.G_local.load_state_dict(torch.load(G_local_path))
        if not self.opt.no_global and not self.opt.no_local:
            G_combiner_path = os.path.join(folder, f'G_combiner.weight')
            self.G_combiner.load_state_dict(torch.load(G_combiner_path))
            
        
        # Load D
        if self.isTrain:
            if not self.opt.no_global:
                D_global_path   = os.path.join(folder, f'D_global.weight')
                self.D_global.load_state_dict(torch.load(D_global_path))
            if not self.opt.no_local:
                if self.opt.architecture == 'SE' or self.opt.all_D or self.opt.final_parts:
                    D_local_tl_path = os.path.join(folder, f'D_local_tl.weight')
                    D_local_tr_path = os.path.join(folder, f'D_local_tr.weight')
                    D_local_d_path  = os.path.join(folder, f'D_local_d.weight')
                    self.D_local_tl.load_state_dict(torch.load(D_local_tl_path))
                    self.D_local_tr.load_state_dict(torch.load(D_local_tr_path))
                    self.D_local_d.load_state_dict(torch.load(D_local_d_path))
                elif self.opt.architecture == 'DE':
                    D_local_t_path = os.path.join(folder, f'D_local_t.weight')
                    D_local_d_path = os.path.join(folder, f'D_local_d.weight')
                    self.D_local_t.load_state_dict(torch.load(D_local_t_path))
                    self.D_local_d.load_state_dict(torch.load(D_local_d_path))
                elif self.opt.architecture == 'TE':
                    D_local_path = os.path.join(folder, f'D_local.weight')
                    self.D_local.load_state_dict(torch.load(D_local_path))


class MyInferenceModel(nn.Module):
    
    def __init__(self, opt:argparse.Namespace, device:str) -> None:
        super(MyInferenceModel, self).__init__()
        
        self.opt = opt
        
        self.G_global   = MyGenerator(opt.input_nc, opt.output_nc, num_downs=7).to(device).apply(weights_init)
        self.G_combiner = MyCombiner(2*opt.output_nc, opt.output_nc).to(device).apply(weights_init)
        if self.opt.architecture == 'SE':
            self.G_local_tl = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
            self.G_local_tr = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
            self.G_local_d  = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
            self.G_list = [self.G_global, self.G_local_tl, self.G_local_tr, self.G_local_d, self.G_combiner]
        elif self.opt.architecture == 'DE':
            self.G_local_t = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4, architecture='DE').to(device).apply(weights_init)
            self.G_local_d = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4).to(device).apply(weights_init)
            self.G_list = [self.G_global, self.G_local_t, self.G_local_d, self.G_combiner]
        elif self.opt.architecture == 'TE':
            self.G_local = MyGenerator(opt.input_nc, opt.output_nc, num_downs=4, architecture='TE').to(device).apply(weights_init)
            self.G_list = [self.G_global, self.G_local, self.G_combiner]
    
    def forward(self, x:Tensor) -> Tensor:
        x_tl, x_tr, x_d = partition_image(x, self.opt.h_ratio, self.opt.w_ratio)
        
        pred_global = self.G_global(x)
        if self.opt.architecture == 'SE':
            pred_local_tl = self.G_local_tl(x_tl)
            pred_local_tr = self.G_local_tr(x_tr)
            pred_local_d  = self.G_local_d(x_d)
            pred_local = concat_image(pred_local_tl, pred_local_tr, pred_local_d)
        elif self.opt.architecture == 'DE':
            pred_local_t  = self.G_local_t(x_tl, x_tr)
            pred_local_d  = self.G_local_d(x_d)
            pred_local = torch.cat([pred_local_t, pred_local_d], dim=2)
        elif self.opt.architecture == 'TE':
            pred_local = self.G_local(x_tl, x_tr, x_d)
        pred = self.G_combiner(torch.cat([pred_global, pred_local], dim=1))
        
        return pred
