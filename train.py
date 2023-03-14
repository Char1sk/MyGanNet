import os

import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import MyDataset
from options.train_options import TrainOptions
from models.MyGanModel import MyGanModel
from utils.LossRecord import LossRecord
from utils.my_logger import get_logger, write_loss, log_loss
from utils.fid_score import get_fid, get_paths_from_list


def main():
    # Options
    opt = TrainOptions().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    
    # Logger/Writer
    logDir = os.path.join(opt.logs_folder, opt.log_name)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    writer = SummaryWriter(logDir)
    logger = get_logger(logDir)
    
    # Data
    trainSet = MyDataset(opt, True)
    testSet  = MyDataset(opt, False)
    trainLoader = DataLoader(dataset=trainSet, batch_size=opt.batch_size, shuffle=True)
    testLoader  = DataLoader(dataset=testSet,  batch_size=1, shuffle=False)
    train_testLoader = DataLoader(dataset=trainSet,  batch_size=1, shuffle=False)
    
    # Model
    gan = MyGanModel(opt, isTrain=True, device=device)
    
    # Print Params
    logger.info(str(opt))
    
    # Train
    logger.info('=========== Training Begin ===========')
    for epoch in range(1, opt.epochs+1):
        epoch_record =  LossRecord()
        
        ### Batch Step
        for (i, data) in enumerate(trainLoader,1):
            print(i)
            inputs, labels = [d.to(device) for d in data]
            gan.do_forward(inputs)
            gan.optimize_step(inputs, labels, epoch_record)
        
        ### Record Log
        epoch_record.div(len(trainLoader))
        log_loss(logger, epoch_record, epoch)
        write_loss(writer, epoch_record, 'train_epoch', epoch)
        
        ### Every Epoch Period: Train FID, Test Loss/FID, and Save
        if epoch >= opt.test_start and (epoch-opt.test_start) % opt.test_period == 0:
            gan.set_models_train(False)
            
            ### Train FID/Image
            logger.info('=========== Train Set ==========')
            with torch.no_grad():
                
                saveDir = os.path.join(opt.image_saves_folder, 'Train', str(epoch))
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                
                for (i, data) in enumerate(train_testLoader, 1):
                    inputs, _ = [d.to(device) for d in data]
                    preds = gan.do_forward(inputs)
                    
                    if i in opt.train_show_list:
                        writer.add_image(f'gen_photos_train/{i}', preds.squeeze(0), epoch)
                    torchvision.utils.save_image(preds, f'{saveDir}/{i}.jpg', normalize=True, scale_each=True)
                
                fid = get_fid(saveDir, get_paths_from_list(opt.data_folder, opt.train_photo_list), path=opt.inception_model)
                logger.info(f'Epoch: {epoch:>3d}; FID: {fid:>9.5f};')
                writer.add_scalar('FID/train', fid, epoch)
            
            ### Test *Loss*/FID/Image
            logger.info('=========== Test Set ==========')
            with torch.no_grad():
                
                saveDir = os.path.join(opt.image_saves_folder, 'Test', str(epoch))
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                
                for (i, data) in enumerate(testLoader, 1):
                    inputs, _ = [d.to(device) for d in data]
                    preds = gan.do_forward(inputs)
                    
                    if i in opt.test_show_list:
                        writer.add_image(f'gen_photos_test/{i}', preds.squeeze(0), epoch)
                    torchvision.utils.save_image(preds, f'{saveDir}/{i}.jpg', normalize=True, scale_each=True)
                
                fid = get_fid(saveDir, get_paths_from_list(opt.data_folder, opt.test_photo_list), path=opt.inception_model)
                logger.info(f'Epoch: {epoch:>3d}; FID: {fid:>9.5f};')
                writer.add_scalar('FID/test', fid, epoch)
            
            ### Save Models
            if opt.save_models:
                gan.save_models(opt.model_saves_folder, epoch)
            
            gan.set_models_train(True)


if __name__ == "__main__":
    main()
