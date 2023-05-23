import os
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as tf
import numpy as np
from torch.utils.data.dataloader import DataLoader

from data.dataset import MyDataset
from models.MyGanModel import MyGanModel
from models.MyGanModel import MyInferenceModel
from options.eval_options import EvalOptions
from utils.fid_score import get_fid, get_folders_from_list, get_paths_from_list
from utils.fsim_score import get_file_list_from_file_list, get_file_list_from_folder, eval_avg_fsim, cal_fsim_with_tensors

from ptflops import get_model_complexity_info


def main():
    opts = EvalOptions().parse()
    if opts.metric.lower() == 'fid':
        path_label_test = tuple(get_paths_from_list(opts.data_folder, opts.test_photo_list))
        path_pred_test = opts.pred_folder
        fid = get_fid(path_label_test, path_pred_test, path=opts.inception_model)
        print(f"FID: {fid:>6.2f}")
        
    elif opts.metric.lower() == 'fsim':
        l1 = get_file_list_from_file_list(opts.data_folder, opts.test_photo_list)
        l2 = get_file_list_from_folder(opts.pred_folder)
        fsim = eval_avg_fsim(l1, l2)
        print(f"FSIM: {fsim:>6.4f}")
        
    elif opts.metric.lower() == 'flops':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = MyInferenceModel(opts, device)
        flops, params = get_model_complexity_info(net, (1,250,200))
        print("flops:", flops)
        print("params:", params)
        
    elif opts.metric.lower() == 'time':
        if os.path.exists('./temp.txt'):
            os.remove('./temp.txt')
        for i in range(100):
            print(i)
            os.system(f'python .\\time_test.py --architecture {opts.architecture} {"--no_global" if opts.no_global else ""} {"--no_local" if opts.no_local else ""}')
        with open('./temp.txt', 'r') as f:
            times = [float(l.strip()) for l in f.readlines()]
            print(times)
            print(np.mean(times))
            
    elif opts.metric.lower() == 'robust':
        
        if opts.process == 'blur':
            kernel_size = 21
            inputTrans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.6022, 0.4003),
                transforms.GaussianBlur(kernel_size),
            ])
            labelTrans = None
            saveDir = f'./Saves/Images/{opts.process}/{kernel_size}'
        elif opts.process == 'shift':
            useTrain = True
            ho, ve = 0.00, 0.10
            inputTrans = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(0, translate=(ho, ve), fill=1),
                transforms.Normalize(0.6022, 0.4003),
            ])
            labelTrans = inputTrans
            saveDir = f'./Saves/Images/{opts.process}/{ho}_{ve}/{"train" if useTrain else "test"}'
        elif opts.process == 'rotate':
            degree = 10
            inputTrans = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=degree, fill=1),
                transforms.Normalize(0.6022, 0.4003),
            ])
            saveDir = f'./Saves/Images/{opts.process}/{degree}'
        elif opts.process == 'scale':
            smin, smax = 0.8, 1.2
            inputTrans = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(0, scale=(smin,smax)),
                transforms.Normalize(0.6022, 0.4003),
            ])
            saveDir = f'./Saves/Images/{opts.process}/{smin}_{smax}'
        
        trainSet = MyDataset(opts, True, inputTrans, labelTrans)
        trainLoader = DataLoader(dataset=trainSet, batch_size=1, shuffle=False)
        testSet = MyDataset(opts, False, inputTrans, labelTrans)
        testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
        
        # if True: ###########################################
        #     for (i, data) in enumerate(testLoader, 1):
        #         inputs, labels = data
                
        #         inputs = transforms.ToPILImage()(inputs.squeeze(0))
        #         labels = transforms.ToPILImage()(labels.squeeze(0))
        #         inputs.show()
        #         labels.show()
        #         input()
        #     return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gan = MyGanModel(opts, False, device)
        gan.load_models('./Saves/Checkpoints/SE_global/350/')
        gan.set_models_train(False)
        
        with torch.no_grad():
            # saveDir = f'./Saves/Images/{opts.process}/{value}'
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            
            fsim = 0.0
            for (i, data) in enumerate( trainLoader if useTrain else testLoader, 1):
                inputs, labels = [d.to(device) for d in data]
                preds = gan.do_forward(inputs)
                fsim += cal_fsim_with_tensors(labels, preds)
                torchvision.utils.save_image(preds, f'{saveDir}/{i}.jpg')
                if i == 20:
                    return
            fsim /= len(testLoader)
            fid = get_fid(saveDir, tuple(get_paths_from_list(opts.data_folder, opts.test_photo_list)), path=opts.inception_model)
            print(f"FID: {fid:>9.5f}; FSIM: {fsim:>9.5f};")
        
    else:
        print(f"Metric {opts.metric} is not defined.")


if __name__ == '__main__':
    main()
