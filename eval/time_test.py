import time

import torch
from torch.utils.data.dataloader import DataLoader

from data.dataset import MyDataset
from options.train_options import TrainOptions
from models.MyGanModel import MyGanModel


def main():
    # Options
    opt = TrainOptions().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    
    # Data
    testSet  = MyDataset(opt, False)
    testLoader  = DataLoader(dataset=testSet,  batch_size=1, shuffle=False)
    
    # Model
    gan = MyGanModel(opt, isTrain=True, device=device)
    
    beg = time.time()
    # Inference
    for data in testLoader:
        inputs, _ = [d.to(device) for d in data]
        gan.do_forward(inputs)
        break
    end = time.time()
    print(beg, end, end-beg)
    


if __name__ == "__main__":
    main()
