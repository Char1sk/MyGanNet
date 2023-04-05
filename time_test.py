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
    
    # Inference
    times = 100
    total_time = 0.0
    for (i,data) in enumerate(testLoader):
        beg = time.time()
        inputs, _ = [d.to(device) for d in data]
        gan.do_forward(inputs)
        end = time.time()
        total_time += end-beg
        if i == times:
            break
    print(total_time/times)
    


if __name__ == "__main__":
    main()
