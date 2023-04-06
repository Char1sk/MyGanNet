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
    beg = time.time()
    for (i,data) in enumerate(testLoader):
        inputs, _ = [d.to(device) for d in data]
        gan.do_forward(inputs)
        if i == times:
            break
    end = time.time()
    print((end-beg)/times)
    


if __name__ == "__main__":
    main()
