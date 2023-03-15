import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import Tensor

import os
import cv2

import argparse
from typing import List, Tuple

class MyDataset(data.Dataset):
    def __init__(self, opt:argparse.Namespace, isTrain:bool):
        super(MyDataset, self).__init__()
        
        self.opt = opt
        self.isTrain = isTrain
        if self.isTrain:
            self.sketchNames = getNames(os.path.join(self.opt.data_folder, self.opt.train_sketch_list))
            self.photoNames = getNames(os.path.join(self.opt.data_folder, self.opt.train_photo_list))
        else:
            self.sketchNames = getNames(os.path.join(self.opt.data_folder, self.opt.test_sketch_list))
            self.photoNames = getNames(os.path.join(self.opt.data_folder, self.opt.test_photo_list))
        assert len(self.sketchNames) == len(self.photoNames)
    
    def __getitem__(self, index:int) -> Tuple[Tensor, Tensor]:
        inputPath = os.path.join(self.opt.data_folder, self.sketchNames[index])
        labelPath = os.path.join(self.opt.data_folder, self.photoNames[index])
        
        inputs = getInputs(inputPath, self.opt.output_shape)
        labels = getLabels(labelPath, self.opt.output_shape)
        
        return (inputs, labels)
    
    def __len__(self) -> int:
        return len(self.sketchNames)


def getNames(path:str) -> List[str]:
    with open(path, 'r') as f:
        ret = [l.strip() for l in f.readlines()]
    return ret


def getInputs(path:str, shape:int, trans:transforms.Compose=None) -> torch.Tensor:
    # img: H*W numpy
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img: 1*SHAPE*SHAPE tensor [0.0, 1.0]
    shapes = getPadShape(img.shape, shape)
    if trans is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Pad(shapes),
            transforms.Normalize(0.6022, 0.4003)
        ])
    img = trans(img)
    # tensor
    return img


def getLabels(path:str, shape:int, trans:transforms.Compose=None) -> torch.Tensor:
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img: 3*SHAPE*SHAPE (RGB) tensor [0.0, 1.0]
    shapes = getPadShape(img.shape, shape)
    if trans is None:
        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Pad(shapes),
        ])
    img = trans(img)
    # tensor
    return img


def getPadShape(shape:int, tshape:int) -> Tuple[int, int, int, int]:
    padHt = (tshape - shape[0]) // 2
    padHd = tshape - shape[0] - padHt
    padWl = (tshape - shape[1]) // 2
    padWr = tshape - shape[1] - padWl
    return (padWl, padHt, padWr, padHd)
