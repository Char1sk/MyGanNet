from image_similarity_measures.quality_metrics import fsim

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

import os
from typing import List


def get_file_list_from_file_list(folder:str, file:str) -> List[str]:
    with open(os.path.join(folder, file), 'r') as f:
        files = [os.path.join(folder, line.strip()) for line in f.readlines()]
    return files


def get_file_list_from_folder(folder:str) -> List[str]:
    files = [os.path.join(folder, p) for p in os.listdir(folder)]
    return files


def read_img_as_ndarray(path:str) -> np.ndarray:
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(img, 2)
    # img = img.transpose(2, 0, 1)
    # print(img.shape)
    return img


def read_img_as_tensor_numpy(path:str) -> Tensor:
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img: 3*H*W (RGB) tensor
    img = transforms.ToTensor()(img)
    
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    # print(img.shape, img.dtype)
    
    return img


def eval_avg_fsim(list1, list2) -> None:
    # assume that they are paired by order
    assert(len(list1) == len(list2))
    length = len(list1)
    
    avg_fsim = 0.0
    for i in range(length):
        img1 = read_img_as_tensor_numpy(list1[i])
        img2 = read_img_as_tensor_numpy(list2[i])
        f = fsim(img1, img2)
        avg_fsim += f
        print(f)
    avg_fsim /= length
    return avg_fsim


if __name__ == '__main__':
    # img_o = read_img_as_ndarray('./eval/1o.jpg')
    # img_p = read_img_as_ndarray('./eval/1p.jpg')
    # img_pp = read_img_as_ndarray('./eval/1pp.jpg')
    # img_o = read_img_as_tensor_numpy('./eval/1o.jpg')
    # img_p = read_img_as_tensor_numpy('./eval/1p.jpg')
    # img_pp = read_img_as_tensor_numpy('./eval/1pp.jpg')
    
    l1 = get_file_list_from_file_list('../Datasets/My-CUFS-New/', 'files/test/list_photo.txt')
    l2 = get_file_list_from_folder('./Saves/Images/train_first_04/Test/700')
    f = eval_avg_fsim(l1, l2)
    print('avg: ', f)
    
    # img1 = read_img_as_ndarray('../Datasets/My-CUFS-New/CUHK/photos/89.jpg')
    # img2 = read_img_as_ndarray('./Saves/Images/train_first_04/Test/700/1.jpg')
    # img1 = read_img_as_tensor_numpy('../Datasets/My-CUFS-New/CUHK/photos/89.jpg')
    # img2 = read_img_as_tensor_numpy('./Saves/Images/train_first_04/Test/700/1.jpg')
    # print(fsim(img1,img2))
