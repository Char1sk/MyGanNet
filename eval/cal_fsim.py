from image_similarity_measures.quality_metrics import fsim

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


def read_img_as_ndarray(path:str) -> np.ndarray:
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(img, 2)
    img = img.transpose(2, 0, 1)
    print(img.shape)
    return img


def read_img_as_tensor(path:str) -> Tensor:
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape, img.dtype)
    # img: 3*H*W (RGB) tensor
    img = transforms.ToTensor()(img)
    print(img.shape, img.dtype)
    
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    print(img.shape, img.dtype)
    
    return img


def main() -> None:
    img_o = read_img_as_tensor('./eval/1o.jpg')
    img_p = read_img_as_tensor('./eval/1p.jpg')
    img_pp = read_img_as_tensor('./eval/1pp.jpg')
    
    # img_o = read_img_as_ndarray('./eval/1o.jpg')
    # img_p = read_img_as_ndarray('./eval/1p.jpg')
    # img_pp = read_img_as_ndarray('./eval/1pp.jpg')
    
    f = fsim(img_o, img_p)
    print(f)


if __name__ == '__main__':
    main()
