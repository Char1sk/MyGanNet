from image_similarity_measures.quality_metrics import fsim

import cv2
from torch import Tensor
from torchvision import transforms

def cal_fsim_with_tensors(img_o:Tensor, img_p:Tensor) -> float:
    assert(img_o.shape == img_p.shape)
    # img: C*H*W Tensor
    if img_o.dim() == 4 and img_o.size(0) == 1:
        img_o = img_o.squeeze(0)
        img_p = img_p.squeeze(0)
    # img: H*W*C ndarray
    img_o = img_o.cpu().numpy().transpose(1, 2, 0)
    img_p = img_p.cpu().numpy().transpose(1, 2, 0)
    
    return fsim(img_o, img_p)


def read_img_as_tensor(path:str) -> Tensor:
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img: 3*H*W (RGB) tensor
    img = transforms.ToTensor()(img)
    
    return img


if __name__ == '__main__':
    img_o = read_img_as_tensor('./eval/1o.jpg')
    img_p = read_img_as_tensor('./eval/1p.jpg')
    img_pp = read_img_as_tensor('./eval/1pp.jpg')
    
    f = cal_fsim_with_tensors(img_o, img_p)
    print(f)
