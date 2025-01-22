import torch
import torch.nn as nn
import math
from PIL import Image
import sys
sys.path.append('/mnt/woozh/SpoofingCFAS/multi_model')
from network_inf import builder_inf
from torchvision.transforms.functional import resize

def get_threshold():
    thresholds = [0.33,0.33,0.33]
    return thresholds

def get_models(device, drop=None):
    print("Start loading models...")
    nets = []

    net = builder_inf(arch='iresnet50', embedding_size=512, resume_path='/mnt/zsb/face_model/arcface_iresnet50_MS1MV2_dp.pth',device=device).eval()
    nets.append(net)
    net = builder_inf(arch='iresnet50', embedding_size=512, resume_path='/mnt/zsb/face_model/magface_iresnet50_MS1MV2_dp.pth',device=device).eval()
    nets.append(net)
    net = builder_inf(arch='iresnet100', embedding_size=512, resume_path='/mnt/zsb/face_model/magface_iresnet100_MS1MV2.pth',device=device).eval()
    nets.append(net)    
    return nets


class MyResize(nn.Module):
    def __init__(self, l=112) -> None:
        super().__init__()
        self.l = l
        self.f = torch.nn.AdaptiveAvgPool2d((l, l))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        s = img.shape[-1]
        if s <= self.l:
            return resize(img, [self.l, self.l], interpolation=Image.BICUBIC)
        else:
            new_size = int(self.l * (s//self.l))
            img = resize(img, [new_size, new_size], interpolation=Image.BICUBIC)
            return self.f(img)

if __name__ == "__main__":
    device = 'cuda:0'
    # model = get_pocketnet(device)
    # x = torch.rand((1, 3, 112, 112)).to(device)
    # y = model(x)
    # print(y.shape)
    # nets = get_models(device)
    # print('0')
    # # net = iresnet34()
    # # net.eval()
    # for net in nets:
    #     x = torch.rand((1, 3, 112, 112)).to(device)
    #     y = net(x)
    #     print(y.shape)
 