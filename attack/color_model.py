import torch
from torch import nn
import os
import skimage.io
from PIL import Image
import pickle
import random
import matplotlib.pyplot as plt
import torchvision as tv

def color_correction(img, device):
    cmodel = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )

    model_names = ['model_hik.pt']
    
    model_dir = '../sim_phy/models_0124/'

    model_name = random.choice(model_names)
    param = torch.load(os.path.join(model_dir, model_name))
    cmodel.load_state_dict(param)
    cmodel.to(device)
    img = img.permute(0, 2, 3, 1)
    img = cmodel(img - 0.5) + 0.5
    return torch.clamp(img, 0, 1).permute(0, 3, 1, 2)



# pil2tensor = tv.transforms.ToTensor()
# patch = pil2tensor(Image.open('/home/zsb/FaceX-zoo-main/sim_phy/ref.png')).cuda(0)
# # patch = patch.astype('float32') / 255.0
# patch = torch.from_numpy(patch.transpose([2, 0, 1])).cuda(0)[0:3]
# imga = color_correction(patch.unsqueeze(0), patch.device) 
# noise = torch.abs(imga).squeeze(0).permute([1, 2, 0]).data.cpu().numpy()
# plt.imsave('color/sim16.png', noise, format='png')     