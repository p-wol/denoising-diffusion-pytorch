import time
import copy
from collections import OrderedDict
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import mlxpy
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from functools import partialmethod
from pathlib import Path
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            #transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

path_out = '/gpfswork/rech/tza/uki35ex/dataset/fickr_faces'
ds = Dataset('/gpfsdswork/dataset/FlickrFace/images1024x1024', image_size = 256)

for i in range(29653, 70000):
    ds[i].save(path_out + '/{:05}.png'.format(i))
