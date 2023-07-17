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
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from functools import partialmethod

import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def assign_device(device):
    device = int(device)
    if device > -1:
        if torch.cuda.is_available():
            device = 'cuda:' + str(device)
        else:
            device = 'cpu'
    elif device == -1:
        device = 'cuda'
    elif device == -2:
        device = 'cpu'
    else:
        ValueError('Unknown device: {}'.format(device))

    return device

def get_dtype(dtype):
    if dtype == 64:
        return torch.double
    elif dtype == 32:
        return torch.float
    else:
        raise ValueError('Unkown dtype: {}'.format(dtype))


class Trainer:
    def __init__(self, config, logger):
        self.args = config
        self.logger= logger

        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)

    def build_dataset(self):
        args = self.args

        if args.dataset.name == 'noise':
            self.trainset = torch.rand(8, 3, 128, 128, device = self.device) # images are normalized from 0 to 1
            self.testset = None #torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
        elif args.dataset.name == 'MNIST':
            #transform = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            transform = [transforms.ToTensor()]
            #if not args.model.name == 'LeNet':
            #    transform.append(transforms.Lambda(lambda x: x.view(-1)))
            transform = transforms.Compose(transform)

            tvsize = 60000

            self.trainset = torchvision.datasets.MNIST(root = args.dataset.path, train = True,
                    download = False, transform = transform)

            self.testset = torchvision.datasets.MNIST(root = args.dataset.path, train = False,
                    download = False, transform = transform)

            self.n_classes = 10
            self.n_channels = 1
            self.channel_size = 28**2
        elif args.dataset.name == 'CIFAR10':
            #transform = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
            transform = [transforms.ToTensor()]
            #if not args.model.name == 'LeNet':
            #    transform.append(transforms.Lambda(lambda x: x.view(-1)))
            transform = transforms.Compose(transform)

            tvsize = 50000

            self.trainset = torchvision.datasets.CIFAR10(root = args.dataset.path, train = True,
                    download = False, transform = transform)
            self.testset = torchvision.datasets.CIFAR10(root = args.dataset.path, train = False,
                    download = False, transform = transform)

            self.n_classes = 10
            self.n_channels = 3
            self.channel_size = 32**2
        else:
            raise NotImplementedError('Unknown dataset: {}.'.format(args.dataset.name))

        # Create training set and validation set
        """
        train_size = tvsize - args.dataset.valid_size
        valid_size = args.dataset.valid_size
        self.trainset, self.validset = tuple(data.random_split(tvset, [train_size, valid_size]))
        """

        # Create loaders
        """
        train_loader = data.DataLoader(self.trainset, args.dataset.batch_size, shuffle = True)
        valid_loader = data.DataLoader(self.validset, args.dataset.batch_size)
        test_loader = data.DataLoader(self.testset, args.dataset.batch_size)
        """

        #return  #train_loader, valid_loader, test_loader


    def train(self, ckpt_name = 'last_ckpt', log_name = 'metrics'):
        print('Build model')
        model = Unet(dim = 64, dim_mults = (1, 2, 4, 8), flash_attn = True).to(device = self.device)

        print('Build diffusion model.')
        diffusion = GaussianDiffusion(model, image_size = 128, timesteps = 1000).to(device = self.device)

        print('Build dataset.')
        self.build_dataset()
        
        #training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
        trainer = Trainer(
            diffusion,
            self.trainset,
            train_batch_size = 32,
            train_lr = 8e-5,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = True              # whether to calculate fid during training
        )

        trainer.train()
        
        """
        print('Compute loss.')
        loss = diffusion(self.trainset)
        loss.backward()
        
        # after a lot of training

        print('Sample images.')
        sampled_images = diffusion.sample(batch_size = 4)
        print(sampled_images.shape) # (4, 3, 128, 128)
        """

        """
            # Logs
            metrics = metrics_tr | metrics_va | metrics_ts
            print(metrics)

            #self.logger.log_checkpoint(self, log_name = ckpt_name)
            self.logger.log_metrics(metrics, log_name = log_name)
            #self.logger.log_metrics({'lr': self.lr}, log_name = 'lrs')
        """ 

        """
        metrics = add_prefix(prefix,metrics)
        print(metrics)
        self.logger.log_checkpoint(self, log_name= ckpt_name)
        self.logger.log_metrics(metrics, log_name=log_name)
        """
