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

class NewCelebA(torchvision.datasets.CelebA):
    def __init__(self, *args, **kwargs):
        super(NewCelebA, self).__init__(*args, **kwargs)

    def _check_integrity(self):
        return True

class ImageDataset(data.Dataset):
    def __init__(self, root = '.', exts = ['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.root = root
        self.paths = [p for ext in exts for p in Path(f'{root}').glob(f'**/*.{ext}')]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class ListDataset(data.Dataset):
    def __init__(self, lst_tensors):
        self.lst_tensors = lst_tensors

    def __getitem__(self, index):
        return self.lst_tensors[index]

    def __len__(self):
        return len(self.lst_tensors)

class WrappedDataset(data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        return self.ds[index][0]

    def __len__(self):
        return len(self.ds)

class Training:
    def __init__(self, config, logger):
        self.args = config
        self.logger= logger

        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)

    def dataset_extract_images(self, ds):
        #images = [(torch.tensor(t, device = self.device, dtype = self.dtype).transpose(0, 2).transpose(1, 2) - 127.5)/127.5 for t in ds.data]
        images = [(torch.tensor(t).transpose(0, 2).transpose(1, 2) - 127.5)/127.5 for t in ds.data]
        return ListDataset(images)

    def build_dataset(self):
        args = self.args

        if args.dataset.name == 'noise':
            self.trainset = torch.rand(8, 3, 128, 128, device = self.device) # images are normalized from 0 to 1
            self.testset = None #torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
            self.image_size = 128
        elif args.dataset.name == 'MNIST':
            #transform = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            #transform = [transforms.ToTensor()]
            #if not args.model.name == 'LeNet':
            #    transform.append(transforms.Lambda(lambda x: x.view(-1)))
            #transform = transforms.Compose(transform)

            tvsize = 60000

            trainset = torchvision.datasets.MNIST(root = args.dataset.path, train = True, download = False)
            self.trainset = self.dataset_extract_images(trainset)

            testset = torchvision.datasets.MNIST(root = args.dataset.path, train = False, download = False)
            self.testset = self.dataset_extract_images(testset)

            self.n_classes = 10
            self.n_channels = 1
            self.image_size = 28
            self.channel_size = 28**2
        elif args.dataset.name == 'CIFAR10':
            #transform = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
            #transform = [transforms.ToTensor()]
            #if not args.model.name == 'LeNet':
            #    transform.append(transforms.Lambda(lambda x: x.view(-1)))
            #transform = transforms.Compose(transform)

            tvsize = 50000

            trainset = torchvision.datasets.CIFAR10(root = args.dataset.path, train = True, download = False)
            self.trainset = self.dataset_extract_images(trainset)

            testset = torchvision.datasets.CIFAR10(root = args.dataset.path, train = False, download = False)
            self.testset = self.dataset_extract_images(testset)

            self.n_classes = 10
            self.n_channels = 3
            self.image_size = 32
            self.channel_size = 32**2
        elif args.dataset.name == 'CelebA':
            transform = transforms.ToTensor()

            #trainset = torchvision.datasets.CelebA(root = args.dataset.path, split = self.args.dataset.type, 
            trainset = NewCelebA(root = args.dataset.path, split = self.args.dataset.type, 
                    download = False, transform = transform)
            self.trainset = WrappedDataset(trainset)

            #testset = torchvision.datasets.CelebA(root = args.dataset.path, split = self.args.dataset.type, 
            testset = NewCelebA(root = args.dataset.path, split = self.args.dataset.type, 
                    download = False, transform = transform)
            self.testset = WrappedDataset(testset)

            self.image_size = (218, 178)
        elif args.dataset.name == 'FlickrFace':
            ds = ImageDataset(args.dataset.path + 'FlickrFace')
            if self.args.dataset.max_size != -1:
                ds = data.Subset(ds, list(range(self.args.dataset.max_size)))
            self.trainset = ds
            self.testset = ds

            self.image_size = (256, 256)
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
        self.args.checkpoints.save_path = self.args.checkpoints.save_path + '/{}'.format(self.logger.forced_log_id)

        print('Build dataset.')
        self.build_dataset()
        if self.args.dataset.type == 'test':
            dataset = self.testset
        elif self.args.dataset.type == 'train':
            dataset = self.trainset
        else:
            raise ValueError('Unknown dataset type: {}'.format(self.args.dataset.type))

        print('Build model')
        model = Unet(dim = 64, dim_mults = (1, 2, 4, 8), flash_attn = True).to(device = self.device)

        print('Build diffusion model.')
        objective = self.args.sampling.objective
        diffusion = GaussianDiffusion(model, image_size = self.image_size, timesteps = 1000,
                objective = objective).to(device = self.device)
        
        #training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
        trainer = Trainer(
            diffusion,
            dataset,
            train_batch_size = self.args.dataset.batch_size,
            train_lr = self.args.optimizer.lr,
            train_num_steps = self.args.optimizer.num_steps,             # total training steps
            gradient_accumulate_every = self.args.optimizer.grad_acc,    # gradient accumulation steps
            ema_decay = 0.995,                                           # exponential moving average decay
            results_folder = self.args.checkpoints.save_path,
            amp = True,                                                  # turn on mixed precision
            calculate_fid = self.args.sampling.calculate_fid,            # whether to calculate fid during training
            num_fid_samples = self.args.sampling.num_fid_samples,
            num_workers = self.args.dataset.num_workers,
            pin_memory = self.args.dataset.pin_memory
        )

        if self.args.checkpoints.mode == 'train':
            trainer.train()
        elif self.args.checkpoints.mode == 'test':
            trainer.load(path = self.args.checkpoints.load_path)

            img = Image.open(self.args.checkpoints.image_test)
            img = transforms.functional.to_tensor(img).to(device = self.device)
            img.mul_(2).add_(-1)
            img = img.unsqueeze(0).expand(1000, *img.size())
            trainer.reconstruct(img, 20, 50, 50, 100)
        
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
