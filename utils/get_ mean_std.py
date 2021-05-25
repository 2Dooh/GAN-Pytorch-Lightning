from __future__ import print_function

import matplotlib
matplotlib.use("pdf")

import torch
import torchvision.datasets as datasets

import torchvision.transforms as transforms

import click

# from models import *
@click.command()
@click.option('--dataset', default='CIFAR10', type=str, help='dataset')
@click.option('--batch_size', default='200', type=int, help='dataset')
@click.option('--folder', default='CIFAR10', type=str)
def cli(dataset, batch_size, folder):
    custom = False
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([transforms.ToTensor()])

    if 'SVHN' == dataset:
        trainset = getattr(datasets, dataset)(root='./data/' + dataset, 
                                              split='train', 
                                              download=True,
                                              transform=transform_train)
    elif dataset in globals() is not None:
        custom = True
        trainset = globals()[dataset](folder, transform_train)
    else:
        try:
            trainset = getattr(datasets, dataset)(root='./data/' + dataset, 
                                                  train=True, 
                                                  download=True, 
                                                  transform=transform_train)
        except:
            trainset = getattr(datasets, dataset)(root='./data/' + folder, 
                                                  transform=transform_train)

    print('%d training samples.' % len(trainset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = 0, 0
    if custom:
        for batch_idx, (inputs) in enumerate(trainloader):
            inputs = inputs.to(device)
            if batch_idx == 0:
                h, w = inputs.size(2), inputs.size(3)
                print(inputs.min(), inputs.max())
                chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
            else:
                chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
        mean = chsum/len(trainset)/h/w
        print('mean: %s' % mean.view(-1))

        chsum = None
        for batch_idx, (inputs) in enumerate(trainloader):
            inputs = inputs.to(device)
            if batch_idx == 0:
                chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            else:
                chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(chsum/(len(trainset) * h * w - 1))
        print('std: %s' % std.view(-1))
    else:
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            if batch_idx == 0:
                h, w = inputs.size(2), inputs.size(3)
                print(inputs.min(), inputs.max())
                chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
            else:
                chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
        mean = chsum/len(trainset)/h/w
        print('mean: %s' % mean.view(-1))

        chsum = None
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            if batch_idx == 0:
                chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            else:
                chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(chsum/(len(trainset) * h * w - 1))
        print('std: %s' % std.view(-1))

if __name__ == '__main__':
    pass