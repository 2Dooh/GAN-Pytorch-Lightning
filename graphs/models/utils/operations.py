from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        self.instance_norm = nn.InstanceNorm2d(in_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        ori_x = x.clone()
        x = self.conv1(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instance_norm(x)
        return ori_x + x

class ContractingBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 use_bn=True,
                 kernel_size=3,
                 activation='ReLU',
                 act_kwargs={}):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels*2,
            kernel_size=kernel_size,
            padding=1,
            stride=2,
            padding_mode='reflect'
        )
        self.activation = getattr(nn, activation, nn.ReLU)(**act_kwargs)
        self.instance_norm = \
            nn.InstanceNorm2d(in_channels*2) if use_bn else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels//2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.instance_norm = \
            nn.InstanceNorm2d(in_channels//2) if use_bn else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )

    def forward(self, x):
        x = self.conv(x)
        return x