from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1, 
            padding_mode='reflect'
        )
        self.conv2 = nn.Conv2d(
            in_channels, 
            in_channels, 
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
                 activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            in_channels * 2, 
            kernel_size=kernel_size, 
            padding=1, 
            stride=2, 
            padding_mode='reflect'
        )
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instance_norm = nn.InstanceNorm2d(in_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instance_norm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, 
            in_channels // 2, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            output_padding=1)
        if use_bn:
            self.instance_norm = nn.InstanceNorm2d(in_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instance_norm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a Generator - 
    maps each the output to the desired number of output channels
    Values:
        in_channels: the number of channels to expect from a given input
        out_channels: the number of channels to expect for a given output
    '''
    def __init__(self, in_channels, out_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x