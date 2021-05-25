from torch import nn

from .utils.operations import ContractingBlock, ResidualBlock, ExpandingBlock, FeatureMapBlock

class Generator(nn.Module):
    N_RES_BLOCK = 9
    RES_MULT = 4
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=64,
                 activation='Tanh'):
        super(Generator, self).__init__()

        self.up_feat = FeatureMapBlock(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        self.contract = nn.Sequential(
            ContractingBlock(hidden_channels),
            ContractingBlock(hidden_channels*2)
        )
        # self.contract1 = ContractingBlock(hidden_channels)
        # self.contract2 = ContractingBlock(hidden_channels*2)
        # self.res_blocks = nn.ModuleList([
        #     ResidualBlock(hidden_channels * self.RES_MULT)\
        #         for _ in range(self.N_RES_BLOCK)
        # ])
        self.res = nn.Sequential(
            *[ResidualBlock(hidden_channels * self.RES_MULT)\
                for _ in range(self.N_RES_BLOCK)]
        )

        self.expand = nn.Sequential(
            ExpandingBlock(hidden_channels*4),
            ExpandingBlock(hidden_channels*2)
        )
        # self.expand2 = ExpandingBlock(hidden_channels*4)
        # self.expand3 = ExpandingBlock(hidden_channels*2)

        self.down_feat = FeatureMapBlock(
            in_channels=hidden_channels,
            out_channels=out_channels
        )

        self.activation = getattr(nn, activation, nn.Tanh)()

    def forward(self, x):
        x = self.up_feat(x)
        x = self.contract(x)
        # x = self.contract1(x)
        # x = self.contract2(x)
        # for block in self.res_blocks:
        #     x = block(x)
        x = self.res(x)
        # x = self.expand2(x)
        # x = self.expand3(x)
        x = self.expand(x)
        x = self.down_feat(x)
        x = self.activation(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels):
        super(Discriminator, self).__init__()

        self.up_feat = FeatureMapBlock(
            in_channels=in_channels,
            out_channels=hidden_channels
        )

        growth_rates = [1, 2, 4]
        self.contract = nn.Sequential(*[
            ContractingBlock(
                in_channels=hidden_channels*rate,
                use_bn=False,
                kernel_size=4,
                activation='LeakyReLU',
                act_kwargs={'negative_slope': .2}
            )\
                for rate in growth_rates
        ])

        self.final = nn.Conv2d(
            in_channels=hidden_channels*8,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        x = self.up_feat(x)
        x = self.contract(x)
        x = self.final(x)
        return x