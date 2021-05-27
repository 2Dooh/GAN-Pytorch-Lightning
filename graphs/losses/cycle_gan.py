from torch import nn
import torch

class DiscriminatorLoss(nn.Module):
    def __init__(self, adv_criterion):
        super().__init__()
        self.adv_criterion = adv_criterion

    def forward(self, real, fake, netD):
        D_fake_out = netD(fake)
        loss_D_fake = \
            self.adv_criterion(
                D_fake_out, 
                torch.zeros_like(D_fake_out)
            )

        D_real_out = netD(real)
        loss_D_real = \
            self.adv_criterion(
                D_real_out,
                torch.ones_like(D_real_out)
            )
        
        loss_D = (loss_D_fake + loss_D_real) / 2

        return loss_D

class GeneratorLoss(nn.Module):
    def __init__(self, 
                 identity_criterion,
                 cycle_criterion,
                 adv_criterion,
                 lambda_identity=.1,
                 lambda_cycle=10):
        super().__init__()

        self.adv_criterion = adv_criterion
        self.identity_criterion = identity_criterion
        self.cycle_criterion = cycle_criterion
        self.lambda_identity = lambda_identity
        self.lambda_cycle = lambda_cycle

    def get_identity_loss(self, real_X, netG_YX):
        identity_X = netG_YX(real_X)
        identity_loss = self.identity_criterion(identity_X, real_X)
        return identity_loss, identity_X

    def get_cycle_consistency_loss(self, 
                                   real_X, 
                                   fake_Y, 
                                   netG_YX):
        cycle_X = netG_YX(fake_Y)
        cycle_loss = self.cycle_criterion(cycle_X, real_X)
        return cycle_loss, cycle_X

    def get_adversarial_loss(self,
                             real_X,
                             netD_Y,
                             netG_XY):
        fake_Y = netG_XY(real_X)
        D_fake_Y_out = netD_Y(fake_Y)
        adv_loss = self.adv_criterion(
            D_fake_Y_out,
            torch.ones_like(D_fake_Y_out)
        )
        return adv_loss, fake_Y

    def forward(self,
                real_A,
                real_B,
                netG_AB,
                netG_BA,
                netD_A,
                netD_B):
        adv_loss_BA, fake_A = \
            self.get_adversarial_loss(
                real_X=real_B,
                netD_Y=netD_A,
                netG_XY=netG_BA
            )
        adv_loss_AB, fake_B = \
            self.get_adversarial_loss(
                real_X=real_A,
                netD_Y=netD_B,
                netG_XY=netG_AB
            )
        adv_loss = adv_loss_AB + adv_loss_BA
        
        identity_loss_A, _ = \
            self.get_identity_loss(
                real_X=real_A,
                netG_YX=netG_BA
            )
        identity_loss_B, _ = \
            self.get_identity_loss(
                real_X=real_B,
                netG_YX=netG_AB
            )
        identity_loss = identity_loss_A + identity_loss_B

        cycle_loss_BA, _ = \
            self.get_cycle_consistency_loss(
                real_X=real_A,
                fake_Y=fake_B,
                netG_YX=netG_BA
            )
        cycle_loss_AB, _ = \
            self.get_cycle_consistency_loss(
                real_X=real_B,
                fake_Y=fake_A,
                netG_YX=netG_AB
            )
        cycle_loss = cycle_loss_AB + cycle_loss_BA

        loss_G = \
            self.lambda_identity * identity_loss + \
            self.lambda_cycle * cycle_loss + \
            adv_loss
        
        return loss_G, fake_A, fake_B