from enum import Enum, auto

import pytorch_lightning as pl

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from graphs.models.cycle_gan import Generator, Discriminator
from graphs.losses.cycle_gan import DiscriminatorLoss, GeneratorLoss

from utils.init_weight import weights_init
from utils.visualize import show_tensor_images

import random

import matplotlib.pyplot as plt
class Optim(Enum):
    D_A = 0
    D_B = auto()
    G = auto()

class CycleGAN(pl.LightningModule):
    def __init__(self,
                 cfg,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.netG_AB = Generator(**cfg.net.G); self.netG_BA = Generator(**cfg.net.G)
        self.netD_A = Discriminator(**cfg.net.D); self.netD_B = Discriminator(**cfg.net.D)

        adv_criterion = getattr(nn, cfg.criterion.adv)()
        recon_criterion = getattr(nn, cfg.criterion.recon)()

        self.criterionG = GeneratorLoss(
            recon_criterion, 
            recon_criterion, 
            adv_criterion
        )
        self.criterionD = DiscriminatorLoss(adv_criterion)

        self.netG_AB.apply(weights_init); self.netG_BA.apply(weights_init)
        self.netD_A.apply(weights_init); self.netD_B.apply(weights_init)

        self.example_input_array = (torch.rand(1, *cfg.input_shape), torch.rand(1, *cfg.input_shape))

    def forward(self, real_A, real_B):
        fake_A = self.netG_BA(real_B)
        fake_B = self.netG_AB(real_A)
        return fake_A, fake_B

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch
        loss_G, fake_A, fake_B = \
                self.criterionG(
                    real_A,
                    real_B,
                    self.netG_AB,
                    self.netG_BA,
                    self.netD_A,
                    self.netD_B
                )
        self.log(
            'loss_G_val',
            loss_G,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return (real_A, real_B), (fake_B, fake_A)

    def configure_optimizers(self):
        optimizers = []
                
        optimD_A = \
            getattr(optim, self.cfg.optim.D.name, 'SGD')(
                self.netD_A.parameters(),
                **self.cfg.optim.D.kwargs
            )
        optimD_B = \
            getattr(optim, self.cfg.optim.D.name, 'SGD')(
                self.netD_B.parameters(),
                **self.cfg.optim.D.kwargs
            )
        optimizers += [optimD_A, optimD_B]
        if 'freq' in self.cfg.optim.D:
            freq = self.cfg.optim.D.freq
            optimizers = [{'optimizer': optim for optim in optimizers if not isinstance(optim, dict)}]
            optimizers[-1].update({'frequency': freq})
            optimizers[-2].update({'frequency': freq})

        optimG = \
            getattr(optim, self.cfg.optim.G.name, 'SGD')(
                list(self.netG_AB.parameters()) + list(self.netG_BA.parameters()),
                **self.cfg.optim.G.kwargs
            )
        optimizers += [optimG]
        if 'freq' in self.cfg.optim.G:
            freq = self.cfg.optim.G.freq
            optimizers = [{'optimizer': optim for optim in optimizers if not isinstance(optim, dict)}]
            optimizers[-1].update({'frequency': freq})

        schedulers = []
        if 'schedulers' in self.cfg:
            sched_lst = self.cfg.schedulers
            for sched in sched_lst:
                schedulers += \
                    [getattr(lr_scheduler, sched.name)(
                        optimizers[sched.optim_idx], 
                        **sched.kwargs
                    )]

        return optimizers, schedulers
        

    def training_step(self, 
                      train_batch, 
                      batch_idx, 
                      optimizer_idx):
        real_A, real_B = train_batch
        if optimizer_idx == Optim.D_A.value:
            with torch.no_grad():
                fake_A = self.netG_BA(real_B)
            loss_D_A = self.criterionD(
                real_A, 
                fake_A, 
                self.netD_A,
                # self.adv_criterion
            )
            self.log(
                'loss_D_A', 
                loss_D_A, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=False, 
                logger=True
            )
            return loss_D_A
        elif optimizer_idx == Optim.D_B.value:
            with torch.no_grad():
                fake_B = self.netG_AB(real_A)
            loss_D_B = self.criterionD(
                real_B, 
                fake_B, 
                self.netD_B
            )
            self.log(
                'loss_D_B',
                loss_D_B,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True
            )
            return loss_D_B
        else:
            loss_G, _, _ = \
                self.criterionG(
                    real_A,
                    real_B,
                    self.netG_AB,
                    self.netG_BA,
                    self.netD_A,
                    self.netD_B
                )
            self.log(
                'loss_G_train',
                loss_G,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )
            return loss_G


    def validation_epoch_end(self, outputs) -> None:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        i = random.randrange(len(outputs))
        real, fake = outputs[i]
        show_tensor_images(ax1, torch.cat(real), num_images=len(real), size=self.cfg.input_shape, title='original')
        show_tensor_images(ax2, torch.cat(fake), num_images=len(fake), size=self.cfg.input_shape, title='transfer features')
        plt.show()

    

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        if optimizer_idx == Optim.D_A.value or optimizer_idx == Optim.D_B.value:
          loss.backward(retain_graph=True)
        else:
          return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)