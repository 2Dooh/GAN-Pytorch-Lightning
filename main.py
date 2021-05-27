from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.utils.data.dataloader import DataLoader

from torchvision import transforms

from agents.cycle_gan import CycleGAN

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.load_cfg import load_cfg
from utils.prepare_seed import prepare_seed

from datasets.dataset import ImageDataset

cfg_path = 'configs/horse2zebra.yaml'
cfg = load_cfg(cfg_path)

prepare_seed(cfg.exp_cfg.seed)

cycle_gan = CycleGAN(cfg=cfg.module_cfg)

transform = transforms.Compose([
    transforms.Resize(cfg.module_cfg.input_shape[1:]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([.5]*3, [.5]*3)
])
train_set = ImageDataset(
    root='data/horse2zebra', 
    mode='train', 
    transform=transform
)
train_queue = DataLoader(train_set, shuffle=True, **cfg.data_loader)
val_set = ImageDataset(
    root='data/horse2zebra', 
    mode='test', 
    transform=transform
)
val_queue = DataLoader(val_set, **cfg.data_loader)

checkpoint_callback = ModelCheckpoint(
    dirpath=cfg.checkpoint_dir,
    **cfg.model_checkpoint
)

logger = TensorBoardLogger(
    name=cfg.exp_name,
    **cfg.logger
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback], 
    default_root_dir=cfg.out_dir, 
    logger=logger,
    **cfg.trainer)
trainer.fit(
    model=cycle_gan,
    train_dataloader=train_queue,
    val_dataloaders=val_queue
)