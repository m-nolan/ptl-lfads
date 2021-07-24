# Example pytorch-lightning training script. Here I'm using a conv AE model
# to reconstruct multichannel ECoG signals. It isn't very accurate.
# ...but, it does implement a few cool tricks like early stopping and WandB reports.

# next up should be a hparam sweep, which isn't too tough with wandb. Maybe hyperopt, instead?

import os

import ecog_data
from ecog_convae import ConvAE

import torch

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def create_config_dict():
    config_dict = {
        'src_len': 50,
        'trg_len': 50,
        'batch_size': 1000,
        'latent_size': 10,
        'n_kernels': 50,
        'kernel_size': 9,
        'pool_size': 2,
        'dropout': 0.2,
        'learning_rate': 1e-3,
        'max_epochs': 100,
    }
    return config_dict

def configure_wandb(config_dict):
    name        = 'ecog_conv_ae_test_run'
    project     = 'ecog-ConvAE-test'
    wandb.init(
        config  = config_dict,
        name    = name,
        project = project
    )
    wandb_logger = WandbLogger(name=name,project=project)
    return wandb_logger

def get_dataset():
    data = ecog_data.GooseWireless250(
        src_len     = wandb.config.src_len,
        trg_len     = wandb.config.trg_len,
        batch_size  = wandb.config.batch_size
    )
    return data

def configure_model(data):
    model   = ConvAE(
        input_size      = data.dims[-1],
        latent_size     = wandb.config.latent_size,
        src_len         = data.src_len,
        trg_len         = data.trg_len,
        n_kernels       = wandb.config.n_kernels,
        kernel_size     = wandb.config.kernel_size,
        pool_size       = wandb.config.pool_size,
        dropout         = wandb.config.dropout,
        learning_rate   = wandb.config.learning_rate
    )
    return model

def get_callbacks():
    checkpoint_dir_path = os.path.join(wandb.run.dir,'checkpoint')
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor         = 'avg_valid_loss',
        dirpath         = os.path.join(wandb.run.dir,'checkpoint'),
        filename        = 'conv_ae-{epoch:03d}-{val_loss:.3f}'
    )
    early_stopping_cb = pl.callbacks.early_stopping.EarlyStopping(
        monitor         = 'avg_valid_loss'
    )
    return [ckpt_cb, early_stopping_cb]

def get_trainer(wandb_logger,callbacks):
    trainer = pl.Trainer(
        max_epochs  = wandb.config.max_epochs,
        logger      = wandb_logger,
        gpus        = 1,
        callbacks   = callbacks
    )
    return trainer

def main():
    config_dict = create_config_dict()
    wandb_logger = configure_wandb(config_dict)
    data = get_dataset()
    model = configure_model(data)
    callbacks = get_callbacks()
    trainer = get_trainer(wandb_logger,callbacks)
    trainer.fit(model,data)

if __name__ == '__main__':
    main()