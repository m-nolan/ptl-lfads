import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import os
import sys
# sys.path.append(os.path.dirname(os.getcwd()))
# import prediction # this is where the pt-l model code is
from ecog_lfads import LfadsModel_ECoG

import h5py
import argparse

import matplotlib.pyplot as plt

# this is a beefy config file! Stanardize this, for sure
wandb.init(
    config = {
        # model-agnostic hyperparameters
        'data_file_path': "D:\\Users\\mickey\\Data\\datasets\\ecog\\goose_wireless\\gw_250_renorm",
        'batch_size': 1000,
        'sequence_length': 50,
        'data_suffix': 'ecog',
        'objective_function': 'mse',
        'learning_rate': 0.001,
        'learning_rate_factor': 0.9,
        'device': 'cuda',

        # model-specific hyperparameters
        'g_encoder_size': 128,
        'c_encoder_size': 0,
        'g_latent_size': 128,
        'u_latent_size': 0,
        'controller_size': 0,
        'generator_size': 128,
        'factor_size': 42,
        'prior': {
            'g0' : {
                'mean' : {
                    'value': 0.0, 
                    'learnable' : True
                    },
                'var'  : {
                    'value': 0.1, 
                    'learnable' : True
                    },
                },
            'u'  : {
                'mean' : {
                    'value': 0.0, 
                    'learnable' : False
                    },
                'var'  : {
                    'value': 0.1, 
                    'learnable' : True
                    },
                'tau'  : {
                    'value': 10, 
                    'learnable' : True
                    },
                },
            },
        'clip_val': 100.0,
        'max_norm': 5.0,
        'do_normalize_factors': True,
        'factor_bias': False,
        'loss_weight_dict': {
            'kl': {
                'weight': 0.0,
                'min': 0.0,
                'max': 1.0,
                'schedule_dur': 1600,
                'schedule_start': 0,
            },
            'l2': {
                'weight': 0.0,
                'min': 0.0,
                'max': 1.0,
                'schedule_dur': 1600,
                'schedule_start': 0.0,
            },
            'l2_con_scale': 0,
            'l2_gen_scale': 2000,
        },
        'l2_gen_scale': 0.9,
        'l2_con_scale': 0.9,
        'dropout': 0.0,
        },
    mode="disabled"
    )

# make the model
prediction_model_shell = LfadsModel_ECoG(wandb.config)
prediction_model_shell.train_dataloader()

# create wandb logger
wandb_logger = WandbLogger(
    name='LFADS-wandbtest',
    project='GW_ECoG-Prediction'
)

# create callbacks, trainer
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = 'avg_valid_loss',
    dirpath = 'D:\\Users\\mickey\\Data\\models\\pytorch-lightning\\',
    filename = 'lfads-{epoch:03d}-{val_loss:.3f}',
)
early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor ='avg_valid_loss'
)
trainer = pl.Trainer(max_epochs=100, 
                    logger = wandb_logger, 
                    gpus=1, 
                    callbacks=[checkpoint_callback, early_stopping_callback])

# train the model
trainer.fit(prediction_model_shell)