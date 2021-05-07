import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import h5py

import prediction

class LfadsModel_ECoG(prediction.Lfads):

    def __init__(self, config):
        self.data_file_path         = config.data_file_path
        self.batch_size             = config.batch_size # check this - set by wandb in hparam sweeps
        self.seq_len                = config.sequence_length
        self.data_suffix            = config.data_suffix
        self.objective_function     = config.objective_function
        self.prepare_data()
        with h5py.File(self.data_file_path,'r') as hdf:
            _, _, self.input_size = hdf[f'test_{self.data_suffix}'].shape
        self.generator_size         = config.generator_size
        self.g_encoder_size         = config.g_encoder_size
        self.g_latent_size          = config.g_latent_size
        self.controller_size        = config.controller_size
        self.c_encoder_size         = config.c_encoder_size
        self.u_latent_size          = config.u_latent_size
        self.factor_size            = config.factor_size

        self.prior                  = config.prior

        self.clip_val               = config.clip_val
        self.factor_bias            = config.factor_bias

        # is it poor form to put this at the end of the init call?
        super(LfadsModel_ECoG, self).__init__(
            src_size = self.train_dataset.tensor.shape[-1],
            encoder_size = self.g_encoder_size,
            encoder_layers = 1,
            generator_size = self.generator_size,
            generator_layers = 1,
            factor_size = self.factor_size,
            loss_weight_dict = config.loss_weight_dict,
            dropout = config.dropout,
            learning_rate = config.learning_rate,
            lr_factor = config.learning_rate_factor
        )

        # # create modules
        # self.encoder = LFADS_Encoder(
        #     self.input_size, 
        #     self.g_encoder_size, 
        #     self.g_latent_size, 
        #     c_encoder_size = self.c_encoder_size, 
        #     dropout = self.dropout, 
        #     clip_val = self.clip_val
        #     )
        # self.controller = LFADS_ControllerCell(
        #     self.input_size, 
        #     self.controller_size, 
        #     self.u_latent_size, 
        #     dropout = self.dropout, 
        #     clip_val = self.clip_val, 
        #     factor_bias=self.factor_bias
        #     )
        # self.generator = LFADS_GeneratorCell(
        #     input_size, 
        #     generator_size, 
        #     factor_size,
        #     attention = False,
        #     dropout=self.dropout, 
        #     clip_val=self.clip_val, 
        #     factor_bias=self.factor_bias
        #     )

    def prepare_data(self):
        # load datasets from hdf5 volume
        # data_dict = self.read_h5(self.data_file_path)
        self.train_dataset  = EcogSrcTrgDatasetFromFile(
            self.data_file_path,
            f'train_{self.data_suffix}',
            self.seq_len
            )
        self.valid_dataset  = EcogSrcTrgDatasetFromFile(
            self.data_file_path,
            f'valid_{self.data_suffix}',
            self.seq_len
            )
        self.test_dataset   = EcogSrcTrgDatasetFromFile(
            self.data_file_path,
            f'test_{self.data_suffix}',
            self.seq_len
            )

    # these are defined adequately in the prediction.Lfads class!
    # def training_step(self,train_batch,batch_idx):
    #     src, trg = train_batch
    #     recon, (factors, gen_inputs) = self.forward(src, trg)
    #     loss = self.loss()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size)
        
    @staticmethod
    def read_h5(data_file_path):
        try:
            with h5py.File(data_file_path, 'r') as hf:
                data_dict = {k: torch.tensor(hf[k].value) for k in hf.keys()}
            return data_dict
        except IOError:
            print(f'Cannot open data file {data_file_path}.')
            raise
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = super(LFADS)
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument() # add this for all LFADS hyperparameters! (like the obj. function)

class EcogSrcTrgDataset(Dataset):
    
    def __init__(self, tensor, seq_len):
        assert tensor.shape[1] >= 2*seq_len, f"sequence length cannot be longer than 1/2 data sample length ({tensor.shape[1]})"
        self.tensor = torch.tensor(tensor).float()
        self.seq_len = seq_len

    def __getitem__(self, index):
        src = self.tensor[index,:self.seq_len,:]
        trg = self.tensor[index,self.seq_len:2*self.seq_len,:]
        return (src, trg)

    def __len__(self):
        return self.tensor.shape[0]

class EcogSrcTrgDatasetFromFile(Dataset):
    def __init__(self, file_path, key, seq_len):
        h5record = h5py.File(file_path,'r')
        self.tensor = h5record[key]
        assert self.tensor.shape[1] >= 2*seq_len, f"sequence length cannot be longer than 1/2 data sample length ({self.tensor.shape[1]})"
        self.seq_len = seq_len

    def __getitem__(self, index):
        src = torch.tensor(self.tensor[index,:self.seq_len,:],dtype=torch.float32)
        trg = torch.tensor(self.tensor[index,self.seq_len:2*self.seq_len,:],dtype=torch.float32)
        return (src, trg)

    def __len__(self):
        return self.tensor.shape[0]
