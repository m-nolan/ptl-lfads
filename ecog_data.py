import h5py
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pytorch_lightning as pl
from math import floor
import os

# - - -- --- ----- -------- ------------- --------------------- ----------------------------------
# pytorch Dataset class implementation for Electrocorticograph (ECoG) data
# ---------------------------------- --------------------- ------------- -------- ----- --- -- - -

class EcogSrcTrgDataset_inmem(Dataset):
    '''
        Dataset module for src/trg pair returns from a given tensor source (in memory, pytorch tensor).

        use:
            src, trg = dset.__getitem__(idx)
    '''
    def __init__(self, tensor, src_len, trg_len=None, transform=None):
        self.src_len = src_len
        if trg_len:
            trg_len = src_len
        self.trg_len = trg_len
        
        assert tensor.shape[1] >= src_len + trg_len, f"sequence length cannot be longer than 1/2 data sample length ({tensor.shape[1]})"
        self.tensor = torch.tensor(tensor).float()
        self.transform = transform

    def __getitem__(self, index):
        src = self.tensor[index,:self.src_len,:]
        trg = self.tensor[index,self.src_len:self.src_len+self.trg_len,:]
        if self.transform:
            src = self.transform(src)
        return (src, trg)

    def __len__(self):
        return self.tensor.shape[0]

class EcogSrcTrgDataset(Dataset):
    '''
        Dataset module for src/trg pair returns from a given tensor source (file location, hdf5)
    '''
    def __init__(self, file_path, src_len, trg_len=None, split_str='train', transform=None):
        None
        self.file_path  = file_path
        self.split_str  = split_str
        self.read_str   = f'{self.split_str}_ecog'
        self.src_len    = src_len
        if trg_len:
            trg_len         = src_len
        self.trg_len    = trg_len
        with h5py.File(self.file_path,'r') as hf:
            self.shape      = hf[self.read_str].shape
        # assert self.shape[1] >= src_len + trg_len, f"sequence length cannot be longer than 1/2 data sample length ({self.shape[1]})"
        self.transform  = transform

    def __getitem__(self, index):
        with h5py.File(self.file_path,'r') as hf:
            sample = hf[self.read_str][index,:,:]
        src = torch.tensor(sample[:self.src_len,:], dtype=torch.float32)
        trg = torch.tensor(sample[:self.trg_len,:], dtype=torch.float32)
        return (src,trg)

    def __len__(self):
        return self.shape[0]

# - - -- --- ----- -------- ------------- --------------------- ----------------------------------
# ECoG sample transforms
# ---------------------------------- --------------------- ------------- -------- ----- --- -- - -

class DropChannels(object):
    '''
        Dataset transform to randomly drop channels (i.e. set all values to zero) within a sample.
        The number of dropped channels is determined by the drop ratio:
            n_drop = floor(drop_ratio*n_ch)
        Channel dimension is assumed to be the last indexed tensor dimension. This may need to be
        adjusted for multidimensional time series data, e.g. spectrograms.
    '''
    def __init__(self,drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self,sample):
        _, n_ch = sample.shape
        n_ch_drop = floor(self.drop_ratio*n_ch)
        drop_ch_idx = torch.randperm(n_ch)[:n_ch_drop]
        sample[:,drop_ch_idx] = 0.
        return sample

''' 
other transform ideas:
    - Additive noise (variable power levels, variable channel counts, all channels v rand, ...)
    - Shot noise (may be implemented in torchvision, I'd be surprised if it weren't)
    - ...?
'''

# - - -- --- ----- -------- ------------- --------------------- ----------------------------------
# pytorch-lightning LightningDataModule implementation for ECoG datasets
# ---------------------------------- --------------------- ------------- -------- ----- --- -- - -

class GooseWireless250(pl.LightningDataModule):
    '''
        Data Module for the (1s max) Goose Wireless dataset.
        
        Data is sampled at 250Hz. No BPF beyond decimation required during downsampling from 1kHz. Dataset has a fixed 80:10:10::train:val:test split.
    '''
    def __init__(self, src_len, trg_len, batch_size, transforms=None, data_device='cpu'):
        super().__init__()

        self.src_len    = src_len
        self.trg_len    = trg_len
        self.batch_size = batch_size
        # this is a hdf5 dataset with the following items (flat structure): dt, train_data, valid_data, test_data.
        file_path       = "D:\\Users\\mickey\\Data\\datasets\\ecog\\goose_wireless\\gw_250_renorm"
        self.file_path  = file_path
        with h5py.File(self.file_path,'r') as hf:
            self.train_dims = hf['train_ecog'].shape
            self.val_dims   = hf['valid_ecog'].shape
            self.test_dims  = hf['test_ecog'].shape
        self.dims       = ( # use this to create model input sizes
            self.train_dims[0] + self.val_dims[0] + self.test_dims[0],    # n_trial
            self.train_dims[1],                                             # n_sample
            self.train_dims[2]                                              # n_channel
        )
        self.transforms     = transforms
        self.data_device    = data_device   # I want to keep the data tensors on the CPU, then read batches to the GPU.

    def prepare_data(self): # run once. 
        assert os.path.exists(self.file_path), "Dataset file not found, check file path string"
        return None

    def setup(self, stage=None): # run on each GPU
        self.train_dataset  = EcogSrcTrgDataset(
            file_path   = self.file_path,
            split_str   = 'train',
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            transform   = self.transforms
        )#.to(self.data_device)
        self.val_dataset    = EcogSrcTrgDataset(
            file_path   = self.file_path,
            split_str   = 'valid',
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            transform   = self.transforms
        )#.to(self.data_device)
        self.test_dataset = EcogSrcTrgDataset(
            file_path   = self.file_path,
            split_str   = 'test',
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            transform   = self.transforms
        )#.to(self.data_device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# - - -- --- ----- -------- ------------- --------------------- ----------------------------------
# ---------------------------------- --------------------- ------------- -------- ----- --- -- - -

def _read_h5(data_file_path):
    '''
        Wrapper function to read all top-level items from an hdf5 dataset located at data_file_path.
    '''
    try:
        with h5py.File(data_file_path, 'r') as hf:
            data_dict = {k: torch.tensor(hf[k].value) for k in hf.keys()}
    except IOError: # do I need to define data_dict is this case?
        raise IOError(f'Cannot open data file {data_file_path}.')
    return data_dict

def _read_h5_key(data_file_path,key):
    try:
        with h5py.File(data_file_path,'r') as hf:
            data = torch.tensor(hf[key].value)
    except IOError:
        raise IOError(f'Cannot open data file {data_file_path}.')
    return data