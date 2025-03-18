import datasets
import numpy as np
import torch
from src.utils import *

class DatasetForTimeSeries:
    def __init__(
        self,
        train_data_path: str = None,
        test_data_path: str = None,
        valid_data_path: str = None,
        data_type:str = "train"
    ):
        if data_type == "train":
            self.dataset = datasets.load_dataset('json', data_files = train_data_path, split = "train")
        elif data_type == "test":
            self.dataset = datasets.load_dataset('json', data_files = test_data_path, split = "train")
        elif data_type == "valid":
            self.dataset = datasets.load_dataset('json', data_files = valid_data_path, split = "train")

        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        output = {}
        npy_data_path = self.dataset[idx]["npy_data_path"]
        data = torch.from_numpy(np.load(npy_data_path)).float()
        output["seq"] = data[0: -1,:]
        output["predict"] = data[-1,:]
        return output
    
    def collate_fn(self,batch):
        return  recursive_collate_fn(batch)