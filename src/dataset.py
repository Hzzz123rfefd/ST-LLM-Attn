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
        inflow_path = self.dataset[idx]["inflow_path"]
        outflow_path = self.dataset[idx]["outflow_path"]
        inflow = torch.from_numpy(np.load(inflow_path)).float()
        outflow = torch.from_numpy(np.load(outflow_path)).float()
        output["inflow"] = inflow[0: -1,:]
        output["inflow_label"] = inflow[-1,:]
        output["outflow"] = outflow[0: -1,:]
        output["outflow_label"] = outflow[-1,:]
        return output
    
    def collate_fn(self,batch):
        return  recursive_collate_fn(batch)