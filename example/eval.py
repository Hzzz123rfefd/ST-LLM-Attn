import sys
import os
sys.path.append(os.getcwd())

import argparse
from torch.utils.data import DataLoader
from src import models, datasets
from src.utils import load_config


def main(args):
    config = load_config(args.model_config_path)

    """ get model"""
    net = models[config["model_type"]](**config["model"])
    
    net.load_pretrained(
        save_model_dir = args.model_path
    )
    
    """get data loader"""
    dataset = datasets[config["dataset_type"]](
        valid_data_path = args.data_path,
        data_type = "valid"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = dataset.collate_fn
    )
    
    net.eval_epoch(epoch = 0, val_dataloader = dataloader, log_path = None)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/st_llm_attn.yml")
    parser.add_argument("--data_path", type=str, default = "gaotie_trainning/16/train.jsonl")
    parser.add_argument("--model_path", type=str, default = "saved_model/st_llm_attn_16")
    args = parser.parse_args()
    main(args)
