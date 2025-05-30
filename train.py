import argparse
from torch.utils.data import DataLoader

from src.dataset import *
from src import datasets,models
from src.utils import *
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def main(args):
    config = load_config(args.model_config_path)

    """ get net struction"""
    net = models[config["model_type"]](**config["model"]).to(config["traininng"]["device"])

    """get data loader"""
    train_datasets = datasets[config["dataset_type"]](**config["dataset"], data_type = "train")
    test_datasets = datasets[config["dataset_type"]](**config["dataset"], data_type = "test")
    val_datasets = datasets[config["dataset_type"]](**config["dataset"], data_type = "valid")

    train_dataloader = DataLoader(
        train_datasets,
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = train_datasets.collate_fn
    )
    
    test_dataloader = DataLoader(
        test_datasets, 
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = test_datasets.collate_fn
    )

    val_dataloader = DataLoader(
        val_datasets, 
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = test_datasets.collate_fn
    )

    """ trainning """
    net.trainning(
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        val_dataloader = val_dataloader,
        optimizer_name = config["traininng"]["optimizer"],
        clip_max_norm = config["traininng"]["clip_max_norm"],
        factor = config["traininng"]["factor"],
        patience = config["traininng"]["patience"],
        lr = config["traininng"]["learning_rate"],
        weight_decay = config["traininng"]["weight_decay"],
        total_epoch = config["traininng"]["epochs"],
        eval_interval = config["logging"]["eval_interval"],
        save_model_dir = config["logging"]["save_dir"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/st_llm_attn.yml")
    args = parser.parse_args()
    main(args)