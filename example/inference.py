import sys
import os
sys.path.append(os.getcwd())

import argparse
from torch.utils.data import DataLoader
from src.utils import *
from src import models

def main(args):
    config = load_config(args.model_config_path)

    """ get model"""
    net = models[config["model_type"]](**config["model"])
    
    net.load_pretrained(
        save_model_dir = args.save_mode_dir
    )
    
    """ load_data """
    data = np.load(args.data_path)
    data = np.transpose(data, (1,0))
    time_step, station_num = data.shape
    min = np.min(data)
    max = np.max(data)
    data_norm = (data - min) / (max - min)
    trues = []
    preds = []
    """ inference """
    for i in range (time_step - 7 + 1):
        y = data[i + 6,:]
        trues.append(y)
        
        x = data_norm[i: i + 6,:]
        pred = net.inference(x)
        pred = pred * (max - min) + min
        preds.append(pred)
    
    true_array = np.array(trues)
    pred_array = np.array(preds)
    np.save('true_array.npy', true_array)
    np.save('pred_array.npy', pred_array)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/st_llm_attn.yml")
    parser.add_argument("--data_path", type=str, default = "dataset/gaotie/8_matrix_o.npy")
    parser.add_argument("--save_mode_dir", type=str, default = "saved_model/st_llm_attn")
    args = parser.parse_args()
    main(args)