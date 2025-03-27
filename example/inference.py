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
        save_model_dir = args.model_path
    )
    
    """ load_data """
    inflow = np.load(args.inflow_path)
    inflow = np.transpose(inflow, (1,0))
    time_step, station_num = inflow.shape
    min_inflow = np.min(inflow)
    max_inflow = np.max(inflow)
    inflow_norm = (inflow - min_inflow) / (max_inflow - min_inflow)

    outflow = np.load(args.outflow_path)
    outflow = np.transpose(outflow, (1,0))
    time_step, station_num = outflow.shape
    min_outflow = np.min(outflow)
    max_outflow = np.max(outflow)
    outflow_norm = (inflow - min_outflow) / (max_outflow - min_outflow)
    
    inflow_trues = []
    inflow_preds = []
    outflow_trues = []
    outflow_preds = []
    """ inference """
    for i in range (time_step - args.step):
        y1 = inflow[i + args.step , :]
        inflow_trues.append(y1)
        x1 = inflow_norm[i: i + args.step, :]

        y2 = outflow[i + args.step , :]
        outflow_trues.append(y2)
        x2 = outflow_norm[i: i + args.step, :]
        
        inflow_predict, outflow_predict = net.inference(x1, x2)
        inflow_predict = inflow_predict * (max_inflow - min_inflow) + min_inflow
        outflow_predict = outflow_predict * (max_outflow - min_outflow) + min_outflow
        inflow_preds.append(inflow_predict)
        outflow_preds.append(outflow_predict)
    
    inflow_trues_array = np.array(inflow_trues)
    inflow_preds_array = np.array(inflow_preds)
    outflow_trues_array = np.array(outflow_trues)
    outflow_preds_array = np.array(outflow_preds)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(args.save_dir + "inflow_trues.npy", inflow_trues_array)
    np.save(args.save_dir + "inflow_preds.npy", inflow_preds_array)
    np.save(args.save_dir + "outflow_trues.npy", outflow_trues_array)
    np.save(args.save_dir + "outflow_preds.npy", outflow_preds_array)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/st_llm_attn.yml")
    parser.add_argument("--inflow_path", type=str, default = "dataset/gaotie/inflow.npy")
    parser.add_argument("--outflow_path", type=str, default = "dataset/gaotie/outflow.npy")
    parser.add_argument("--model_path", type = str, default = "saved_model/st_llm_attn")
    parser.add_argument("--step", type = int, default = 7)
    parser.add_argument("--save_dir", type = str, default = "result/")
    args = parser.parse_args()
    main(args)