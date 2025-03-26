import numpy as np
import os
import json
from datetime import datetime, timedelta
import sys
import pandas as pd
sys.path.append(os.getcwd())


def normalization(data):
    min = np.min(data)
    max = np.max(data)
    norm = (data - min) / (max - min)
    return norm

def transform_and_save(inflow_path, outflow_path, output_dir, timestamp, train_output_dir, start_time, time_interval):
    inflow = pd.read_csv(inflow_path, header=None).to_numpy()
    outflow = pd.read_csv(outflow_path, header=None).to_numpy()
    
    np.save("dataset\gaotie\8_matrix_o.npy", inflow)
    np.save("dataset\gaotie\8_matrix_d.npy", outflow)
    
    inflow = normalization(inflow)
    outflow = normalization(outflow)
    
    n, t = inflow.shape
    new_t = t - timestamp
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)
    
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    time_interval = timedelta(minutes=time_interval)
    
    jsonl_path = os.path.join(train_output_dir, "train.jsonl")
    
    with open(jsonl_path, "w") as jsonl_file:
        for i in range(new_t):
            sub_inflow = inflow[:, i:i + timestamp + 1]  
            sub_outflow = outflow[:, i:i + timestamp + 1]  
            output_path1 = os.path.join(output_dir, f"inflow_{i}.npy")
            output_path2 = os.path.join(output_dir, f"outflow_{i}.npy")
            
            sub_inflow = np.transpose(sub_inflow, (1, 0))
            sub_outflow = np.transpose(sub_outflow, (1, 0))
            
            np.save(output_path1, sub_inflow)
            np.save(output_path2, sub_outflow)
            
            time_begin = start_time + i * time_interval
            time_str = time_begin.strftime("%Y-%m-%d %H:%M")
            
            json_entry = {"time_begin": time_str, "inflow_path": output_path1, "outflow_path": output_path2}
            jsonl_file.write(json.dumps(json_entry) + "\n")
            

start_time = "2020-09-10 10:10"  
interval_minutes = 60  
timestamp = 7                                                                                                     # 用历史多少个时间步
inflow_path = "dataset/gaotie/8_matrix_o.csv"                                                 # 进站流文件路径
outflow_path = "dataset/gaotie/8_matrix_d.csv"                                               # 出站流文件路径
output_directory = "dataset/gaotie/npy_data"  
train_output_dir = "gaotie_trainning" 
transform_and_save(inflow_path, outflow_path, output_directory, timestamp, train_output_dir, start_time, interval_minutes)
