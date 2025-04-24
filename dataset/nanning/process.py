import argparse
import numpy as np
import os
import json
from datetime import datetime, timedelta
import sys
sys.path.append(os.getcwd())

def normalization(data):
    min = np.min(data)
    max = np.max(data)
    norm = (data - min) / (max - min)
    print(f"max = {max}, min = {min}")
    return norm

def transform_and_save(inflow_path, outflow_path, output_dir, timestamp, train_output_dir, start_time, time_interval):
    inflow = np.load(inflow_path)  
    outflow = np.load(outflow_path)  
    np.save("dataset/nanning/inflow.npy", inflow)
    np.save("dataset/nanning/outflow.npy", outflow)
    inflow = normalization(inflow)
    outflow = normalization(outflow)
    
    n, t = inflow.shape
    new_t = t - timestamp
    
    os.makedirs(output_dir + str(timestamp), exist_ok=True)
    os.makedirs(train_output_dir + str(timestamp), exist_ok=True)
    
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    time_interval = timedelta(minutes=time_interval)
    
    jsonl_path = os.path.join(train_output_dir, str(timestamp), "train.jsonl")
    
    with open(jsonl_path, "w") as jsonl_file:
        for i in range(new_t):
            sub_inflow = inflow[:, i: i + timestamp + 1]  
            sub_outflow = outflow[:, i:i + timestamp + 1]  
            output_path1 = os.path.join(output_dir, str(timestamp), f"inflow_{i}.npy")
            output_path2 = os.path.join(output_dir, str(timestamp), f"outflow_{i}.npy")
            
            sub_inflow = np.transpose(sub_inflow, (1, 0))
            sub_outflow = np.transpose(sub_outflow, (1, 0))
            
            np.save(output_path1, sub_inflow)
            np.save(output_path2, sub_outflow)
            
            time_begin = start_time + i * time_interval
            time_str = time_begin.strftime("%Y-%m-%d %H:%M")
            
            json_entry = {"time_begin": time_str, "inflow_path": output_path1, "outflow_path": output_path2}
            jsonl_file.write(json.dumps(json_entry) + "\n")
            

if __name__ == "__main__":
    start_time = "2020-09-10 10:10"  
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval_minutes", type = int, default = 15)
    parser.add_argument("--timestamp", type = int, default = 16)
    parser.add_argument("--inflow_path", type = str, default = "dataset/nanning/数据/2019元旦/inflow_data_15min.npy")
    parser.add_argument("--outflow_path", type = str, default = "dataset/nanning/数据/2019元旦/outflow_data_15min.npy")
    parser.add_argument("--output_dir", type = str, default = "dataset/nanning/npy_data/")
    parser.add_argument("--train_output_dir", type = str, default = "nanning_trainning/")
    args = parser.parse_args()                                                                                   
    transform_and_save(args.inflow_path, args.outflow_path, args.output_dir, args.timestamp, args.train_output_dir, start_time, args.interval_minutes)
