import numpy as np
import os
import json
from datetime import datetime, timedelta
import sys
sys.path.append(os.getcwd())

def transform_and_save(input_path, output_dir, train_output_dir, start_time, time_interval):
    data = np.load(input_path)  
    n, t = data.shape
    
    new_t = t - 7 + 1
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)
    
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    time_interval = timedelta(minutes=time_interval)
    
    jsonl_path = os.path.join(train_output_dir, "train.jsonl")
    
    with open(jsonl_path, "w") as jsonl_file:
        for i in range(new_t):
            segment = data[:, i:i+7]  # 形状 (n, 7)
            output_path = os.path.join(output_dir, f"data_{i}.npy")
            segment = np.transpose(segment, (1, 0))
            np.save(output_path, segment)
            
            time_begin = start_time + i * time_interval
            time_str = time_begin.strftime("%Y-%m-%d %H:%M")
            
            json_entry = {"time_begin": time_str, "npy_data_path": output_path}
            jsonl_file.write(json.dumps(json_entry) + "\n")
            

start_time = "2020-09-10 10:10"  
interval_minutes = 15  # 时间间隔（单位：分钟）
input_npy = "data.npy"  # 替换为你的输入文件路径
output_directory = "dataset/nanning"  # 输出目录
train_output_dir = "naning_trainning"  # 输出目录
transform_and_save(input_npy, output_directory, train_output_dir, start_time, interval_minutes)
