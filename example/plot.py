import sys
import os
sys.path.append(os.getcwd())
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15 

def moving_average(x, window_size=5):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

def generate_custom_time_index(n_points, points_per_day=17, start_date=datetime(2024, 4, 1, 7, 0)):
    time_index = []
    for i in range(n_points):
        day_offset = i // points_per_day
        hour_offset = i % points_per_day
        time_point = start_date + timedelta(days=day_offset, hours=hour_offset)
        time_index.append(time_point)
    return time_index

def plot(inflow_trues, inflow_preds, outflow_trues, outflow_preds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    t, d = inflow_trues.shape
    time_index = generate_custom_time_index(n_points=119)
    for i in range(d):
            # inflow
            true = inflow_trues[-120:-1, i]
            pred = inflow_preds[-120:-1, i]

            true = moving_average(true)
            pred = moving_average(pred)
            
            plt.figure(figsize=(7, 5))
            plt.plot(time_index, true, label='True', linewidth=0.3, color='black')
            plt.plot(time_index, pred, label='Predicted', linewidth=0.3, color='red')
            # plt.xlabel('inflow ')
            plt.legend(loc='upper left')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # 每天一个主刻度
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_dir + f'站点{i}_inflow.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()

            # outflow
            true = outflow_trues[-120:-1, i]
            pred = outflow_preds[-120:-1, i]

            true = moving_average(true)
            pred = moving_average(pred)
            
            plt.figure(figsize=(7, 5))
            plt.plot(time_index, true, label='True', linewidth=0.3, color='black')
            plt.plot(time_index, pred, label='Predicted', linewidth=0.3, color='red')
            # plt.xlabel('outflow')
            plt.legend(loc='upper left')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())  
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_dir + f'站点{i}_outflow.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()


def main(args):
    inflow_trues = np.load(args.save_dir + "inflow_trues.npy")
    inflow_preds = np.load(args.save_dir + "inflow_preds.npy")
    outflow_trues = np.load(args.save_dir + "outflow_trues.npy")
    outflow_preds = np.load(args.save_dir + "outflow_preds.npy")
    plot(inflow_trues, inflow_preds, outflow_trues, outflow_preds, args.save_dir)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="result/")
    args = parser.parse_args()
    main(args)
