import sys
import os
sys.path.append(os.getcwd())
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    inflow_trues = np.load(args.save_dir + "inflow_trues.npy")
    inflow_preds = np.load(args.save_dir + "inflow_preds.npy")
    outflow_trues = np.load(args.save_dir + "outflow_trues.npy")
    outflow_preds = np.load(args.save_dir + "outflow_preds.npy")
    t, d = inflow_trues.shape
    for i in range(d):
        true = inflow_trues[0:200, i]
        pred = inflow_preds[0:200, i]
        plt.figure(figsize=(10, 6))
        plt.plot(true, label='True', linewidth=1)
        plt.plot(pred, label='Predicted', linewidth=1)
        plt.title(f'station{i}')
        plt.xlabel('timestemp')
        plt.ylabel('value')
        plt.legend()
        plt.savefig(args.save_dir + f'站点{i}_inflow.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close() 
        true = outflow_trues[0:200, i]
        pred = outflow_preds[0:200, i]
        plt.figure(figsize=(10, 6))
        plt.plot(true, label='True', linewidth=1)
        plt.plot(pred, label='Predicted', linewidth=1)
        plt.title(f'station{i}')
        plt.xlabel('timestemp')
        plt.ylabel('value')
        plt.legend()
        plt.savefig(args.save_dir + f'站点{i}_outflow.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type = str, default = "result/")
    args = parser.parse_args()
    main(args)