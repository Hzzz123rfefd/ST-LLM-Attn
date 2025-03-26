import os
import numpy as np
import matplotlib.pyplot as plt

true_array = np.load('true_array.npy')
pred_array = np.load('pred_array.npy')
os.makedirs("result", exist_ok=True)
plt.rcParams.update({'font.size': 10, 'font.weight': 'light'})

t, d = true_array.shape
for i in range(d):
    true = true_array[0:200, i]
    pred = pred_array[0:200, i]
    plt.figure(figsize=(10, 6))
    plt.plot(true, label='True', linewidth=1)
    plt.plot(pred, label='Predicted', linewidth=1)
    plt.title(f'station{i}')
    plt.xlabel('timestemp')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(f'result/站点{i}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close() 
