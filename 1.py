import numpy as np
a = np.load("dataset/nanning/数据/2019元旦/inflow_data_15min.npy")

print("shape:", a.shape)
# 计算最小值和最大值
a_min = np.min(a)
a_max = np.max(a)

print("最大值：", a_min)
print("最小值：", a_max)

if a_max - a_min != 0:
    a_norm = (a - a_min) / (a_max - a_min)
else:
    a_norm = np.zeros_like(a)

np.save("data.npy", a_norm)