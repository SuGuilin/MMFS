import numpy as np

# 替换为你的 .npy 文件路径
file_path = '01598D.npy'

# 读取 .npy 文件
data = np.load(file_path)

# 打印数据的形状
print("Shape of the data:", data.shape)