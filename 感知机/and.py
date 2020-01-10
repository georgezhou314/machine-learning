# 感知机实现AND逻辑电路
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    # sum计算矩阵的逐项和
    return np.sum(x*w)+b > 0


flag = AND(0, 0)
print(flag)
flag = AND(1, 0)
print(flag)
flag = AND(0, 1)
print(flag)
flag = AND(1, 1)
print(flag)
