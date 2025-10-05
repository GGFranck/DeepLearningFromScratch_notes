import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy(y, t):
    delta = 1e-7  # 防止log(0)情况
    return -np.sum(t * np.log(y + delta))


y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

mse = mean_squared_error(np.array(y), np.array(t))
print(mse)

ce = cross_entropy(np.array(y), np.array(t))
print(ce)
