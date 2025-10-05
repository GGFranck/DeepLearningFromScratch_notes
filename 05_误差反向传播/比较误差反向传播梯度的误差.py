import sys
import os
import numpy as np
# fmt: off
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet
# fmt: on

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True, normalize=True, one_hot_label=True)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
    print(key + ":" + str(diff))
