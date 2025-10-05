import sys
import os
import numpy as np
# fmt: off
sys.path.append(r"D:\Data\知识库\深度学习\深度学习入门_鱼书\代码\03_神经网络\MNIST")
from mnist import load_mnist
# fmt: on

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
