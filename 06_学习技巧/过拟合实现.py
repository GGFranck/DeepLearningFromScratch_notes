import numpy as np
import sys
import os
import matplotlib.pyplot as plt
# fmt:off
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.multi_layer_net import *
from common.optimizer import SGD
# fmt:on

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[
                        100, 100, 100, 100, 100, 100], output_size=10)
network.weight_decay_lambda = 0.1
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(
            f"epoch: {epoch_cnt}, train acc: {train_acc}, test acc: {test_acc}")

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 同时绘制训练数据和测试数据accuracy的变化
plt.plot(train_acc_list, label='train acc')
plt.plot(test_acc_list, label='test acc')
plt.legend()
plt.show()
