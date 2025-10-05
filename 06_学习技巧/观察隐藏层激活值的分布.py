import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = np.random.randn(node_num, node_num)*1
    # w = np.random.randn(node_num, node_num)*0.01
    w = np.random.randn(node_num, node_num)/np.sqrt(node_num)

    z = np.dot(x, w)
    a = sigmoid(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()


# 用tanh函数
def tanh(x):
    return np.tanh(x)


# 使用tanh函数进行激活
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    w = np.random.randn(node_num, node_num)/np.sqrt(node_num)

    z = np.dot(x, w)
    a = tanh(z)
    activations[i] = a
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(), 30, range=(-1, 1))
plt.show()
