import numpy as np
import matplotlib.pyplot as plt


def ReLU(x):
    return np.maximum(0, x)


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = np.random.randn(node_num, node_num)*0.01
    # w = np.random.randn(node_num, node_num)/np.sqrt(node_num)
    w = np.random.randn(node_num, node_num)/np.sqrt(node_num/2)

    z = np.dot(x, w)
    a = ReLU(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
