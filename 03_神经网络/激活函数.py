import numpy as np
import matplotlib.pyplot as plt

# 激活函数：阶跃函数
'''
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
'''


def step_function(x):
    return np.array(x > 0, dtype=np.int32)  # 已经没有np.int这个了


x = np.arange(-5, 5, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()

# 激活函数：Sigmoid函数


def sigmoid(x):
    return 1/(1+np.exp(-x))


x = np.arange(-5, 5, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()

y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1, label='Step Function', linestyle='--')
plt.plot(x, y2, label='Sigmoid Function')
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.legend()
plt.show()

# ReLU函数


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1, 6)  # 指定y轴的范围
plt.show()
