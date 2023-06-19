import numpy as np

# 画激活函数图

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def tanh(x):
    return np.tanh(x)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# sigmoid
x = np.linspace(-4, 4, 1000)

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_position(('data', 0))
ax[0].spines['left'].set_position(('data', 0))

ax[0].set_xticks(np.arange(-4, 5, 2))
ax[0].set_yticks(np.arange(0, 1.1, 0.5))
ax[0].tick_params(labelsize=20)
ax[0].tick_params(axis='y', which='both', length=0, labelbottom=True)
ax[0].plot(x, sigmoid(x), color='skyblue', linewidth=3)
ax[0].set_title('Sigmoid', fontsize=26, color='skyblue')
# relu
x = np.linspace(-4, 4, 100)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_position(('data', 0))
ax[1].spines['left'].set_position(('data', 0))

ax[1].set_xticks(np.arange(-4, 5, 2))
ax[1].set_yticks(np.arange(0, 4.1, 1))
ax[1].tick_params(labelsize=20)
ax[1].plot(x, relu(x), color='skyblue', linewidth=3)
ax[1].set_title('ReLU', fontsize=26, color='skyblue')
# Tanh
x = np.linspace(-4, 4, 1000)

ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].spines['bottom'].set_position(('data', 0))
ax[2].spines['left'].set_position(('data', 0))

ax[2].set_xticks(np.arange(-4, 5, 2))
ax[2].set_yticks(np.arange(-1, 1.1, 0.5))
ax[2].tick_params(labelsize=20)
ax[2].plot(x, tanh(x), color='skyblue', linewidth=3)
ax[2].set_title('Tanh', fontsize=26, color='skyblue')

plt.savefig('acticate.jpg')