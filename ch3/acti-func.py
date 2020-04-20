import numpy as np
import matplotlib.pyplot as plt

def step_funciton(x): # xはnumpy.array
  y = x > 0 # xに対して不等号で演算すると、arrayの各要素に対して演算の結果がbooleanで入る。yはbooleanの配列
  return y.astype(np.int) # yを型変換→boolからintへ

def sigmoid(x):
  return 1 / (1 + np.exp(-x)) # numpyのbroadcastによって、np.arrayに対応できている

def relu(x):
  return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = step_funciton(x)
plt.plot(x, y)

y = sigmoid(x)
plt.plot(x, y)

y = relu(x)
plt.plot(x, y)

plt.savefig('acti.png')
