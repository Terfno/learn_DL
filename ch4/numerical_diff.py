import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
  h = 1e-4
  return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x) # xと同じ計上の0配列

  for idx in range(x.size):
    tmp_val = x[idx]

    # f(x+h)
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x-h)
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val # 元の値を代入し直す
  
  return grad

def function_1(x):
  return 0.01*x**2 + 0.1*x

def function_2(x):
  return np.sum(x**2)

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.savefig("diff.png")

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
