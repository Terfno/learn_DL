import numpy as np

# activation func
def step_funciton(x):
  y = x > 0
  return y.astype(np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)

# output
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

def identity_function(x):
  return x

# loss func
def mean_squared_err(y, t):
  return 0.5 * np.sum((y-t)**2)

def cross_entropy_err(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))

# gradient
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
