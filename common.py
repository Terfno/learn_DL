import numpy as np

# activation func
def step_funciton(x):
  y = x > 0
  return y.astype(np.int)

def sigmoid(x):
  sigmoid_range = 34.538776394910684
  x2 = np.maximum(np.minimum(x, sigmoid_range), -sigmoid_range)

  return 1 / (1 + np.exp(-x2))

def relu(x):
  return np.maximum(0, x)

# output
def softmax(a):
  c = np.max(a) # get max num
  exp_a = np.exp(a - c) # sub largest value from each element

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
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x) # f(x+h)

    x[idx] = tmp_val - h
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)

    x[idx] = tmp_val # 値を元に戻す
    it.iternext()

  return grad

def sigmoid_grad(x):
  return (1.0 - sigmoid(x)) * sigmoid(x)
