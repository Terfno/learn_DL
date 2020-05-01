import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray: # x:input / return output of sigmoid func
  """
  ref:
    http://www.kamishima.net/mlmpyja/lr/sigmoid.html
  """
  sigmoid_range = 34.538776394910684
  x = np.clip(x, -sigmoid_range, sigmoid_range)

  return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray: # x:input / return output of softmax func
  """
  ref:
    https://qiita.com/shohei-ojs/items/d66783bcead3eb7efd82
    "axis = -1, keepdims = True" is 2dim and 1dim to share process
  """
  c = np.max(x, axis = -1, keepdims = True)
  exp_x = np.exp(x - c) # for overflow
  sum_exp_x = np.sum(exp_x, axis = -1, keepdims = True)
  y = exp_x / sum_exp_x

  return y


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float: # y:output of NN, t:correct label / return value of cross entropy error
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  delta = 1e-7 # to prevent log(0)
  err = -np.sum(t * np.log(y + delta))
  batch_size = y.shape[0] # to normalize
  return err / batch_size


def sigmoid_grad(x: np.ndarray) -> np.ndarray: # fast ver / x:input / return output of sigmoid func
  return (1.0 - sigmoid(x)) * sigmoid(x)
