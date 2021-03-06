import numpy as np
from func import *

class Relu:
  def __init__(self):
    self.mask = None # np.ndarray of True and False

  def forward(self, x: np.ndarray) -> np.ndarray: # x:sum of inputs / return jadgment results
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0

    return out

  def backward(self, dout: np.ndarray) -> np.ndarray: # dout:diff value of nn output / return dx
    dout[self.mask] = 0
    dx = dout

    return dx


class Sigmoid:
  def __init__(self):
    self.out = None

  def forward(self, x: np.ndarray): # x:sum of input / return jadgment results
    out = sigmoid(x)
    self.out = out

    return out

  def backward(self, dout: np.ndarray) -> np.ndarray: # dout:diff value of nn output / return dx
    dx = dout * (1.0 - self.out) * self.out

    return dx


class Affine:
  def __init__(self, W: np.ndarray, b: np.ndarray):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x: np.ndarray) -> np.ndarray: # x:input to NN / return sum of bias and product of input to NN and Weights
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out

  def backward(self, dout: np.ndarray) -> np.ndarray: # dout: diff value of nn output / return dx
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx


class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None # loss
    self.y = None # outpu of softmax
    self.t = None # data of teacher(one-hot vec)

  def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray: # x:input to nn, t:data of teacher / return value of cross entropy error
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss

  def backward(self, dout: np.ndarray=1) -> np.ndarray: # dout: diff value of nn output / return dx
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx

