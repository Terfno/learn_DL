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

  def backward(self, dout): # dout:diff value of nn output / return dx
    dx = dout * (1.0 - self.out) * self.out

    return dx


