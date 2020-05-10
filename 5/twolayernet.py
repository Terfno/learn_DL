import sys, os
sys.path.append(os.pardir)
import numpy as np
from layer import *
from collections import OrderedDict

class TwoLayerNet:
  def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
    # init weights
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    # generate layer
    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x: np.ndarray) -> np.ndarray: # prediction by NN / x:input to NN / return output of NN
    for layer in self.layers.values():
      x = layer.forward(x)

    return x

  def loss(self, x:np.ndarray, t: np.ndarray) -> float: # x:input to NN, t:correct label / return value of loss func
    y = self.predict(x)
    return self.lastLayer.forward(y, t)

  def accuracy(self, x: np.ndarray, t: np.ndarray) -> float: # x:input to NN, t:correct label / return value of recognition accuracy
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1:
      t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict: # calc the gradient of weights / x:input to NN, t:correct label / return dict of grad
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads

  def gradient(self, x: np.ndarray, t: np.ndarray) -> dict: # calc the gradient of weights / x:input to NN, t:correct label / return dict of gradient
    # forward
    self.loss(x, t)

    # backward
    dout = 1
    dout = self.lastLayer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    # config
    grads = {}
    grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
    grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

    return grads

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
