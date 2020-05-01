from func import *
import numpy as np

class TwoLayerNet:

  def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float = 0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)


  def predict(self, x: np.ndarray) -> np.ndarray: # prediction by NN / x:input to NN / return output of NN
    # get param
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    # calc of NN
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y


  def loss(self, x: np.ndarray, t: np.ndarray) -> float: # loss func / x:input to NN, t:correct label / return value of loss func
    # prediction
    y = self.predict(x)

    # calc of error
    loss = cross_entropy_error(y, t)

    return loss


  def accuracy(self, x: np.ndarray, t: np.ndarray) -> float: # x:input to NN, t:correct label / return value of recognition accuracy
    # prediction
    y = self.predict(x)

    # get index of max element
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    # "np.sum(y == t)" counts the number of matching elements for each element of y and t.
    # This is then divided by "x.shpae[0]" to get the percentage of matches per block.
    accuracy = np.sum(y == t) / x.shape[0]
    return accuracy


  def gradient(self, x:np.ndarray, t:np.ndarray) -> dict : # calc the gradient of weight / x:input to NN, t:correct label / return dictionary of gradient
    # get param
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    # backward
    dy = (y - t) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)

    dz1 = np.dot(dy, W2.T)
    da1 = sigmoid_grad(a1) * dz1
    grads['W1'] = np.dot(x.T, da1)
    grads['b1'] = np.sum(da1, axis=0)

    return grads
