import sys, os
sys.path.append(os.pardir)
from common import *
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y= softmax(a2)

    return y

  def loss(self, x, t):
    y = self.predict(x)
    return cross_entropy_err(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis = 1)
    t = np.argmax(t, axis = 1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads

  def gradient(self, x, t):
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
    grads['b2'] = np.sum(dy, axis = 0)

    dz1 = np.dot(dy, W2.T)
    da1 = sigmoid_grad(a1) * dz1
    grads['W1'] = np.dot(x.T, da1)
    grads['b1'] = np.sum(da1, axis = 0)

    return grads


def main():
  # load data
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

  network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

  # hyper params
  Iters_num = 10000
  Train_size = x_train.shape[0]
  Batch_size = 100
  Learning_rate = 0.1

  train_loss_list = []
  train_acc_list = []

  iter_per_epoch = max(Train_size / Batch_size, 1)

  for i in range(Iters_num):
    # get mini batch
    batch_mask = np.random.choice(Train_size, Batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # gradient
    grad = network.gradient(x_batch, t_batch)

    # update params
    for key in ('W1', 'b1', 'W2', 'b2'):
      network.params[key] -= Learning_rate * grad[key]

    # log
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # epoch
    if i % iter_per_epoch == 0:
      train_acc = network.accuracy(x_train, t_train)
      test_acc = network.accuracy(x_test, t_test)
      train_acc_list.append(test_acc)
      print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

  # draw graph
  x = np.arange(len(train_acc_list))
  plt.plot(x, train_acc_list, label='train acc')
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.ylim(0, 1.0)
  plt.legend(loc='lower right')
  plt.savefig('l.png')


if __name__ == "__main__":
  main()
