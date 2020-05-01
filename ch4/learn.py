import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append(os.pardir)
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

def graph_loss(train_loss_list: list):
  print("now creating graph of loss")
  x = np.arange(len(train_loss_list))
  plt.plot(x, train_loss_list, label='loss')
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.xlim(left=0)
  plt.ylim(bottom=0)
  plt.savefig('loss.png')


def graph_acc(train_acc_list: list, test_acc_list: list):
  print("now creating graph of accuracy")
  x2 = np.arange(len(train_acc_list))
  plt.plot(x2, train_acc_list, label='train acc')
  plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.xlim(left=0)
  plt.ylim(0, 1.0)
  plt.legend(loc='lower right')
  plt.savefig('acc.png')


def main():
  # get mnist train data and test data
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

  # hyper param
  iters_num = 10000
  batch_size = 100
  learning_rate = 0.1

  # log of result
  train_loss_list = []
  train_acc_list = []
  test_acc_list = []

  # size of train data
  train_size = x_train.shape[0]

  iter_per_epoch = max(train_size / batch_size, 1)

  # init tow layer nn
  network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

  # learning
  for i in range(iters_num):
    # mini batch
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calc gradient
    grad = network.gradient(x_batch, t_batch)

    # update weight param
    for key in ('W1', 'b1', 'W2', 'b2'):
      network.params[key] -= learning_rate * grad[key]

    # calc value of loss
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # calc value of recognition accuracy per 1 epoch
    if i % iter_per_epoch == 0:
      train_acc = network.accuracy(x_train, t_train)
      test_acc = network.accuracy(x_test, t_test)
      train_acc_list.append(train_acc)
      test_acc_list.append(test_acc)

      # print log
      print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

  # graph_loss(train_loss_list)
  # graph_acc(train_acc_list, test_acc_list)

  print("done.")


if __name__ == "__main__":
  main()
