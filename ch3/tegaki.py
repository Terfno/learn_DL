import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをimportするために
import numpy as np
import pickle
from PIL import Image
from dataset.mnist import load_mnist


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y


def get_data():
  (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
  return x_test, t_test


def init_network():
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)
  return network


def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  z1 = sigmoid(np.dot(x, W1) + b1)
  z2 = sigmoid(np.dot(z1, W2) + b2)
  y = softmax(np.dot(z2, W3)+ b3)

  return y


def main():
  x, t = get_data()
  network = init_network()

  accuracy_cnt = 0
  for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
      accuracy_cnt += 1
  
  print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == "__main__":
  main()
