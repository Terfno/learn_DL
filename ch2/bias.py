import numpy as np

def AND(x1, x2):
  input = np.array([x1, x2])
  weight = np.array([0.5, 0.5])
  bias = -0.7

  tmp = np.sum(weight * input) + bias
  if tmp <= 0:
    return 0
  else:
    return 1

def NAND(x1, x2):
  input = np.array([x1, x2])
  weight = np.array([-0.5, -0.5])
  bias = 0.7

  tmp = np.sum(weight * input) + bias
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  input = np.array([x1, x2])
  weight = np.array([0.5, 0.5])
  bias = -0.2

  tmp = np.sum(weight * input) + bias
  if tmp <= 0:
    return 0
  else:
    return 1

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)

  y = AND(s1, s2)
  
  return y

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))