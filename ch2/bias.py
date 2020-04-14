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

print("and")
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

print("nand")
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))

print("or")
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
