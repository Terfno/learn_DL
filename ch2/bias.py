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

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
