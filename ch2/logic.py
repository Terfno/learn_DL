def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1 * w1 + x2 * w2
  if tmp > theta:
    return 1
  else:
    return 0

def NAND(x1, x2):
  w1, w2, theta = -0.5, -0.5, -0.7
  tmp = x1 * w1 + x2 * w2
  if tmp > theta:
    return 1
  else:
    return 0

def OR(x1, x2):
  w1, w2, theta = 1, 1, 0.5
  tmp = x1 * w1 + x2 * w2
  if tmp > theta:
    return 1
  else:
    return 0

print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
