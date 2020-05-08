import numpy as np


class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    self.x = x
    self.y = y
    out = x * y

    return out

  def backward(self, dout: np.ndarray) -> tuple:
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy


def main():
  mul = MulLayer()
  x = np.array([1, 2, 3])
  y = np.array([10, 100, 1000])

  print(mul.forward(x, y))
  print(mul.backward(1))


if __name__ == "__main__":
  main()
