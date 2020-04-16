import numpy as np
import matplotlib.pyplot as plt

def step_funciton(x): # xはnumpy.array
  y = x > 0 # xに対して不等号で演算すると、arrayの各要素に対して演算の結果がbooleanで入る。yはbooleanの配列
  return y.astype(np.int) # yを型変換→boolからintへ

x = np.arange(-5.0, 5.0, 0.1)
y = step_funciton(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y座標の範囲を指定
plt.savefig('step-func.png')
