import numpy as np

def step_funciton(x): # xはnumpy.array
  y = x > 0 # xに対して不等号で演算すると、arrayの各要素に対して演算の結果がbooleanで入る。yはbooleanの配列
  return y.astype(np.int) # yを型変換→boolからintへ

x = np.array([-1.0, 1.0, 2.0])
print(step_funciton(x))
