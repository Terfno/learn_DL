import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

dotAB = np.dot(A, B)
print(dotAB)

# C = np.array([[1, 2], [3, 4]])
# dotBC = np.dot(B, C)
# print(dotBC)
# 行列Bの1次元目と、Cの0次元目の要素数が一致しないのでエラー
