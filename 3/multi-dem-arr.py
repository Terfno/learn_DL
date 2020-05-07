import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

dotAB = np.dot(A, B)
print(dotAB)

# C = np.array([[1, 2], [3, 4]])
# dotBC = np.dot(B, C)
# print(dotBC)
# 行列Bの1次元目と、Cの0次元目の要素数が一致しないのでエラー

InputX = np.array([1, 2])
WeightW = np.array([[1, 3, 5], [2, 4, 6]])
OutputY = np.dot(InputX, WeightW)

print(OutputY)
