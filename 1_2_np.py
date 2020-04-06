import numpy as np

# create array
## for np array: numpy.ndarray
ndarray = np.array([1.0, 2.0, 3.0])

print(ndarray)
print(type(ndarray))

## calc array with np
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

### element wise
print(x + y)
print(x - y)
print(x * y)
print(x / y)

# N-dim array
a = np.array([[1, 2], [3, 4], [5, 6]])

print(a)
print(a.shape)
print(a.dtype)

## calc N-dim array with np
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[6, 5], [4, 3], [2, 1]])

### shape of X, Y is the same.
print(X + Y)
print(X - Y)
print(X * Y)
print(X / Y)

# broadcast
## shape A, B is not same: it gets transformed nicely.
A = np.array([[1, 2], [3, 4]]) # 2x2
B = 10 # 1x1
print(A * B) # [[10 20] [30 40]]

B = [10, 20]
print(A * B) # [[10 40] [30 80]]

# access to element nicely.
Z = np.array([[39, 10], [17, 27], [87, 53], [0, 4]])
print(Z)
print(Z[0])
print(Z[1][1])

Zf = Z.flatten() # convert to 1-Dim array
print(Zf)
print(Zf[0])
print(Zf[np.array([1, 3, 4])]) # Zf takes one argument. You can access multiple indexes at once by passing a single array for this argument.
print(Zf[Zf>15]) # similr method
