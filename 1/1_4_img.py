import matplotlib.pyplot as plt
from matplotlib.image import imread

lena = imread('../dataset/lena.png')
plt.imshow(lena)

plt.savefig('1_4z.png')
