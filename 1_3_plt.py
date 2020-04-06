import numpy as np
import matplotlib.pyplot as plt

# draw simple graph
## data
x = np.arange(0, 6, 0.1) # from 0 to 6 step 0.1
y = np.sin(x)

## draw
plt.plot(x, y)
plt.show()
