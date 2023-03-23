import matplotlib.pyplot as plt
import numpy as np

latency = np.array([0.052, 0.054, 0.065, 0.13, 0.17, 0.23])
x = np.arange(len(latency))
plt.bar(x, latency)
plt.show()
