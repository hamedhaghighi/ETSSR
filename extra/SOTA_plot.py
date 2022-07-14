import matplotlib.pyplot as plt
import numpy as np


all_flops = [4.917, 1.745, 1.102, 0.362, 0.533, 0.703]
all_psnrs = [37.49, 37.90, 38.12, 38.21, 38.45, 38.55]
colors = ['green', 'blue', 'yellow', 'red', 'red', 'red']
all_params = np.array([0.67, 1.42, 1.43, 0.47,0.62, 0.67]) * 100
plt.scatter(all_flops, all_psnrs, all_params, colors, "o")
plt.xlim([0, 10])
plt.ylim([37, 39])
plt.xlabel('FLOPs')
plt.ylabel('PSNR')
plt.legend()
plt.show()
