import matplotlib.pyplot as plt
import numpy as np

PSNR_swinT = np.array([38.22, 38.39, 38.57])

FLOPS_swinT = np.array([365.30, 537.21, 709.12])

PSNR_ipassr = np.array([37.84, 37.96, 38.10])

FLOPS_ipassr = np.array([315.63, 512.33, 777.77])

params_swinT = np.array([0.47, 0.62, 0.76]) * 100
params_ipassr = np.array([0.54, 0.80, 1.09]) * 100
plt.scatter(FLOPS_swinT, PSNR_swinT, params_swinT, 'green', "o")
plt.scatter(FLOPS_ipassr, PSNR_ipassr, params_ipassr, 'blue', "*")
plt.xlim([300, 1000])
plt.ylim([37.5, 39])
plt.show()
