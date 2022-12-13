import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

all_flops = [
    4.917,
    7.020,
    10.168,
    18.282,
    8.461,
    1.745,
    1.102,
    5.004,
    0.362,
    0.533,
    0.703]
all_psnrs = [
    37.49,
    38.08,
    38.32,
    38.09,
    36.18,
    37.90,
    38.12,
    36.62,
    38.21,
    38.45,
    38.55]
colors = [
    'green',
    'blue',
    'yellow',
    'cyan',
    'brown',
    'purple',
    'orange',
    'pink',
    'red',
    'red',
    'red']
labels = [
    'VDSR',
    'RCAN',
    'RDN',
    'EDSR',
    'StereoSR',
    'PASSRnet',
    'iPASSR',
    'SSRDEFNet',
    'ETSSR_S (Ours)',
    'ETSSR_B (Ours)',
    'ETSSR_L (Ours)']
all_params = np.array([0.67, 15.36, 22.04, 38.90, 1.15,
                      1.42, 1.43, 2.26, 0.47, 0.62, 0.67]) * 100
plt.rcParams.update({'font.family': 'sans-serif'})
plt.scatter(all_flops, all_psnrs, all_params, colors, "o")

for i, txt in enumerate(labels):
    plt.annotate(txt, (all_flops[i], all_psnrs[i]), fontsize=9)

x_tick = []
x_tick.extend(list(range(0, 25, 5)))
x_tick.extend(list(range(0, 5, 1)))
y_tick = []
y_tick.extend(list(np.arange(36, 39, 0.5)))
y_tick.extend(list(np.arange(37.8, 39, 0.2)))
y_tick = list(sorted(y_tick))
# plt.title('Quantitative Compari')
plt.xticks(x_tick)
plt.yticks(y_tick)
plt.xlim([0, 21])
plt.ylim([36, 39])
plt.xlabel('FLOPs (T)')
plt.ylabel('PSNR (dB)')
plt.grid(linewidth=0.5, linestyle='--', alpha=0.3)
# plt.legend()
plt.show()
