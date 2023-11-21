import matplotlib
import numpy as np
import random
import torch


from data import Data
from model import Model
from attack import color_hsv_to_rgb
from attack import color_rgb_to_hsv


seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

data = Data('imagenet')
model = Model('alexnet', data)
model_atttr = Model('vgg', data)

RESULT_SHOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 15, 20, 21, 22, 23, 24, 25, 44, 56, 74, 88, 95, 97, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 254, 300, 500, 583, 999]

for i in RESULT_SHOW:

    x, c, l = data.get(i, tst=True)
    data.plot_base(data.tr_norm_inv(x), '', size=6, fpath='image.png')

    z = data.tr_norm_inv(x)
    z = z.numpy()
    z = np.moveaxis(z, 0, -1)
    z = matplotlib.colors.rgb_to_hsv(z)
    #z = color_rgb_to_hsv(z)
    z = np.moveaxis(z, -1, 0)

    for k in range(500):
        i = np.random.choice(np.arange(224))
        j = np.random.choice(np.arange(224))
        z[1, i, j] -= 0.3 * z[1, i, j]
        z[1, i, j] = max(0, z[1, i, j])
    for k in range(500):
        i = np.random.choice(np.arange(224))
        j = np.random.choice(np.arange(224))
        z[2, i, j] += 0.3 * (1 - z[2, i, j])
        z[2, i, j] = max(0, z[2, i, j])
    #z = color_hsv_to_rgb(z)
    z = np.moveaxis(z, 0, -1)
    z = np.moveaxis(matplotlib.colors.hsv_to_rgb(z), -1, 0)
    data.plot_base(z, '', size=6, fpath=f'image_tmp_{i}.png')

y_all = model.run(x).detach().to('cpu').numpy()
y = y_all[c]

c_target = np.argsort(y_all)[::-1][10]
y_target = y_all[c_target]

print(f'Predicted class : {c:-4d} (y = {y:-7.1e}) [{data.labels[c]}]')
print(f'Target    class : {c_target:-4d} (y_target = {y:-7.1e}) [{data.labels[c_target]}]')

if False:
    x_attr = model.attrib(x, c, steps=10, iters=10)
    data.plot_attr(x_attr, fpath='image_attr.png')

    x_attr_target = model.attrib(x, c_target, steps=10, iters=10)
    data.plot_attr(x_attr_target, fpath='image_attr_target.png')
