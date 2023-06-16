from contextlib import nullcontext
import numpy as np
import os
import sys
import torch
import warnings


# To remove the warning of torchvision:
warnings.filterwarnings('ignore', category=UserWarning)


MODEL_NAMES = ['alexnet', 'vgg16', 'vgg19']


class Model:
    def __init__(self, name, data, device='cpu'):
        if not name in MODEL_NAMES:
            raise ValueError(f'Model name "{name}" is not supported')
        self.name = name

        self.data = data
        self.device = device

        self.probs = torch.nn.Softmax(dim=1)

        self.load()

    def attrib(self, x, c=None, steps=3, iters=10):
        if c is None:
            y, c, l = self.run_pred(x)

        x = self.data.tr_norm_inv(x)
        x = np.uint8(np.moveaxis(x.numpy(), 0, 2) * 256)

        def _img_to_x(x):
            # m = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
            # s = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
            # x = (x / 255 - m) / s
            x = np.transpose(x, (2, 0, 1)) / 255
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            x = self.data.tr_norm(x).unsqueeze(0)
            #x = np.expand_dims(x, 0)
            #x = np.array(x)
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            return x

        def _iter(x):
            x = _img_to_x(x)
            x.requires_grad_()
            y = self.probs(self.net(x))[0, c]
            self.net.zero_grad()
            y.backward()
            return x.grad.detach().cpu().numpy()[0]

        def _thr(a, p=60):
            if p >= 100:
                return np.min(a)
            a_sort = np.sort(np.abs(a.flatten()))[::-1]
            s = 100. * np.cumsum(a_sort) / np.sum(a)
            i = np.where(s >= p)[0]
            return a_sort[i[0]]

        ig = []
        for _ in range(iters):
            x0 = 255. * np.random.random(x.shape)
            xs = [x0 + 1.*i/steps * (x-x0) for i in range(steps)]

            g = [_iter(x_) for x_ in xs]
            g_avg = np.average(g, axis=0)
            g_avg = np.transpose(g_avg, (1, 2, 0))

            x_delta = _img_to_x(x) - _img_to_x(x0)
            x_delta = x_delta.detach().squeeze(0).cpu().numpy()
            x_delta = np.transpose(x_delta, (1, 2, 0))

            ig.append(x_delta * g_avg)

        a = np.average(np.array(ig), axis=0)

        a = np.average(np.clip(a, 0., 1.), axis=2)
        m = _thr(a, 1)
        e = _thr(a, 100)
        a_thr = (np.abs(a) - e) / (m - e)
        a_thr *= np.sign(a)
        a_thr *= (a_thr >= 0.)
        x = np.expand_dims(np.clip(a_thr, 0., 1.), 2) * [0, 255, 0]

        x = np.moveaxis(x, 2, 0)
        x = 0.2989 * x[0, :, :] + 0.5870 * x[1, :, :] + 0.1140 * x[2, :, :]
        return x / np.max(x)

    def check(self, tst=True, only_one_batch=False):
        data = self.data.dataloader_tst if tst else self.data.dataloader_trn
        n, m = 0, 0

        for x, l_real in data:
            y = self.run(x)
            l = torch.argmax(y, axis=1).detach().to('cpu')
            m += (l == l_real).sum()
            n += len(l)

            if only_one_batch:
                break

        return n, m

    def load(self):
        self.net = None

        if self.name in ['alexnet', 'vgg16', 'vgg19']:
            if self.data.name != 'imagenet':
                msg = f'Model "{self.name}" is ready only for "imagenet"'
                raise NotImplementedError(msg)

            # TODO: set path to data

            self.net = torch.hub.load('pytorch/vision:v0.10.0', self.name,
                weights=True)

        if self.net is not None:
            self.net.to(self.device)
            self.net.eval()

    def run(self, x, with_grad=False):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]
        x = x.to(self.device)

        with nullcontext() if with_grad else torch.no_grad():
            y = self.net(x)
            y = self.probs(y)

        return y if is_batch else y[0]

    def run_pred(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]

        y = self.run(x).detach().to('cpu').numpy()

        c = np.argmax(y, axis=1)
        y = np.array([y[i, c_cur] for i, c_cur in enumerate(c)])
        l = [self.data.labels[c_cur] for c_cur in c]

        return (y, c, l) if is_batch else (y[0], c[0], l[0])
