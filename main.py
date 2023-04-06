import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sys
import teneva
from time import perf_counter as tpc
import torch


sys.path.append('../neural_tensor_train')
from manager import Manager as ManagerBase
from opt import opt_protes
from opt import opt_ttopt
from utils import sort_vector


class Manager(ManagerBase):
    def __init__(self, data='cifar10', gen=None, model='densenet', task='attack', kind='base', root='result', use_ttopt=False):
        super().__init__(data, gen, model, task, kind, root=root)

    def task_attack_base(self, n=5, m=int(1.E+5), k=50, k_top=5, sc=100):
        self._set_image()
        self._plot_real()

        self.h = float((self.x_max - self.x_min) / sc)

        def func(I):
            s = I.shape[0]
            I = (I - (n-1)/2) * self.h
            I = np.array(I).reshape(-1, self.ch, self.sz, self.sz)
            x = np.repeat(self.x_real_np[None], s, axis=0) + I
            x = torch.tensor(x, dtype=torch.float32)
            y = self.model.run(x).detach().to('cpu').numpy()
            y = y[:, self.c_targ] - y[:, self.c_real]
            return y

        if use_ttopt:
            tm = self.log.prc(f'Optimization with TTOpt method:')
            i, _, hist = opt_ttopt(func, self.d, n, m, is_max=True)
            self.log.res(tpc()-tm)
        else:
            tm = self.log.prc(f'Optimization with PROTES method:')
            i, _, hist = opt_protes(func, self.d, n, m, k=k, k_top=k_top,
                with_qtt=False, is_max=True)

        i = np.array(i).reshape(self.ch, self.sz, self.sz)
        self.z_chng = self.h * torch.tensor(i)
        self.x_chng = self.x_real + self.z_chng

        res_pred, res_base = self._get_pred(self.x_chng, self.l_pred)
        self.y_pred_chng, self.c_pred_chng, self.l_pred_chng = res_pred
        self.y_base_chng, self.c_base_chng, self.l_base_chng = res_base

        self._plot_chng()

    def _get_pred(self, x, l_targ=None):
        y = self.model.run(x).detach().to('cpu').numpy()
        vals = sort_vector(y)
        c_pred, y_pred = vals[0]
        c_targ, y_targ = vals[1]

        if l_targ:
            c_targ = [c for c, l in self.data.labels.items() if l == l_targ][0]
            y_targ = [y for c, y in vals if c == c_targ][0]
        else:
            c_targ, y_targ = vals[1]

        l_pred = self.data.labels[c_pred]
        l_targ = self.data.labels[c_targ]

        return (y_pred, c_pred, l_pred), (y_targ, c_targ, l_targ)

    def _plot(self, x, title, fpath):
        x = self.data.tensor_to_plot_cifar10(x)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.detach().to('cpu').squeeze()

        fig = plt.figure(figsize=(3.3, 4.3))
        plt.imshow(x)
        plt.title(title, fontsize=11, loc='left',
            fontdict={'family':'monospace'})
        plt.axis('off')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close(fig)

    def _plot_real(self):
        title = ''
        # title += f'Real             | {self.l_real}\n'
        title += f'Pred {self.y_pred:-8.2e} | {self.l_pred}\n'
        title += f'Targ {self.y_targ:-8.2e} | {self.l_targ}'
        self._plot(self.x_real, title, self.get_path('img/real.png'))

    def _plot_chng(self):
        title = ''
        title += f'Pred {self.y_pred_chng:-8.2e} | {self.l_pred_chng}\n'
        title += f'Base {self.y_base_chng:-8.2e} | {self.l_base_chng}'
        self._plot(self.x_chng, title, self.get_path('img/chng.png'))

        title = ''
        title += f'Pred {self.y_pred_chng:-8.2e} | {self.l_pred_chng}\n'
        title += f'Base {self.y_base_chng:-8.2e} | {self.l_base_chng}'
        self._plot(self.z_chng, title, self.get_path('img/chng_delta.png'))

    def _set_image(self, i=10, log=True):
        self.ch = self.data.ch
        self.sz = self.data.sz
        self.d = self.ch * self.sz * self.sz
        self.x_real, self.c_real, self.l_real = self.data.get(i, tst=True)
        self.x_real_np = self.x_real.detach().to('cpu').numpy()
        self.x_avg = torch.mean(self.x_real)
        self.x_max = torch.max(self.x_real)
        self.x_min = torch.min(self.x_real)
        if log:
            text = f'Image : [{self.x_min:-8.1e}, {self.x_max:-8.1e}] '
            text += f'(avg: {self.x_avg:-8.1e})'
            print(text)

        res_pred, res_targ = self._get_pred(self.x_real)
        self.y_pred, self.c_pred, self.l_pred = res_pred
        self.y_targ, self.c_targ, self.l_targ = res_targ


def args_build():
    parser = argparse.ArgumentParser(
        prog='tetradat',
        description='Software product for generation of adversarial examples for artificial neural networks using tensor train (TT) decomposition and optimizers based on it, i.e., TTOpt and PROTES methods.',
        epilog = '© Andrei Chertkov'
    )
    parser.add_argument('-d', '--data',
        type=str,
        help='Name of the used dataset',
        default='cifar10',
        choices=['mnist', 'mnistf', 'cifar10', 'imagenet']
    )
    parser.add_argument('-g', '--gen',
        type=str,
        help='Name of the used generator',
        default=None,
        choices=['gan_sn', 'vae_vq']
    )
    parser.add_argument('-m', '--model',
        type=str,
        help='Name of the used model',
        default=None,
        choices=['densenet', 'vgg16']
    )
    parser.add_argument('-t', '--task',
        type=str,
        help='Name of the task',
        default=None,
        choices=['check', 'train', 'am']
    )
    parser.add_argument('-k', '--kind',
        type=str,
        help='Kind of the task',
        default=None,
    )
    parser.add_argument('-c', '--c',
        type=str,
        help='Target class',
        default=None,
    )
    parser.add_argument('-l', '--l',
        type=str,
        help='Target layer',
        default=None,
    )
    parser.add_argument('-f', '--f',
        type=str,
        help='Target filter',
        default=None,
    )
    parser.add_argument('-r', '--root',
        type=str,
        help='Path to the folder with results',
        default='result'
    )

    args = parser.parse_args()
    return args.data, args.gen, args.model, args.task, args.kind, args.c, args.l, args.f, args.root


if __name__ == '__main__':
    Manager() # *args_build()
