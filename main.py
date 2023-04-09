import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sys
from time import perf_counter as tpc
import torch


sys.path.append('../neural_tensor_train')
from manager import Manager as ManagerBase
from model import Model
from opt import opt_protes
from opt import opt_ttopt
from utils import sort_matrix
from utils import sort_vector


class Manager(ManagerBase):
    def __init__(self, data='imagenet', gen=None, model='alexnet', model_attr='vgg19', task='attack', kind='attr', root='result'):
        super().__init__(data, gen, model, task, kind, root=root)

        self.model_attr_name = model_attr

        tm = self.log.prc(f'Loading "{self.model_attr_name}" model')
        self.model_attr = Model(self.model_attr_name, self.data, self.device)
        self.log.res(tpc()-tm)
        self.log('')

    def task_attack_attr(self, d=250, n=3, m=1.E+5, k=50, k_top=5, sc=5, attr_steps=3, attr_iters=10, imgs=np.arange(100, 200)):
        # imgs=[10, 13, 20, 35, 507, 563, 600, 669, 680, 689, 694]
        self.d_pix = d
        self.n = n
        self.m = int(m)
        self.opt_k = k
        self.opt_k_top = k_top
        self.opt_sc = sc

        for i in imgs:
            self._set_image(i)
            self.x_attr = self.model_attr.attrib(self.x_real,
                steps=attr_steps, iters=attr_iters)
            self.pix = sort_matrix(self.x_attr)[:self.d_pix]
            self._attack()
            self._plot()

    def _attack(self, use_ttopt=False):
        self.h = float((self.x_max - self.x_min) / self.opt_sc)

        def change(I):
            s = I.shape[0]
            dx = (I - (self.n-1)/2) * self.h
            dx = np.array(dx) # .reshape(-1, self.ch, self.sz, self.sz)

            x = np.repeat(self.x_real_np[None], s, axis=0)

            for j, p in enumerate(self.pix):
                x[:, 0, p[0], p[1]] += dx[:, j]
                x[:, 1, p[0], p[1]] += dx[:, j]
                x[:, 2, p[0], p[1]] += dx[:, j]

            return torch.tensor(x, dtype=torch.float32)

        def change_one(i):
            return change(i.reshape(1, -1))[0]

        def get_changes(i):
            z = np.zeros((self.data.sz, self.data.sz), dtype=int)
            dx = (i - (self.n-1)/2) * self.h
            for j, p in enumerate(self.pix):
                if dx[j] > 0.: z[p[0], p[1]] = +1
                if dx[j] < 0.: z[p[0], p[1]] = -1
            return z

        def func(I):
            y = self.model.run(change(I)).detach().to('cpu').numpy()
            return y[:, self.c_targ] - y[:, self.c_real]

        if use_ttopt:
            tm = self.log.prc(f'Optimization with TTOpt method:')
            i, e, hist = opt_ttopt(func, self.d_pix, self.n, self.m,
                is_max=True)
        else:
            tm = self.log.prc(f'Optimization with PROTES method:')
            i, e, hist = opt_protes(func, self.d_pix, self.n, self.m,
                k=self.opt_k, k_top=self.opt_k_top, with_qtt=False, is_max=True)

        print(f'Result is : {e:-7.1e}')
        self.log.res(tpc()-tm)

        self.z_chng = get_changes(i)
        self.x_chng = change_one(i)

        res_pred, res_base = self._get_pred(self.x_chng, self.c_pred)
        self.y_pred_chng, self.c_pred_chng, self.l_pred_chng = res_pred
        self.y_base_chng, self.c_base_chng, self.l_base_chng = res_base

    def _get_pred(self, x, c_targ=None):
        y = self.model.run(x).detach().to('cpu').numpy()
        vals = sort_vector(y)

        c_pred, y_pred = vals[0]

        if c_targ:
            #c_targ = [c for c, l in self.data.labels.items() if l == l_targ][0]
            y_targ = [y for c, y in vals if c == c_targ][0]
        else:
            c_targ, y_targ = vals[1]

        l_pred = self.data.labels[c_pred]
        l_targ = self.data.labels[c_targ]

        return (y_pred, c_pred, l_pred), (y_targ, c_targ, l_targ)

    def _plot(self):
        def prep(x):
            x = torch.tensor(x) if not torch.is_tensor(x) else x
            x = self.data.tr_norm_inv(x)
            x = x.detach().to('cpu').squeeze().numpy()
            x = np.clip(x, 0, 1) if np.mean(x) < 2 else np.clip(x, 0, 255)
            x = x.transpose(1, 2, 0)
            return x

        def prep_2d(x):
            x = torch.tensor(x) if not torch.is_tensor(x) else x
            x = x.detach().to('cpu').squeeze().numpy()
            x = np.clip(x, 0, 1) if np.mean(x) < 2 else np.clip(x, 0, 255)
            return x

        def draw(x, title, cmap=None, is_left=True):
            plt.imshow(x, cmap=cmap)
            plt.title(title, fontsize=11, loc='left' if is_left else 'center',
                fontdict={'family':'monospace'})
            plt.axis('off')

        def cut(text, l=30):
            return text if len(text) < l else text[:l-3] + '...'

        fig = plt.figure(figsize=(12, 12))

        fig.add_subplot(2, 2, 1)
        title =  f'Pred {self.y_pred:-8.2e} | {cut(self.l_pred)}\n'
        title += f'Targ {self.y_targ:-8.2e} | {cut(self.l_targ)}'
        draw(prep(self.x_real), title)

        fig.add_subplot(2, 2, 2)
        title =  f'Pred {self.y_pred_chng:-8.2e} | {cut(self.l_pred_chng)}\n'
        title += f'Base {self.y_base_chng:-8.2e} | {cut(self.l_base_chng)}'
        draw(prep(self.x_chng), title)

        fig.add_subplot(2, 2, 3)
        title = f'Attribution from "{self.model_attr_name}" model'
        draw(prep_2d(self.x_attr), title, cmap='hot', is_left=False)

        fig.add_subplot(2, 2, 4)
        num = np.sum(np.abs(self.z_chng) > 0)
        title = f'Changed (# {num}) pixels for "{self.model_name}" model'
        draw(self.z_chng, title,
            cmap=mcolors.ListedColormap(['white', 'black', 'red']),
            is_left=False)
        plt.gca().set_facecolor('black')

        fpath = f'img/attack_with-{self.model_attr_name}_{self.i}.png'
        plt.savefig(self.get_path(fpath), bbox_inches='tight')
        plt.close(fig)

    def _set_image(self, i, log=True):
        self.i = i
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


if __name__ == '__main__':
    Manager(model='alexnet', model_attr='vgg19').run()
    Manager(model='vgg19', model_attr='alexnet').run()
