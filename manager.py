import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import teneva
from time import perf_counter as tpc
import torch
import torchattacks


from data import DATA_NAMES
from data import Data
from model import MODEL_NAMES
from model import Model
from opt import opt
from utils import Log
from utils import sort_matrix
from utils import sort_vector


class Manager:
    def __init__(self, data, model, model_attr, task, kind, opt_d, opt_n, opt_m,
                 opt_k, opt_k_top, opt_k_gd, opt_lr, opt_r, opt_sc, attr_steps,
                 attr_iters, attack_num_max, root='result', device=None):
        self.data_name = data
        self.model_name = model
        self.model_attr_name = model_attr
        self.task = task
        self.kind = kind

        self.opt_d = opt_d
        self.opt_n = opt_n
        self.opt_m = opt_m
        self.opt_k = opt_k
        self.opt_k_top = opt_k_top
        self.opt_k_gd = opt_k_gd
        self.opt_lr = opt_lr
        self.opt_r = opt_r
        self.opt_sc = opt_sc

        self.attr_steps = attr_steps
        self.attr_iters = attr_iters
        self.attack_num_max = attack_num_max

        self.set_rand()
        self.set_device(device)
        self.set_path(root)
        self.set_log()

        self.load_data()
        self.load_model()
        self.load_model_attr()

    def end(self):
        self.log.end()

    def get_path(self, fpath):
        fpath = os.path.join(self.path, fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        return fpath

    def load_data(self, log=True):
        if self.data_name is None:
            raise ValueError('Name of the dataset is not set')

        if log:
            tm = self.log.prc(f'Loading "{self.data_name}" dataset')

        self.data = Data(self.data_name)

        if log:
            self.log.res(tpc()-tm)

        if log:
            self.log('')

    def load_model(self, log=True):
        name = self.model_name

        if name is None:
            return

        if log:
            tm = self.log.prc(f'Loading "{name}" model')

        try:
            self.model = Model(name, self.data, self.device)

            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn(f'Can not load Model')

        if log:
            self.log('')

    def load_model_attr(self, log=True):
        name = self.model_attr_name

        if name is None:
            return

        if log:
            tm = self.log.prc(f'Loading "{name}" model')

        try:
            self.model_attr = Model(name, self.data, self.device)

            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn(f'Can not load Model for attribution')

        if log:
            self.log('')

    def run(self):
        eval(f'self.task_{self.task}_{self.kind}()')
        self.end()

    def set_device(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    def set_log(self):
        info = ''
        if self.data_name:
            info += f'Data                : "{self.data_name}"\n'
        if self.model_name:
            info += f'Model               : "{self.model_name}"\n'
        if self.model_attr_name:
            info += f'Model attr.         : "{self.model_attr_name}"\n'
        if self.task:
            info += f'Task                : "{self.task}"\n'
        if self.kind:
            info += f'Kind of task        : "{self.kind}"\n'

        if self.task == 'attack' and self.kind == 'attr':
            if self.opt_d:
                info += f'Opt. dimension      : "{self.opt_d}"\n'
            if self.opt_n:
                info += f'Opt. mode size      : "{self.opt_n}"\n'
            if self.opt_m:
                info += f'Opt. budget         : "{self.opt_m}"\n'

        self.log = Log(self.get_path(f'log.txt'))
        self.log.title(f'Computations ({self.device})', info)

    def set_path(self, root='result'):
        fbase = f'{self.data_name}'
        if self.model_name:
            fbase += f'-{self.model_name}'
        if self.model_attr_name:
            fbase += f'-{self.model_attr_name}'

        ftask = f'{self.task}-{self.kind}'

        self.path = os.path.join(root, fbase, ftask)

        self.get_path(root)

    def set_rand(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def task_attack_attr(self):
        self.attack_num_full = 0
        self.attack_num_succ = 0

        for i in range(len(self.data.data_tst)):
            if not self._set_image(i):
                continue

            self.x_attr = self.model_attr.attrib(self.x_real, self.c_real,
                self.attr_steps, self.attr_iters)
            self.pix = sort_matrix(self.x_attr)[:self.opt_d]

            tm = self.log.prc(f'Attack with PROTES method')

            self.is_succ = self._attack()
            self.attack_num_succ += 1 if self.is_succ else 0

            chng = np.sum(np.abs(self.z_chng) > 0)

            text = f'Image  # {i+1:-7d}  | '
            text += f'in [{self.x_min:-8.1e}, {self.x_max:-8.1e}] '
            text += f'(avg: {self.x_avg:-8.1e})\n'
            text += f'Result : {"SUCC" if self.is_succ else "FAIL"}     '
            text += f'| score {self.e_result:-10.3e} '
            text += f'| changed {chng:-4d}\n'
            text += 'ORIGIN : '
            text += f'{self.y_pred:-8.2e} / {self.y_targ:-8.2e}\n'
            text += 'ATTACK : '
            text += f'{self.y_pred_chng:-8.2e} / {self.y_base_chng:-8.2e}'

            self.log(text)
            self.log.res(tpc()-tm)

            self._plot()
            self._plot_full()

            self.attack_num_full += 1
            if self.attack_num_max:
                if self.attack_num_full >= self.attack_num_max:
                    break

        text = 'Completed. '
        succ = self.attack_num_succ / self.attack_num_full * 100
        text += f'Successful: {succ:-5.2f}% '
        text += f'(total images {self.attack_num_full})'
        self.log('\n\n' + text)

    def task_attack_bs1(self):
        # model = models.resnet18(pretrained=True).eval()
        preprocessing = dict(
            mean=self.data.norm_m, std=self.data.norm_v, axis=-3)
        fmodel = fb.PyTorchModel(self.model.net, bounds=(0, 1),
            preprocessing=preprocessing)

        for i in range(len(self.data.data_tst)):
            x, c, l = self.data.get(i, tst=True)

            x = self.data.tr_norm_inv(x)
            x = torch.unsqueeze(x, dim=0)

            attack = fb.attacks.BoundaryAttack() #LinfPGD()

            raw_advs, x_adv, success = attack(fmodel,
                x, torch.tensor([c]), epsilons=0.01)

            x_adv = self.data.tr_norm(x_adv[0])

            y = self.model.run(x_adv).detach().to('cpu').numpy()
            vals = sort_vector(y)
            c_pred, y_pred = vals[0]

            fpath = self.get_path(f'img/fb_{i}.png')
            self.data.plot(x_adv, f'Real {c} | Pred {c_pred}', fpath)

    def task_check_data(self):
        name = self.data.name
        tm = self.log.prc(f'Check data for "{name}" dataset')
        self.log(self.data.info())
        self.data.plot_many(fpath=self.get_path(f'img/{name}.png'))
        self.log.res(tpc()-tm)

    def task_check_model(self, trn=True, tst=True):
        tm = self.log.prc(f'Check model accuracy')

        for mod in ['trn', 'tst']:
            if mod == 'trn' and not trn or mod == 'tst' and not tst:
                continue
            if mod == 'trn' and self.data.data_trn is None:
                continue
            if mod == 'tst' and self.data.data_tst is None:
                continue

            t = tpc()
            n, m = self.model.check(tst=(mod == 'tst'),
                only_one_batch=(str(self.device)=='cpu'))
            t = tpc() - t

            text = f'Accuracy   {mod}'
            text += f' : {float(m)/n*100:.2f}% ({m:-9d} / {n:-9d})'
            text += f' | time = {t:-10.2f} sec'
            self.log(text)

            self.log('')

        self.log.res(tpc()-tm)

        tm = self.log.prc(f'Plot several model predictions')

        X, titles = [], []
        for i in range(16):
            x, c_real, l_real = self.data.get(i, tst=True)
            y, c_pred, l_pred = self.model.run_pred(x)
            X.append(x)
            _l_real = l_real[:17] + '...' if len(l_real) > 20 else l_real
            _l_pred = l_pred[:17] + '...' if len(l_pred) > 20 else l_pred
            if c_real == c_pred:
                titles.append(f'{_l_real}\n')
            else:
                titles.append(f'{_l_pred}\n(real: {_l_real})')

        self.data.plot_many(X, titles, cols=4, rows=4, size=3,
            fpath=self.get_path(f'img/{self.model.name}.png'))

        self.log.res(tpc()-tm)

    def _attack(self):
        self.h = float((self.x_max - self.x_min) / self.opt_sc)

        def change(I):
            s = I.shape[0]
            dx = (I - (self.opt_n-1)/2) * self.h
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
            dx = (i - (self.opt_n-1)/2) * self.h
            for j, p in enumerate(self.pix):
                if dx[j] > 0.: z[p[0], p[1]] = +1
                if dx[j] < 0.: z[p[0], p[1]] = -1
            return z

        def func(I):
            y = self.model.run(change(I)).detach().to('cpu').numpy()
            return y[:, self.c_targ] - y[:, self.c_real]

        i, e, m_hist, y_hist = opt(func, self.opt_d, self.opt_n, self.opt_m,
            self.opt_k, self.opt_k_top, self.opt_k_gd, self.opt_lr, self.opt_r)
        self.e_result = e

        self.z_chng = get_changes(i)
        self.x_chng = change_one(i)

        res_pred, res_base = self._get_pred(self.x_chng, self.c_pred)
        self.y_pred_chng, self.c_pred_chng, self.l_pred_chng = res_pred
        self.y_base_chng, self.c_base_chng, self.l_base_chng = res_base

        return self.c_pred != self.c_pred_chng

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
        x = torch.tensor(self.x_chng)
        x = self.data.tr_norm_inv(x)
        fold = 'img_succ' if self.is_succ else 'img_fail'
        fpath = f'{fold}/attack_with-{self.model_attr_name}_{self.i}.png'
        self.data.plot_base(x, '', size=6, fpath=self.get_path(fpath))

    def _plot_full(self):
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

        fold = 'img_full_succ' if self.is_succ else 'img_full_fail'
        fpath = f'{fold}/attack_with-{self.model_attr_name}_{self.i}.png'
        plt.savefig(self.get_path(fpath), bbox_inches='tight')
        plt.close(fig)

    def _set_image(self, i):
        self.i = i
        self.ch = self.data.ch
        self.sz = self.data.sz
        self.d = self.ch * self.sz * self.sz
        self.x_real, self.c_real, self.l_real = self.data.get(i, tst=True)
        self.x_real_np = self.x_real.detach().to('cpu').numpy()
        self.x_avg = np.mean(self.x_real_np)
        self.x_max = np.max(self.x_real_np)
        self.x_min = np.min(self.x_real_np)

        res_pred, res_targ = self._get_pred(self.x_real)
        self.y_pred, self.c_pred, self.l_pred = res_pred
        self.y_targ, self.c_targ, self.l_targ = res_targ

        return self.c_real == self.c_pred


def args_build():
    parser = argparse.ArgumentParser(
        prog='tetradat',
        description='Software product for generation of adversarial examples for artificial neural networks using tensor train (TT) decomposition and optimizer based on it, i.e., PROTES optimizer.',
        epilog = '© Andrei Chertkov'
    )
    parser.add_argument('-d', '--data',
        type=str,
        help='Name of the used dataset',
        default=None,
        choices=DATA_NAMES
    )
    parser.add_argument('-m', '--model',
        type=str,
        help='Name of the used model',
        default=None,
        choices=MODEL_NAMES
    )
    parser.add_argument('-a', '--model_attr',
        type=str,
        help='Name of the used model for attribution',
        default=None,
        choices=MODEL_NAMES
    )
    parser.add_argument('-t', '--task',
        type=str,
        help='Name of the task',
        default='attack',
        choices=['check', 'attack']
    )
    parser.add_argument('-k', '--kind',
        type=str,
        help='Kind of the task',
        default='attr',
        choices=['data', 'model', 'attr', 'bs1']
    )
    parser.add_argument('--opt_d',
        type=int,
        help='Dimension for optimization',
        default=500,
    )
    parser.add_argument('--opt_n',
        type=int,
        help='Mode size for optimization',
        default=3,
    )
    parser.add_argument('--opt_m',
        type=int,
        help='Budget for optimization',
        default=10000,
    )
    parser.add_argument('--opt_k',
        type=int,
        help='Batch size for optimization',
        default=50,
    )
    parser.add_argument('--opt_k_top',
        type=int,
        help='Number of selected candidates in the batch',
        default=5,
    )
    parser.add_argument('--opt_k_gd',
        type=int,
        help='Number of gradient lifting iterations',
        default=10,
    )
    parser.add_argument('--opt_lr',
        type=float,
        help='Learning rate for gradient lifting iterations',
        default=1.E-2,
    )
    parser.add_argument('--opt_r',
        type=int,
        help='TT-rank of the constructed probability TT-tensor',
        default=5,
    )
    parser.add_argument('--opt_sc',
        type=float,
        help='Scale for the noize image',
        default=7,
    )
    parser.add_argument('--attr_steps',
        type=int,
        help='Number of attribution steps',
        default=10,
    )
    parser.add_argument('--attr_iters',
        type=int,
        help='Number of attribution iterations',
        default=10,
    )
    parser.add_argument('--attack_num_max',
        type=int,
        help='Maximum number of attacks (if 0, then use full dataset)',
        default=0,
    )
    parser.add_argument('--root',
        type=str,
        help='Path to the folder with results',
        default='result'
    )

    args = parser.parse_args()
    return (args.data, args.model, args.model_attr, args.task, args.kind,
        args.opt_d, args.opt_n, args.opt_m, args.opt_k, args.opt_k_top,
        args.opt_k_gd, args.opt_lr, args.opt_r, args.opt_sc, args.attr_steps,
        args.attr_iters, args.attack_num_max, args.root)


if __name__ == '__main__':
    Manager(*args_build()).run()
