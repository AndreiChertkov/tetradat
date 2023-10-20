import argparse
import numpy as np
import os
import random
from time import perf_counter as tpc
import torch


from attack import AttackAttr
from data import DATA_NAMES
from data import Data
from model import MODEL_NAMES
from model import Model
from utils import Log


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
        if os.path.dirname(fpath):
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
        return fpath

    def load_data(self, log=True):
        self.data = None
        name = self.data_name

        if name is None:
            raise ValueError('Name of the dataset is not set')

        if log:
            tm = self.log.prc(f'Loading "{name}" dataset')

        self.data = Data(name)

        if log:
            self.log.res(tpc()-tm)

        if log:
            self.log('')

    def load_model(self, log=True):
        self.model = None
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
        self.model_attr = None
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

        if self.task in ['attack', 'result'] and self.kind == 'attr':
            if self.opt_sc:
                info += f'Opt. scale          : {self.opt_sc}\n'
            if self.opt_d:
                info += f'Opt. dimension      : {self.opt_d}\n'
            if self.opt_n:
                info += f'Opt. mode size      : {self.opt_n}\n'
            if self.opt_m:
                info += f'Opt. budget         : {self.opt_m}\n'
            if self.opt_k:
                info += f'Opt. batch size     : {self.opt_k}\n'
            if self.opt_k_top:
                info += f'Opt. k-top          : {self.opt_k_top}\n'
            if self.opt_k_top:
                info += f'Opt. gd iters       : {self.opt_k_gd}\n'
            if self.opt_lr:
                info += f'Opt. learn. rate    : {self.opt_lr}\n'
            if self.opt_r:
                info += f'Opt. TT-rank        : {self.opt_r}\n'

        fname = 'log_result.txt' if self.task == 'result' else 'log.txt'
        self.log = Log(self.get_path(fname))
        self.log.title(f'Computations ({self.device})', info)

    def set_path(self, root='result'):
        fbase = f'{self.data_name}'
        if self.model_name:
            fbase += f'-{self.model_name}'

        task = 'attack' if self.task == 'result' else self.task
        ftask = f'{task}-{self.kind}'

        if self.model_attr_name:
            ftask += f'-{self.model_attr_name}'

        if self.task in ['attack', 'result'] and self.kind == 'attr':
            ftask += f'-sc{self.opt_sc}'

        self.path = os.path.join(root, fbase, ftask)

        self.get_path(root)

    def set_rand(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def task_attack_attr(self):
        tm = self.log.prc(f'Start attack on images')
        result = {}

        for i in range(len(self.data.data_tst)):
            x, c, l = self.data.get(i, tst=True)

            att = AttackAttr(self.model, x, c, l,
                self.opt_sc, self.opt_d, self.opt_n)
            if not att.check():
                continue

            att.prep(self.model_attr, self.attr_steps, self.attr_iters)
            print(f'\n   ---> Attack # {i:-4d}')
            att.run(self.opt_m, self.opt_k, self.opt_k_top, self.opt_k_gd,
                self.opt_lr, self.opt_r, log=True)
            result[c] = att.result()

            fpath = self.get_path('result.npz')
            np.savez_compressed(self.get_path('result.npz'), result=result)

        succ = np.sum([r['success'] for r in result.values()])
        full = len(result.keys())
        text = 'Completed. '
        text += f'Successful: {succ/full*100:-5.2f}% '
        text += f'(total images {full})'
        self.log('\n' + text)

        self.log.res(tpc()-tm)

    def task_result_attr(self):
        fpath = self.get_path('result.npz')
        result = np.load(fpath, allow_pickle=True).get('result').item()

        for i in range(len(self.data.data_tst)):
            if not i in result:
                continue

            if result[i].get('err'):
                text = f'Image  | i: {i:-4d} | Oops: ' + result[i]['err']
                self.log(text)
                continue

            if not result[i].get('success'):
                continue

            x, c_real, l_real = self.data.get(i, tst=True)
            y, c, l = self.model.run_pred(x)

            x_new = x.detach().to('cpu').numpy().copy()
            for p1, p2, dx in result[i]['changes']:
                for ch in range(3):
                    x_new[ch, p1, p2] += dx
            x_new = torch.tensor(x_new, dtype=torch.float32)
            y_new, c_new, l_new = self.model.run_pred(x_new)

            dx = np.linalg.norm(np.array(x_new) - np.array(x))

            if c != c_real:
                raise ValueError('Invalid result (c != c_real)')
            if result[i]['success'] and c_new != result[i]['c_new']:
                print('oops... ', c, c_new, result[i]['c_new'])
                # raise ValueError('Invalid result (c_new != c_new from res)')

            text = f'Image  | c: {c:-4d} > {c_new:-4d} | '
            text += f'y: {result[i]["y"]:-9.3e} > {result[i]["y_old"]:-9.3e} | '
            text += f'y_new: {result[i]["y_new"]:-9.3e} | '
            text += f'dx: {dx:-8.2e}'
            text += f'\n        Class ini: {l[:40]}'
            text += f'\n        Class new: {l_new[:40]}'
            self.log(text)

            x = self.data.tr_norm_inv(x)
            fpath = f'image_ini/attack_c-{c}.png'
            self.data.plot_base(x, '', size=6, fpath=self.get_path(fpath))

            x_new = self.data.tr_norm_inv(x_new)
            fpath = f'image_new/attack_c-{c}.png'
            self.data.plot_base(x_new, '', size=6, fpath=self.get_path(fpath))

        succ = np.sum([r['success'] for r in result.values()])
        full = len(result.keys())
        text = f'Successful: {succ/full*100:-5.2f}% '
        text += f'(total images {full})'
        self.log('\n' + text)

    def task_attack_bs1(self):
        raise NotImplementedError

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
        choices=['check', 'attack', 'result']
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
        default=1000,
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
        default=100,
    )
    parser.add_argument('--opt_k_top',
        type=int,
        help='Number of selected candidates in the batch',
        default=10,
    )
    parser.add_argument('--opt_k_gd',
        type=int,
        help='Number of gradient lifting iterations',
        default=100,
    )
    parser.add_argument('--opt_lr',
        type=float,
        help='Learning rate for gradient lifting iterations',
        default=1.E-3,
    )
    parser.add_argument('--opt_r',
        type=int,
        help='TT-rank of the constructed probability TT-tensor',
        default=5,
    )
    parser.add_argument('--opt_sc',
        type=int,
        help='Scale for the noize image',
        default=10,
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
