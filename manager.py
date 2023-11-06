import argparse
import numpy as np
import os
import random
from time import perf_counter as tpc
import torch


from attack import AttackAttr
from attack import AttackBs
from data import DATA_NAMES
from data import Data
from model import MODEL_NAMES
from model import Model
from utils import Log


RESULT_SHOW = [0, 1, 2, 3, 4, 10, 15, 44, 56, 74, 88, 254, 300, 500, 583, 999]


class Manager:
    def __init__(self, data, model, model_attr, task, kind, opt_d, opt_n, opt_m,
                 opt_k, opt_k_top, opt_k_gd, opt_lr, opt_r, opt_sc, attr_steps,
                 attr_iters, attack_num_target, attack_num_max, root='result',
                 postfix='', device=None):
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

        self.attack_num_target = attack_num_target
        self.attack_num_max = attack_num_max

        self.set_rand()
        self.set_device(device)
        self.set_path(root, postfix)
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

        if self.task in ['attack', 'attack_target'] and self.kind == 'attr':
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
        if self.task == 'attack_target' and self.kind == 'attr':
            if self.attack_num_target is not None:
                info += f'Target class (delt) : {self.attack_num_target}\n'

        self.log = Log(self.get_path('log.txt'))
        self.log.title(f'Computations ({self.device})', info)

    def set_path(self, root='result', postfix=''):
        fbase = f'{self.data_name}'
        if self.model_name:
            fbase += f'-{self.model_name}'

        ftask = f'{self.task}-{self.kind}'

        if self.model_attr_name:
            ftask += f'-{self.model_attr_name}'

        if postfix:
            ftask += f'-{postfix}'

        self.path = os.path.join(root, fbase, ftask)

        self.get_path(root)

    def set_rand(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def task_attack_attr(self):
        self._attacks()

    def task_attack_bs_onepixel(self):
        self._attacks('onepixel')

    def task_attack_bs_pixle(self):
        self._attacks('pixle')

    def task_attack_bs_square(self):
        self._attacks('square')

    def task_attack_target_attr(self):
        self._attacks(target=True)

    def task_attack_target_bs_onepixel(self):
        self._attacks('onepixel', target=True)

    def task_attack_target_bs_pixle(self):
        self._attacks('pixle', target=True)

    def task_attack_target_bs_square(self):
        self._attacks('square', target=True)

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

    def _attack(self, i, name=None, target=False, show=False):
        x, c, l = self.data.get(i, tst=True)

        y_all = self.model.run(x).detach().to('cpu').numpy()
        y = y_all[c]

        if np.argmax(y_all) != c:
            # Invalid prediction for target image; skip
            return

        if target:
            c_attack = np.argsort(y_all)[::-1][self.attack_num_target]
            y_attack = y_all[c_attack]
        else:
            c_attack = c
            y_attack = y_all[np.argsort(y_all)[::-1][1]]

        text = f'\n--> # {i:-4d} | '
        text += f'c     {c:-4d} | '
        if target:
            text += f'c_att    {c_attack:-4d} | '
        text += f'y     {y:-7.1e} | '
        text += f'y_{"att" if target else "next"} {y_attack:-7.1e}'
        self.log(text)

        Att = AttackBs if name else AttackAttr
        att = Att(self.model.net, x, c_attack, self.opt_m, name or 'tetradat',
            self.data.norm_m, self.data.norm_v, target)

        if name:
            result = att.run()
        else:
            print('')
            x_attr = self.model_attr.attrib(x, c_attack,
                self.attr_steps, self.attr_iters)
            result = att.run(x_attr, self.opt_d, self.opt_n, self.opt_sc,
                self.opt_k, self.opt_k_top, self.opt_k_gd, self.opt_lr,
                self.opt_r)
            print('')

        result['l'] = l
        result['l_new'] = self.data.labels[result['c_new']]

        if att.success:
            y_all_new = self.model.run(att.x_new).detach().to('cpu').numpy()
            y_old = y_all_new[c]
            text = f'+++ >        c_new {result["c_new"]:-4d} | '
            text += f'y_new {result["y_new"]:-7.1e} | '
            text += f'y_old {y_old:-7.1e} | '
            text += f'evals {result["m"]:-5d}\n'
            text += f'    : changes {result["changes"]:-5d} | '
            text += f'dx1 {result["dx1"]:-7.1e} | '
            text += f'dx2 {result["dx2"]:-7.1e}\n'
            text += f'    : l_old : {result["l"][:50]}\n'
            text += f'    : l_new : {result["l_new"][:50]}\n'
        else:
            text = f'    > fail | '
            text += f'evals {result["m"]:-5d}'
        self.log(text)

        if not att.success or not show:
            return result

        self.data.plot_base(self.data.tr_norm_inv(x), '', size=6,
            fpath=self.get_path(f'img/{c}/base.png'))
        self.data.plot_base(self.data.tr_norm_inv(att.x_new), '', size=6,
            fpath=self.get_path(f'img/{c}/changed.png'))

        if name is not None:
            return result

        self.data.plot_attr(x_attr,
            fpath=self.get_path(f'img/{c}/attr.png'))

        x_attr_new = self.model_attr.attrib(att.x_new.detach().to('cpu'),
            c, self.attr_steps, self.attr_iters)
        self.data.plot_attr(x_attr_new,
            fpath=self.get_path( f'img/{c}/attr_new.png'))

        if target:
            x_attr_target = self.model_attr.attrib(x.detach().to('cpu'),
                c_attack, self.attr_steps, self.attr_iters)
            self.data.plot_attr(x_attr_target,
                fpath=self.get_path(f'img/{c}/attr_target.png'))

        return result

    def _attacks(self, name=None, target=False):
        if target:
            title = 'Start targeted attack on images'
        else:
            title = 'Start attack on images'
        if name:
            title += f' with baseline "{name}"'
        tm = self.log.prc(title)

        result = {}
        for i in range(len(self.data.data_tst)):
            if self.attack_num_max and len(result.keys())>=self.attack_num_max:
                break
            res = self._attack(i, name, target, show=(i in RESULT_SHOW))
            if res is not None:
                result[i] = res

        fpath = self.get_path('result.npz')
        np.savez_compressed(self.get_path('result.npz'), result=result)

        succ = np.sum([r.get('success', False) for r in result.values()])
        full = np.sum([True for r in result.values()])

        text = 'Completed. '
        text += f'Successful: {succ/full*100:-5.2f}% '
        text += f'(total images {full})'
        self.log('\n' + text)

        self.log.res(tpc()-tm)


def args_build():
    parser = argparse.ArgumentParser(
        prog='tetradat',
        description='Library for generation of adversarial examples for artificial neural networks using tensor train (TT) decomposition and optimizer based on it, i.e., PROTES optimizer.',
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
        default='attack_target',
        choices=['check', 'attack', 'attack_target']
    )
    parser.add_argument('-k', '--kind',
        type=str,
        help='Kind of the task',
        default='attr',
        choices=['data', 'model', 'attr',
            'bs_onepixel', 'bs_pixle', 'bs_square']
    )
    parser.add_argument('--opt_d',
        type=int,
        help='Dimension for optimization',
        default= 1000, # TODO
    )
    parser.add_argument('--opt_n',
        type=int,
        help='Mode size for optimization',
        default=5, # TODO
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
        default=10, # TODO
    )
    parser.add_argument('--opt_lr',
        type=float,
        help='Learning rate for gradient lifting iterations',
        default=1.E-2, # TODO
    )
    parser.add_argument('--opt_r',
        type=int,
        help='TT-rank of the constructed probability TT-tensor',
        default=3, # TODO
    )
    parser.add_argument('--opt_sc',
        type=int,
        help='Scale for the noize image',
        default=15,
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
    parser.add_argument('--attack_num_target',
        type=int,
        help='Target top class number for targeted attack (>= 1)',
        default=5,
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
    parser.add_argument('--postfix',
        type=str,
        help='Postfix for the folder with results',
        default=''
    )

    args = parser.parse_args()
    return (args.data, args.model, args.model_attr, args.task, args.kind,
        args.opt_d, args.opt_n, args.opt_m, args.opt_k, args.opt_k_top,
        args.opt_k_gd, args.opt_lr, args.opt_r, args.opt_sc, args.attr_steps,
        args.attr_iters, args.attack_num_target, args.attack_num_max, args.root,
        args.postfix)


if __name__ == '__main__':
    Manager(*args_build()).run()
