import jax
# jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from protes import protes
from time import perf_counter as tpc
import torch
import torchattacks


from utils import sort_matrix


class Attack:
    def __init__(self, model, x, c, l, sc=10):
        self.model = model
        self.x = x
        self.c = c
        self.l = l
        self.sc = sc

        self.init()

    @property
    def h(self):
        return float((self.x_max - self.x_min) / self.sc)

    @property
    def x_avg(self):
        return np.mean(self.x_np)

    @property
    def x_max(self):
        return np.max(self.x_np)

    @property
    def x_min(self):
        return np.min(self.x_np)

    @property
    def x_np(self):
        return self.x.detach().to('cpu').numpy()

    def change(self, changes):
        x = self.x_np.copy()
        for p1, p2, dx in changes:
            for ch in range(3):
                x[ch, p1, p2] += dx if isinstance(dx, (int, float)) else dx[ch]
        return x

    def check(self):
        self.y, c, l = self.model.run_pred(self.x)
        return c == self.c

    def check_new(self, x_new=None):
        if x_new is None:
            self.x_new = self.change(self.changes)
        else:
            self.x_new = x_new
        self.x_new = torch.tensor(self.x_new, dtype=torch.float32)

        self.y_old = float(self.model.run(self.x_new)[self.c])
        self.y_new, self.c_new, self.l_new = self.model.run_pred(self.x_new)

        return self.c_new != self.c

    def init(self):
        self.m = 0
        self.t = 0.
        self.y = None
        self.y_old = None
        self.x_new = None
        self.c_new = None
        self.l_new = None
        self.y_new = None
        self.changes = []
        self.success = False
        self.err = None

    def result(self):
        return {
            'm': self.m,
            't': self.t,
            'c': self.c,
            'l': self.l,
            'y': self.y,
            'y_old': self.y_old,
            'c_new': self.c_new,
            'l_new': self.l_new,
            'y_new': self.y_new,
            'changes': self.changes,
            'success': self.success,
            'err': self.err}


class AttackAttr(Attack):
    def __init__(self, model, x, c, l, sc=10, d=1000, n=3, eps_success=1.E-3):
        super().__init__(model, x, c, l, sc)
        self.d = d
        self.n = n
        self.eps_success = eps_success

    def changes_build(self, i):
        i = np.array(i)
        x = (i - (self.n-1)/2) * self.h
        changes = []
        for k in np.where(np.abs(x) > 1.E-16)[0]:
            changes.append([self.pixels[k][0], self.pixels[k][1], x[k]])
        return changes

    def prep(self, model_attr, attr_steps=10, attr_iters=10):
        _t = tpc()
        self.x_attr = model_attr.attrib(self.x, self.c, attr_steps, attr_iters)
        self.pixels = sort_matrix(self.x_attr)[:self.d]
        self.t += tpc() - _t

    def run(self, m=1.E+4, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, log=True):
        _t = tpc()
        try:
            i, y = protes(self.target, self.d, self.n, m, k, k_top, k_gd, lr, r,
                is_max=False, log=log)
        except Exception as e:
            pass
        self.t += tpc() - _t

        if self.success:
            success = self.check_new()
            if not success:
                self.err = 'Success result check is failed'
                self.success = False

    def target(self, I):
        changes_all = [self.changes_build(i) for i in I]

        X_all = [self.change(changes) for changes in changes_all]
        X_all = torch.tensor(X_all, dtype=torch.float32)

        y_all = self.model.run(X_all).detach().to('cpu').numpy()
        self.m += len(I)

        c_all = np.argmax(y_all, axis=1)
        for k, c in enumerate(c_all):
            if c != self.c and y_all[k,c] - y_all[k,self.c] > self.eps_success:
                self.changes = changes_all[k]
                self.success = True
                return

        return y_all[:, self.c]


class AttackBs(Attack):
    def __init__(self, model, x, c, l, sc=10):
        super().__init__(model, x, c, l, sc)

    def prep(self, name, seed=42):
        if name == 'square':
            self.atk = torchattacks.Square(self.model.net, seed=seed, eps=2/255)
        elif name == 'onepixel':
            self.atk = torchattacks.OnePixel(self.model.net, pixels=100)
        elif name == 'pixle':
            self.atk = torchattacks.Pixle(self.model.net)
        else:
            raise NotImplementedError(f'Baseline "{name}" is not supported')

        self.name = name
        self.atk.set_normalization_used(
            mean=self.model.data.norm_m, std=self.model.data.norm_v)

    def run(self, log=True):
        x_ = torch.unsqueeze(self.x, dim=0)
        c_ = torch.tensor([self.c])
        x_new = self.atk(x_, c_)[0]

        self.changes = []
        for i in range(self.x.shape[1]):
            for j in range(self.x.shape[2]):
                change = np.zeros(self.x.shape[0])
                for ch in range(self.x.shape[0]):
                    change[ch] = x_new[ch, i, j] - self.x[ch, i, j]
                if np.max(np.abs(change)) > 1.E-16:
                    self.changes.append([i, j, change])

        self.success = self.check_new(x_new)

        if log:
            text = f'Img {self.c:-5d} | ' + ('OK' if self.success else 'fail')
            print(text)
