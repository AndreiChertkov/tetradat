import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from protes import protes
from time import perf_counter as tpc
import torch
import torchattacks


from utils import sort_matrix


class Attack:
    def __init__(self, model, x, c, l, sc=10, target=False):
        self.model = model
        self.x = x
        self.c = c
        self.l = l
        self.sc = sc
        self.is_target = target

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
        return change(self.x_np, changes)

    def check(self):
        y_all = self.model.run(self.x).detach().to('cpu').numpy()

        c = np.argmax(y_all)
        self.y = y_all[c]

        if self.is_target:
            self.c_target = np.argmin(y_all)
            self.y_target = y_all[self.c_target]

        return c == self.c

    def check_new(self, x_new=None):
        if x_new is None:
            self.x_new = self.change(self.changes)
        else:
            self.x_new = x_new
        self.x_new = torch.tensor(self.x_new, dtype=torch.float32)

        self.y_old = float(self.model.run(self.x_new)[self.c])
        self.y_new, self.c_new, self.l_new = self.model.run_pred(self.x_new)

        if self.c_new == self.c:
            return False

        if self.is_target and self.c_new != self.c_target:
            return False

        return True

    def init(self):
        self.m = 0            # Number of model calls
        self.t = 0.           # Computation time
        self.y = None         # Prob of correct class on original image
        self.y_old = None     # Prob of correct class on updated image
        self.x_new = None     # Updated image (after attack)
        self.c_new = None     # New class
        self.l_new = None     # New label related to the new class
        self.y_new = None     # Prob (maximum) on updated image
        self.c_target = None  # Target class for attack (optional)
        self.y_target = None  # Prob on target class (if used)
        self.changes = []     # List of changed pixels
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
            'c_target': self.c_target,
            'y_target': self.y_target,
            'changes': self.changes,
            'success': self.success,
            'err': self.err}


class AttackAttr(Attack):
    def __init__(self, model, x, c, l, sc=10, d=500, n=3, eps_success=1.E-4,
                 target=False):
        super().__init__(model, x, c, l, sc, target)
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

    def prep(self, model_attr, attr_steps=15, attr_iters=15):
        _t = tpc()

        self.x_attr = model_attr.attrib(self.x, self.c,
            attr_steps, attr_iters)

        if not self.is_target:
            self.pixels = sort_matrix(self.x_attr)[:self.d]
            self.t += tpc() - _t
            return

        self.x_attr_target = model_attr.attrib(self.x, self.c_target,
            attr_steps, attr_iters)

        pixels1 = sort_matrix(self.x_attr)[:int(self.d/2)]
        pixels2 = sort_matrix(self.x_attr_target)[:int(self.d/2)]
        self.pixels = pixels1 + pixels2

        self.t += tpc() - _t

    def run(self, m=1.E+4, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, log=True):
        _t = tpc()
        try:
            i, y = protes(self.loss, self.d, self.n, m, k, k_top, k_gd, lr, r,
                is_max=(True if self.is_target else False), log=log)
        except Exception as e:
            pass
        self.t += tpc() - _t

        if self.success:
            success = self.check_new()
            if not success:
                self.err = 'Success result check is failed'
                self.success = False

    def loss(self, I):
        changes_all = [self.changes_build(i) for i in I]

        X_all = [self.change(changes) for changes in changes_all]
        X_all = torch.tensor(X_all, dtype=torch.float32)

        y_all = self.model.run(X_all).detach().to('cpu').numpy()
        self.m += len(I)

        c_all = np.argmax(y_all, axis=1)
        for k, c in enumerate(c_all):
            if self.is_target:
                if c == self.c_target:
                    if y_all[k, c] - y_all[k, self.c] > self.eps_success:
                        self.changes = changes_all[k]
                        self.success = True
                        return
            else:
                if c != self.c:
                    if y_all[k, c] - y_all[k, self.c] > self.eps_success:
                        self.changes = changes_all[k]
                        self.success = True
                        return

        return y_all[:, self.c_target if self.is_target else self.c]


class AttackBs(Attack):
    def __init__(self, model, x, c, l, sc=10, target=False):
        super().__init__(model, x, c, l, sc, target)

    def prep(self, name, seed=42):
        if name == 'onepixel':
            self.atk = torchattacks.OnePixel(self.model.net, pixels=100)
        elif name == 'pixle':
            self.atk = torchattacks.Pixle(self.model.net)
        elif name == 'square':
            self.atk = torchattacks.Square(self.model.net, eps=2/255, seed=seed)
        else:
            raise NotImplementedError(f'Baseline "{name}" is not supported')

        self.name = name
        self.atk.set_normalization_used(
            mean=self.model.data.norm_m, std=self.model.data.norm_v)

        if self.is_target:
            self.atk.set_mode_targeted_least_likely(1)

    def run(self, log=True):
        x_ = torch.unsqueeze(self.x, dim=0)
        c_ = torch.tensor([self.c_target if self.is_target else self.c])
        x_new = self.atk(x_, c_)[0]

        self.changes = []
        for p1 in range(self.x.shape[1]):
            for p2 in range(self.x.shape[2]):
                change = np.zeros(self.x.shape[0])
                for ch in range(self.x.shape[0]):
                    change[ch] = x_new[ch, p1, p2] - self.x[ch, p1, p2]
                if np.max(np.abs(change)) > 1.E-16:
                    self.changes.append([p1, p2, change])

        self.success = self.check_new(x_new)

        if log:
            text = f'Img {self.c:-5d} | ' + ('OK' if self.success else 'fail')
            print(text)


def change(x, changes, to_torch=False):
    x = x.copy()
    for p1, p2, dx in changes:
        for ch in range(3):
            x[ch, p1, p2] += dx if isinstance(dx, (int, float)) else dx[ch]
    if to_torch:
        x = torch.tensor(x, dtype=torch.float32)
    return x
