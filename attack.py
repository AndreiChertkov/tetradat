import jax
#jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from protes import protes
from time import perf_counter as tpc
import torch
import torchattacks
import torchvision


class Attack:
    def __init__(self, net, x, c, m_max, name, norm_m, norm_v, target=False):
        self.net = net
        self.x = x
        self.c = c
        self.m_max = int(m_max)
        self.name = name
        self.norm_m = norm_m
        self.norm_v = norm_v
        self.target = target

        self.m = 0               # Number of model calls
        self.t = 0.              # Computation time

        self.x_new = None        # Updated image (after attack)
        self.c_new = None        # New class for updated image
        self.y_new = None        # Prob (maximum) on updated image
        self.y = None            # Prob for attacked class on updated image

        self.success = False     # Result of the attack
        self.changes = 0         # Number of changed pixels
        self.dx1 = 0.            # L1 norm for changes
        self.dx2 = 0.            # L2 norm for changes

        self.device = next(self.net.parameters()).device
        self.probs = torch.nn.Softmax(dim=1)

        self.err = None

    def check(self, x_new):
        x = x_new[None].to(self.device)
        with torch.no_grad():
            self.y_all = self.net(x)
            self.y_all = self.probs(self.y_all)
        self.y_all = self.y_all[0].detach().to('cpu').numpy()

        self.x_new = x_new
        self.c_new = np.argmax(self.y_all)
        self.y_new = self.y_all[self.c_new]
        self.y = self.y_all[self.c]

        if self.target:
            self.success = self.c_new == self.c
        else:
            self.success = self.c_new != self.c

        self.changes = torch.sum((self.x_new - self.x)**2, axis=0)
        self.changes = torch.sum(self.changes > 1.E-6).item()
        self.dx1 = torch.norm(self.x_new - self.x, p=1).item()
        self.dx2 = torch.norm(self.x_new - self.x, p=2).item()

    def result(self):
        return {
            'm': self.m,
            't': self.t,
            'c': self.c,
            'c_new': self.c_new,
            'y_new': self.y_new,
            'y': self.y,
            'success': self.success,
            'changes': self.changes,
            'dx1': self.dx1,
            'dx2': self.dx2,
            'err': self.err}


class AttackAttr(Attack):
    def change(self, i):
        h, s, v = torch.clone(self.x_base_hsv)

        delta = (np.array(i) - (self.n-1)/2) * self.sc
        delta = torch.tensor(delta).to(self.device)

        s[self.pixels[:, 0], self.pixels[:, 1]] += delta
        s[s > 1.] = 1.
        s[s < 0.] = 0.

        #v[self.pixels[:, 0], self.pixels[:, 1]] += delta
        #v[v > 1.] = 1.
        #v[v < 0.] = 0.

        #h_target = h[self.pixels[:, 0], self.pixels[:, 1]]
        #idx = h_target + delta > 1.
        #delta[idx] = delta[idx] - 1.
        #idx = h_target + delta < 0.
        #delta[idx] = 1. + delta[idx]
        #h[self.pixels[:, 0], self.pixels[:, 1]] += delta

        x_base = color_hsv_to_rgb(torch.stack((h, s, v)))
        return self.trans(x_base)

    def run(self, x_attr, d, n, sc, k, k_top, k_gd, lr, r, label=None):
        t = tpc()

        self.d = d
        self.n = n
        self.sc = sc

        self.trans = torchvision.transforms.Normalize(
            self.norm_m, self.norm_v)
        self.trans_base = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                [0., 0., 0.], 1./np.array(self.norm_v)),
            torchvision.transforms.Normalize(
                -np.array(self.norm_m), [1., 1., 1.])])

        I = np.unravel_index(np.argsort(x_attr, axis=None), x_attr.shape)
        I = [(I[0][k], I[1][k]) for k in range(x_attr.size)]
        self.pixels = torch.tensor(I[::-1][:self.d]).to(self.device)

        self.x = self.x.to(self.device)
        self.x_base = self.trans_base(self.x)
        self.x_base_hsv = color_rgb_to_hsv(self.x_base)

        if not label:
            loss = self.loss
        else:
            loss = self.loss_label
            self.check(self.x)
            self.label_num = label
            self.label_top = np.argsort(self.y_all)[::-1][:self.label_num]
            if not self.target:
                raise NotImplementedError
            if self.c != self.label_top[1]:
                raise NotImplementedError
            #print('Labels start', self.label_top)

        try:
            i, y = protes(loss, d, n, self.m_max, k, k_top, k_gd, lr, r,
                is_max=(True if self.target else False), log=True)
        except Exception as e:
            pass

        self.t += tpc() - t
        return self.result()

    def loss(self, I):
        result = []
        for i in I:
            self.m += 1
            self.check(self.change(i))
            if self.success:
                return
            result.append(self.y)
        return np.array(result)

    def loss_label(self, I, rew=10.):
        result = []
        for i in I:
            self.m += 1
            self.check(self.change(i))
            if self.success:
                return
            label_top = np.argsort(self.y_all)[::-1][:self.label_num]
            reward = 0.
            if self.c != label_top[1]:
                reward -= 5 * rew * (self.label_num-2)
            for k in range(self.label_num):
                if not label_top[k] in self.label_top:
                    reward += rew
            result.append(reward)
            #print('Labels curre', label_top, reward)
        return np.array(result)


class AttackBs(Attack):
    def run(self, onepixel=100, pixle=100, square=5/255, seed=42):
        t = tpc()
        self._build(onepixel, pixle, square, seed)

        x_ = torch.unsqueeze(self.x, dim=0).to('cpu')
        c_ = torch.tensor([self.c]).to('cpu')

        x_new = self.atk(x_, c_)[0].detach().to('cpu')

        self.check(x_new)
        self.m = self.atk.model_evals

        self.t += tpc() - t
        return self.result()

    def _build(self, onepixel, pixle, square, seed):
        if self.name == 'onepixel':
            self.atk = _OnePixel(self.net,
                pixels=onepixel,
                steps=19) # TODO: check (now it for 1E+4 evals)

        elif self.name == 'pixle':
            restarts = pixle
            max_iterations = int(self.m_max / 2 / restarts)
            self.atk = _Pixle(self.net,
                restarts=restarts,
                max_iterations=max_iterations)

        elif self.name == 'square':
            self.atk = _Square(self.net,
                eps=square,
                n_queries=self.m_max,
                seed=seed)

        else:
            raise NotImplementedError(f'Baseline "{self.name}" not supported')

        self.atk.set_normalization_used(mean=self.norm_m, std=self.norm_v)

        if self.target:
            self.atk.set_mode_targeted_by_label(quiet=True)


class _OnePixel(torchattacks.OnePixel):
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if not hasattr(self, 'model_evals'):
            self.model_evals = 0
        self.model_evals += inputs.shape[0]
        return super().get_logits(inputs, labels, *args, **kwargs)


class _Pixle(torchattacks.Pixle):
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if not hasattr(self, 'model_evals'):
            self.model_evals = 0
        self.model_evals += inputs.shape[0]
        return super().get_logits(inputs, labels, *args, **kwargs)


class _Square(torchattacks.Square):
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if not hasattr(self, 'model_evals'):
            self.model_evals = 0
        self.model_evals += inputs.shape[0]
        return super().get_logits(inputs, labels, *args, **kwargs)


def color_hsv_to_rgb(hsv):
    is_batch = len(hsv.shape) == 4
    if not is_batch:
        hsv = hsv[None]

    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m

    return rgb if is_batch else rgb[0]


def color_rgb_to_hsv(rgb):
    is_batch = len(rgb.shape) == 4
    if not is_batch:
        rgb = rgb[None]

    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    hsv = torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    return hsv if is_batch else hsv[0]
