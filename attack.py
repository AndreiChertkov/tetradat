import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


from copy import deepcopy as copy
import numpy as np
from protes_own import protes
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

        self.trans = torchvision.transforms.Normalize(
            self.norm_m, self.norm_v)
        self.trans_base = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                [0., 0., 0.], 1./np.array(self.norm_v)),
            torchvision.transforms.Normalize(
                -np.array(self.norm_m), [1., 1., 1.])])

        self.d = int(self.x.shape[1] * self.x.shape[2])

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
            self.y2 = self.y_all[np.argsort(self.y_all)[::-1][0]]
        else:
            self.success = self.c_new != self.c
            self.y2 = self.y_all[np.argsort(self.y_all)[::-1][1]]

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


class AttackAttrMulti:
    def __init__(self, *args):
        self.args = args
        self.m_max = int(args[3])
        self.m = 0
        self.t = 0.

    def prep(self, net=None, d=None, attr_steps=10, attr_iters=10, thr=1.E-5):
        att = AttackAttr(*self.args)
        att.prep(net, d, attr_steps, attr_iters, thr)
        self.x_attr = att.x_attr
        self.pixels = att.pixels
        self.d = att.d
        self.t = att.t

    def run(self, n, sc, k, k_top, k_gd, lr, r, label=None, sc_delt=0.2):
        t = tpc()

        self.P = None
        self.sc = sc + sc_delt
        result_best = None

        while self.sc > sc_delt * 1.5:
            self.sc -= sc_delt

            print(f'\n AttackMulti start (sc = {self.sc:-7.1e})\n')

            att = AttackAttr(*self.args)
            att.d = self.d
            att.m_max = self.m_max - self.m
            att.x_attr = self.x_attr
            att.pixels = self.pixels

            result = att.run(n, self.sc, k, k_top, k_gd, lr, r, label, P=self.P)
            self.P = att.P
            self.m += result['m']

            if result_best is None or result['success']:
                result_best = copy(result)
                result_best['sc'] = self.sc
                self.x_new = copy(att.x_new)
                self.success = att.success

            text = f'\n\n AttackMulti end (m_total = {self.m:-7.1e})'
            text += ' | ! success' if result['success'] else ' | - fail'
            text += '\n\n\n'
            print(text)

            if self.m + k > self.m_max:
                break

        self.t += tpc() - t

        result_best['m'] = self.m
        result_best['t'] = self.t

        return result_best


class AttackAttr(Attack):
    def attrib(self, net, x, c, steps=10, iters=10):
        x = self.trans_base(x)
        x = np.uint8(np.moveaxis(x.numpy(), 0, 2) * 256)

        def _img_to_x(x):
            # m = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
            # s = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
            # x = (x / 255 - m) / s
            x = np.transpose(x, (2, 0, 1)) / 255
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            x = self.trans(x).unsqueeze(0)
            #x = np.expand_dims(x, 0)
            #x = np.array(x)
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            return x

        def _iter(x):
            x = _img_to_x(x)
            x.requires_grad_()
            y = self.probs(net(x))[0, c]
            net.zero_grad()
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

    def change(self, i):
        h, s, v = torch.clone(self.x_base_hsv)

        dn = (self.n-1) / 2
        delta = (np.array(i) - (self.n-1)/2) * self.sc / dn
        delta = torch.tensor(delta).to(self.device)

        ind = delta < 0
        pix = self.pixels[ind]
        s[pix[:, 0], pix[:, 1]] += delta[ind] * s[pix[:, 0], pix[:, 1]]

        ind = delta > 0
        pix = self.pixels[ind]
        v[pix[:, 0], pix[:, 1]] += delta[ind] * (1. - v[pix[:, 0], pix[:, 1]])

        x_base = color_hsv_to_rgb(torch.stack((h, s, v)))
        return self.trans(x_base)

    def loss(self, I):
        result = []
        for i in I:
            self.m += 1
            self.check(self.change(i))
            if self.success:
                return
            result.append(self.y - self.y2)

        return np.array(result)

    def loss_label(self, I):
        """This is the loss function for the label-based attacks.

        Note that we do not use network's scores here. We just collect
        top-labels from the network's prediction.

        """
        result = np.zeros(len(I), dtype=int)

        for num, i in enumerate(I):
            # Run the neural network for the current permuted image:
            self.m += 1
            self.check(self.change(i))
            if self.success:
                return

            # Collect "label_num" top-labels from prediction:
            label_top = np.argsort(self.y_all)[::-1][:self.label_num]

            # Count the number of new labels in the top-set:
            for k in range(1, self.label_num):
                if not label_top[k] in self.label_top:
                    result[num] += 1

        # We select only samples with the highest number of new labels
        # (but if now such samples are in batch, then we do not train):
        num = np.max(result)
        if num == 0:
            return []
        else:
            return np.where(result == num)[0]

    def prep(self, net=None, d=None, attr_steps=10, attr_iters=10, thr=1.E-5):
        t = tpc()

        if net is None:
            if d is not None:
                raise NotImplementedError
            n = int(np.sqrt(self.d))
            I = []
            for i in range(n):
                for j in range(n):
                    I.append([i, j])
            self.pixels = torch.tensor(I).to(self.device)
            return

        self.x_attr = self.attrib(net, self.x, self.c,
            attr_steps, attr_iters)

        sh = self.x_attr.shape
        sz = self.x_attr.size

        I = np.unravel_index(np.argsort(self.x_attr, axis=None), sh)
        I = [(I[0][k], I[1][k]) for k in range(sz)]
        I = I[::-1]

        if d is not None:
            self.d = d
            I = I[:self.d]
        else:
            raise NotImplementedError

        self.pixels = torch.tensor(I).to(self.device)

        self.t += tpc() - t

    def run(self, n, sc, k, k_top, k_gd, lr, r, label=None, P=None):
        t = tpc()

        self.n = n
        self.sc = sc

        self.x = self.x.to(self.device)
        self.x_base = self.trans_base(self.x)
        self.x_base_hsv = color_rgb_to_hsv(self.x_base)

        if not label:
            loss = self.loss
        else:
            if label < 3:
                raise NotImplementedError
            loss = self.loss_label
            self.check(self.x)
            self.label_num = label
            self.label_top = np.argsort(self.y_all)[::-1][:self.label_num]

        is_max = True if self.target or label else False
        info = {}
        protes(loss, self.d, self.n, self.m_max, k, k_top, k_gd, lr, r,
            info=info, P=P, is_max=is_max, is_func_ind=bool(label), log=True)
        self.P = info['P']

        self.t += tpc() - t
        return self.result()


class AttackBs(Attack):
    def run(self, onepixel=100, pixle=100, square=4/255, seed=42):
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
