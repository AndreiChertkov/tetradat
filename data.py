from PIL import Image
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import torch
from torch.utils.data import DataLoader
import torchvision


from data_opts import DATA_OPTS


DATA_NAMES = list(DATA_OPTS.keys())


class Data:
    def __init__(self, name, batch_trn=256, batch_tst=32,
                 norm_m=None, norm_v=None, root='result'):
        if not name in DATA_NAMES:
            raise ValueError(f'Dataset name "{name}" is not supported')
        self.name = name
        self.opts = DATA_OPTS[self.name]
        self.root = root

        self.batch_trn = batch_trn
        self.batch_tst = batch_tst

        self.norm_m = norm_m
        if self.norm_m is None:
            self.norm_m = self.opts.get('norm_m')

        self.norm_v = norm_v
        if self.norm_v is None:
            self.norm_v = self.opts.get('norm_v')

        self.labels = self.opts.get('labels', {})
        self.sz = self.opts['sz']
        self.ch = self.opts['ch']

        self._set_transform()
        self._load()

    def animate(self, X, titles, size=3, interval=6, fps=0.7, fpath=None):
        if X is None or len(X) == 0 or len(X) != len(titles):
            print('WRN: invalid data for animation')
            return

        def prep(x):
            x = self.tr_norm_inv(x).detach().to('cpu').squeeze().numpy()
            x = np.clip(x, 0, 1) if np.mean(x) < 2 else np.clip(x, 0, 255)
            x = x.transpose(1, 2, 0) if len(x.shape) == 3 else x
            return x

        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111)
        ax.axis('off')

        img = ax.imshow(prep(X[0]))

        def update(k, *args):
            ax.set_title(titles[k], fontsize=9)
            img.set_data(prep(X[k]))
            return (img,)

        anim = FuncAnimation(fig, update, interval=interval,
            frames=len(X), blit=True, repeat=False)

        if fpath:
            anim.save(fpath, writer='pillow', fps=fps)

        plt.close(fig)

    def get(self, i=None, tst=False):
        data = self.data_tst if tst else self.data_trn

        if i is None:
            i = torch.randint(len(data), size=(1,)).item()

        x, c = data[i]
        l = self.labels.get(c)

        return x, c, l

    def img_load(self, fpath, device='cpu', resize=False, norm=True):
        x = torchvision.io.read_image(fpath)
        x = torchvision.transforms.ConvertImageDtype(torch.float32)(x)
        x = x.expand(self.ch, *x.shape[1:]) if x.shape[0] < self.ch else x
        x = self.tr_size(x) if resize else x
        x = self.tr_norm(x) if norm else x
        return x.to(device)

    def img_rand(self, device='cpu', norm=True):
        pix = np.random.rand(self.sz, self.sz, self.ch) * 255
        img = Image.fromarray(pix.astype('uint8')).convert('RGB')
        x = self.tr_tens(x)
        x = self.tr_norm(x) if norm else x
        return x.to(device)

    def info(self):
        text = ''
        text += f'Dataset             : {self.name}\n'
        text += f'Number of classes   : {len(self.labels):-10d}\n'

        if self.data_trn is not None:
            text += f'Size of trn dataset : {len(self.data_trn):-10d}\n'

        if self.data_tst is not None:
            text += f'Size of tst dataset : {len(self.data_tst):-10d}\n'

        return text

    def plot(self, x, title='', fpath=None, is_new=True, do_tr=True):
        x = self.tr_norm_inv(x) if do_tr else x
        size = self.opts['plot_size']
        cmap = self.opts['plot_cmap']
        return self.plot_base(x, title, size, cmap, fpath, is_new)

    def plot_attr(self, x, fpath=None):
        x = torch.tensor(x) if not torch.is_tensor(x) else x
        x = x.detach().to('cpu').squeeze().numpy()
        x = np.clip(x, 0, 1) if np.mean(x) < 2 else np.clip(x, 0, 255)

        fig = plt.figure(figsize=(12, 12))
        plt.imshow(x)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_base(self, x, title, size=3, cmap='hot', fpath=None, is_new=True):
        if torch.is_tensor(x):
            x = x.detach().to('cpu').squeeze().numpy()
        if len(x.shape) == 3:
            x = x.transpose(1, 2, 0)
        x = np.clip(x, 0, 1) if np.mean(x) < 2 else np.clip(x, 0, 255)

        if is_new:
            fig = plt.figure(figsize=(size, size))

        plt.imshow(x, cmap=cmap)
        if title:
            plt.title(title, fontsize=9)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
        elif is_new:
            plt.show()

        if is_new:
            plt.close(fig)

    def plot_changes(self, x, fpath=None):
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(x, cmap=mcolors.ListedColormap(['white', 'black', 'red']))
        plt.axis('off')
        plt.gca().set_facecolor('black')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_many(self, X=None, titles=None, cols=4, rows=4, size=3,
                  fpath=None, wspace=0.3, hspace=0.3):
        fig = plt.figure(figsize=(size*cols, size*rows))
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

        for j in range(1, cols * rows + 1):
            if X is None:
                i = torch.randint(len(self.data_tst), size=(1,)).item()
                x, c, l = self.get(i, tst=True)
                title = l[:17] + '...' if len(l) > 20 else l
            else:
                x = X[j-1].detach().to('cpu')
                title = titles[j-1] if titles else ''

            fig.add_subplot(rows, cols, j)
            self.plot(x, title, is_new=False)

        plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()
        plt.close(fig)

    def _load(self):
        self.data_trn = None
        self.data_tst = None

        self.dataloader_trn = None
        self.dataloader_tst = None

        fpath = os.path.join(self.root, '_data', self.name)
        load = not os.path.isdir(fpath)
        os.makedirs(fpath, exist_ok=True)

        if self.opts.get('dataset'):
            func = eval(f'torchvision.datasets.{self.opts["dataset"]}')

            self.data_trn = func(root=fpath, train=True,
                download=load, transform=self.tr)
            self.data_tst = func(root=fpath, train=False,
                download=load, transform=self.tr)

            self.dataloader_trn = DataLoader(self.data_trn,
                batch_size=self.batch_trn, shuffle=True)
            self.dataloader_tst = DataLoader(self.data_tst,
                batch_size=self.batch_tst, shuffle=True)

        if self.opts.get('repo'):
            # TODO: add support for trn/tst repo
            if load:
                load_repo(self.opts['repo'], fpath)
            repo = self.opts['repo'].split('.git')[0].split('/')[-1]
            fpath = os.path.join(fpath, repo)

            # Below is the code to parse IMAGENET images from the repo:
            # https://github.com/EliSchwartz/imagenet-sample-images.git

            l_rep = {}

            for f in os.listdir(fpath):
                if not f.endswith('JPEG'):
                    continue
                l = ' '.join(f.split('.JPEG')[0].split('_')[1:])
                l = l.lower().replace("'", '`')
                c = None
                is_found = False
                for c_real, l_real in self.labels.items():
                    if l == l_real.split(',')[0]:
                        if is_found:
                            l_rep[l] = 0
                        else:
                            if not l in l_rep or l_rep[l] == 1:
                                c = c_real
                                is_found = True
                            else:
                                l_rep[l] = 1
                if c is None:
                    raise ValueError

                path_old = os.path.join(fpath, f)
                path_new = os.path.join(fpath, f'{c}.jpg')
                os.rename(path_old, path_new)

            class Dataset(torch.utils.data.Dataset):
                def __init__(self, fpath, labels, transform):
                    self.fpath = fpath
                    self.labels = labels
                    self.transform = transform

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, i):
                    img_path = os.path.join(fpath, f'{i}.jpg')
                    x = torchvision.io.read_image(img_path)
                    x = torchvision.transforms.ConvertImageDtype(
                        torch.float32)(x)
                    if x.shape[0] == 1:
                        x = x.expand(3, *x.shape[1:])
                    x = self.transform(x)
                    return x, i

            tr = torchvision.transforms.Compose([self.tr_size, self.tr_norm])
            self.data_tst = Dataset(fpath, self.labels, transform=tr)

            self.dataloader_tst = DataLoader(self.data_tst,
                batch_size=self.batch_tst, shuffle=True)

    def _set_transform(self):
        self.tr_tens = torchvision.transforms.ToTensor()

        self.tr_size = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.sz),
            torchvision.transforms.CenterCrop(self.sz)])

        self.tr_norm = lambda x: x
        self.tr_norm_inv = lambda x: x
        if self.norm_m is not None and self.norm_v is not None:
            self.tr_norm = torchvision.transforms.Normalize(
                self.norm_m, self.norm_v)
            self.tr_norm_inv = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(
                    [0., 0., 0.], 1./np.array(self.norm_v)),
                torchvision.transforms.Normalize(
                    -np.array(self.norm_m), [1., 1., 1.]),
            ])

        # TODO: see https://github.com/pytorch/vision/blob/f69eee6108cd047ac8b62a2992244e9ab3c105e1/torchvision/transforms/_presets.py#L38

        self.tr = torchvision.transforms.Compose([self.tr_tens, self.tr_norm])


def load_repo(url, fpath):
    prc = subprocess.getoutput(f'cd {fpath} && git clone {url}')
    print(prc)
