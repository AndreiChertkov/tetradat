from datetime import datetime
import numpy as np
import requests
import subprocess
from time import perf_counter as tpc
import torch
from urllib.parse import urlencode


class Log:
    def __init__(self, fpath=None):
        self.fpath = fpath
        self.is_new = True

    def __call__(self, text):
        print(text)
        if self.fpath:
            with open(self.fpath, 'w' if self.is_new else 'a') as f:
                f.write(text + '\n')
        self.is_new = False

    def end(self):
        dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        content = f'Work is finished ({tpc()-self.tm:-.2f} sec. total)'
        text = '\n\n' + '=' * 21 + ' ' + '-' * len(content) + '\n'
        text += f'[{dt}] {content}\n\n'
        self(text)

    def prc(self, content=''):
        self(f'\n.... {content}')
        return tpc()

    def res(self, t, content=''):
        self(f'DONE ({t:-9.2f} sec.) {content}')

    def title(self, content, info):
        self.tm = tpc()
        dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        text = f'[{dt}] {content}'
        text += '\n' + '=' * 21 + ' ' + '-' * len(content) + '\n'
        text += info
        text += '=' * (22 + len(content)) + '\n'
        self(text)

    def wrn(self, content=''):
        self(f'WRN ! {content}')


def load_repo(url, fpath):
    prc = subprocess.getoutput(f'cd {fpath} && git clone {url}')
    print(prc)


def sort_vector(a, rev=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=rev)
