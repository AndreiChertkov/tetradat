import numpy as np


ROOT = 'result'
DATASET = 'imagenet'
MODEL = 'alexnet'
MODEL_ATTR = 'vgg'
D_LIST = [200, 1000, 5000, 25000, 0]


def load(model, d, bs=None):
    if d == 5000: # Base run
        fpath = f'{ROOT}/{DATASET}-{model}/attack-'
        fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
        fpath += f'/result.npz'
        res = np.load(fpath, allow_pickle=True).get('result').item()
        return res

    result = {}
    for i in range(1, 11):
        fpath = f'{ROOT}/{DATASET}-{model}/attack-'
        if d == 0: # Run without attr
            fpath += 'base'
        else:
            fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
        fpath += f'-d{d}-{i}'
        fpath += f'/result.npz'
        res = np.load(fpath, allow_pickle=True).get('result').item()
        result.update(res)
    return result


def show(markdown=False):
    print(f'\n\nResults >>>')
    for d in D_LIST:
        show_method(MODEL, d, title=True)


def show_method(model, d, bs=None, title=False):
    result = load(model, d, bs)
    if len(list(result.keys())) == 0:
        name = 'tetradat' if bs is None else f'{bs}'
        text = name + ' '*max(0, 10-len(name)) + ' >>> NOT READY'
        print(text)

    succ = np.sum([r['success'] for r in result.values() if r])
    full = len(result.keys())

    asr = succ/full*100
    dx0 = np.mean([r['changes'] for r in result.values() if r['success']])
    dx1 = np.mean([r['dx1'] for r in result.values() if r['success']])
    dx2 = np.mean([r['dx2'] for r in result.values() if r['success']])
    
    name = 'tetradat' if bs is None else f'{bs}'
    text = ''
    if title:
        text += f'\n\n{model} (attr: {MODEL_ATTR}; d = {d}) | (total {full})\n'
    text += name + ' '*max(0, 10-len(name)) + ' >>> '
    text += f'asr: {asr:-6.2f}% | '
    text += f'total: {full} | '
    text += f'changes: {dx0:-6.0f} | '
    text += f'dx1: {dx1:-8.1f} | '
    text += f'dx2: {dx2:-8.1f}'
    print(text)


if __name__ == '__main__':
    show()