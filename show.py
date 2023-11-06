import numpy as np


MODEL_ATTR = 'alexnet'
MODELS = ['googlenet', 'inception', 'mobilenet', 'resnet', 'vgg']
BASELINES = ['onepixel', 'pixle', 'square']


def show():
    for model in MODELS:
        print()
        for bs in BASELINES:
            show_method(model, bs)
        show_method(model)


def show_method(model, bs=None, target=True):
    fpath = f'result/imagenet-{model}/attack'
    fpath += '_target-' if target else '-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else 'bs_{bs}'
    fpath += f'/result.npz'

    result = np.load(fpath, allow_pickle=True).get('result').item()

    succ = np.sum([r['success'] for r in result.values() if r])
    full = len(result.keys())

    if bs is None:
        text = f'{model} | {MODEL_ATTR}'
    else:
        text = f'{model} | bs: {bs}'

    text += ' '*max(0, 30-len(text)) + ' >>> '
    text += f'Successful: {succ/full*100:-5.2f}% '
    text += f'(total images {full})'

    print(text)


if __name__ == '__main__':
    show()
