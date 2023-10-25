import numpy as np

def show(model, model_attr=None, sc=None, bs=None):
    fpath = f'result/imagenet-{model}/attack-'
    if bs is None:
        fpath += f'attr-{model_attr}-sc{sc}'
    else:
        fpath += f'bs_{bs}'
    fpath += f'/result.npz'
    result = np.load(fpath, allow_pickle=True).get('result').item()

    succ = np.sum([r['success'] for r in result.values() if r])
    full = len(result.keys())
    if bs is None:
        text = f'{model} | {model_attr} | {sc}'
    else:
        text = f'{model} | bs: {bs}'

    text += ' '*max(0, 30-len(text)) + ' >>> '
    text += f'Successful: {succ/full*100:-5.2f}% '
    text += f'(total images {full})'

    print(text)


for model in ['alexnet', 'vgg16', 'vgg19']:
    print()
    for bs in ['onepixel', 'pixle']: #, 'square']:
        show(model, bs=bs)
    for model_attr in ['alexnet', 'vgg16', 'vgg19']:
        if model == model_attr:
            continue
        for sc in [6, 8, 10]:
            show(model, model_attr, sc)
