import matplotlib.pyplot as plt
import numpy as np
import os


MODELS = ['alexnet', 'googlenet', 'inception', 'mobilenet', 'resnet']
MODEL_ATTR = 'vgg'
BASELINES = ['onepixel', 'pixle', 'square']
CLASSES = 1000
SEED = 0


def get_image(num, model, bs=None):
    fpath = f'result/imagenet-{model}/attack-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += f'/img/{num}/changed.png'
    if not os.path.isfile(fpath):
        return
    return plt.imread(fpath)


def plot(with_onepixel=True, num_total=5, dpi=300):
    for model in MODELS:
        images = []

        for _ in range(1000000):
            image_num = np.random.choice(CLASSES)

            images_curr = []
            for bs in BASELINES:
                if not with_onepixel and bs == 'onepixel':
                    continue
                images_curr.append(get_image(image_num, model, bs))
            images_curr.append(get_image(image_num, model))

            if len([True for img in images_curr if img is None]):
                continue
            else:
                images.append(images_curr)

            if len(images) >= num_total:
                break

        fig = plt.figure(figsize=(16, 12 if with_onepixel else 9))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        cnt = 0
        BS = BASELINES if with_onepixel else BASELINES[1:]
        for j, method in enumerate(BS + ['TETRADAT']):
            for i, num in enumerate(range(num_total)):
                cnt += 1
                fig.add_subplot(4, num_total, cnt)
                plt.imshow(images[i][j])
                # plt.title(method, fontsize=9)
                plt.axis('off')

        fname = f'all_{model}' if with_onepixel else f'part_{model}'
        os.makedirs('result/_show', exist_ok=True)
        plt.savefig(f'result/_show/{fname}.png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)

def show():
    for model in MODELS:
        for num, bs in enumerate(BASELINES):
            show_method(model, bs,  title=(num==0))
        show_method(model)

    plot()
    plot(with_onepixel=False)


def show_method(model, bs=None, title=False):
    fpath = f'result/imagenet-{model}/attack-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += f'/result.npz'

    result = np.load(fpath, allow_pickle=True).get('result').item()

    succ = np.sum([r['success'] for r in result.values() if r])
    full = len(result.keys())

    dx0 = np.mean([r['changes'] for r in result.values() if r['success']])
    dx1 = np.mean([r['dx1'] for r in result.values() if r['success']])
    dx2 = np.mean([r['dx2'] for r in result.values() if r['success']])

    name = 'tetradat' if bs is None else f'{bs}'

    text = ''
    if title:
        text += f'\n\n{model} (attr: {MODEL_ATTR}) | (total images {full})\n'
    text += name + ' '*max(0, 10-len(name)) + ' >>> '
    text += f'Succ: {succ/full*100:-6.2f}% | '
    text += f'changes: {dx0:-6.0f} | '
    text += f'dx1: {dx1:-8.1f} | '
    text += f'dx2: {dx2:-8.1f}'
    print(text)


if __name__ == '__main__':
    np.random.seed(SEED)
    show()
