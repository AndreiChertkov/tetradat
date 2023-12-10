import matplotlib.pyplot as plt
import numpy as np
import os


ROOT = 'result_base'
DATASET = 'imagenet'
MODELS = ['alexnet', 'googlenet', 'inception', 'mobilenet', 'resnet']
MODEL_ATTR = 'vgg'
BASELINES = ['onepixel', 'pixle', 'square']
PLOT_NAMES = ['Onepixel', 'Pixle', 'Square', 'TETRADAT']
CLASSES = 1000
SEED = 42


def check_image(num, model, bs=None):
    return os.path.isfile(get_image_path(num, model, bs))


def get_image(num, model, bs=None):
    if check_image(num, model, bs):
        return plt.imread(get_image_path(num, model, bs))


def get_image_nums(model, bs=None):
    nums = []
    fpath = f'{ROOT}/{DATASET}-{model}/attack-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += '/img'
    for num in os.listdir(fpath):
        if num.isnumeric():
            nums.append(num)
    return nums


def get_image_path(num, model, bs=None):
    fpath = f'{ROOT}/{DATASET}-{model}/attack-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += f'/img/{num}/changed.png'
    return fpath


def load_data(model, bs=None):
    fpath = f'{ROOT}/{DATASET}-{model}/attack-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += f'/result.npz'
    return np.load(fpath, allow_pickle=True).get('result').item()


def plot(num_total=5, dpi=150, bs_ref='onepixel'):
    print(f'\n\nFigures >>>')
    
    for model in MODELS:
        for _ in range(100000000):
            # Note that we select "num_total" random images from the of
            # successful attacks with "onepixel" method (see "bs_ref"),
            # since this method gives the lowest percentage of asr:
            nums = get_image_nums(model, bs_ref)
            nums = np.random.choice(nums, num_total, replace=False)
            
            is_ok = True
            for bs in BASELINES:
                is_ok_curr = [check_image(num, model, bs) for num in nums]
                if False in is_ok_curr:
                    is_ok = False
                    break
            is_ok_curr = [check_image(num, model) for num in nums]
            
            if False in is_ok_curr:
                is_ok = False
            if is_ok:
                break
        
        if not is_ok:
            raise ValueError('Can not find {num_total} images')
        
        def build_title(item):
            t = item['l_new']
            for _ in range(3):
                if len(t) > 30 and ', ' in t:
                    t = ', '.join(t.split(', ')[:-1])
            return t

        images = []
        titles = []
        for bs in BASELINES:
            result = load_data(model, bs)
            images.append([get_image(num, model, bs) for num in nums])
            titles.append([build_title(result[int(num)]) for num in nums])
        result = load_data(model)
        images.append([get_image(num, model) for num in nums])
        titles.append([build_title(result[int(num)]) for num in nums])

        print(f'Model {model} | Selected random images: {nums}')

        fig = plt.figure(figsize=(14, 12))
        plt.subplots_adjust(wspace=0.01, hspace=0.2)

        cnt = 0
        text_positions = [420, 340, 365, 420]
        for j, method in enumerate(BASELINES + ['TETRADAT']):
            for i, num in enumerate(range(num_total)):
                cnt += 1
                fig.add_subplot(4, num_total, cnt)
                plt.imshow(images[j][i])
                plt.title(titles[j][i], fontsize=9, color='#8b1d1d')
                plt.axis('off')
                if i == 0:
                    plt.text(-150, text_positions[j], PLOT_NAMES[j],
                        rotation='vertical',
                        fontfamily='monospace', fontsize=25,
                        color='#000099' if j == 3 else '#000099',
                        fontweight=1000 if j == 3 else 500,
                        horizontalalignment='left')

        fname = f'{model}'
        os.makedirs(f'{ROOT}/_show', exist_ok=True)
        plt.savefig(f'{ROOT}/_show/{fname}.png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        #break


def show():
    print(f'\n\nResults >>>')
    
    for model in MODELS:
        for num, bs in enumerate(BASELINES):
            show_method(model, bs,  title=(num==0))
        show_method(model)

    
def show_method(model, bs=None, title=False):
    result = load_data(model, bs)

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
    text += f'asr: {succ/full*100:-6.2f}% | '
    text += f'total: {full} | '
    text += f'changes: {dx0:-6.0f} | '
    text += f'dx1: {dx1:-8.1f} | '
    text += f'dx2: {dx2:-8.1f}'
    print(text)


if __name__ == '__main__':
    np.random.seed(SEED)
    show()
    plot()