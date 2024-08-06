import matplotlib.pyplot as plt
import numpy as np
import os


ROOT = 'result'
DATASET = 'imagenet'
MODEL = 'vgg'
MODEL_ATTR = 'vgg'
BASELINES = ['square']
PLOT_NAMES = ['Square', 'TETRADAT']
CLASSES = 1000
LLAVA_NUMS = [9, 43, 263, 318, 435, 510, 517, 632, 651, 975]


def check_image(num, bs=None):
    return os.path.isfile(get_image_path(num, bs))


def get_image(num, bs=None):
    # if check_image(num, bs):
    return plt.imread(get_image_path(num, bs))


def get_image_path(num, bs=None):
    fpath = f'{ROOT}/{DATASET}-{MODEL}/attack_llava-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += f'-{num}/img/{LLAVA_NUMS[num-1]}/changed.png'
    return fpath


def load_data(num, bs=None):
    fpath = f'{ROOT}/{DATASET}-{MODEL}/attack_llava-'
    fpath += f'attr-{MODEL_ATTR}' if bs is None else f'bs_{bs}-{MODEL_ATTR}'
    fpath += f'-{num}/result.npz'
    return np.load(fpath, allow_pickle=True).get('result').item()


def plot_text(score, dx1, dx2, out):
    text = f'S = {score:-.2f}; L1 = {dx1:-.0f}; L2 = {dx2:-.0f}'
    plt.text(0., 0.82, text,
        # rotation='vertical',
        fontfamily='monospace', fontsize=9,
        color='#000099', # if j == 3 else '#000099',
        fontweight=300,
        horizontalalignment='left')
    
    out = 'Output: ... ' + out[31:] # The main object in the picture is
    out = [(t+'\n' if i > 0 and i%30==0 else t)
        for i, t in enumerate(out)]
    out = (''.join(out))[:170] 
    out = out.replace('\n ', '\n')
    out = ' '.join(out.split(' ')[:-1])
    out = out + ' ...'
    plt.text(0, 0.05, out,
        # rotation='vertical',
        fontfamily='monospace', fontsize=9,
        color='#000099', # if j == 3 else '#000099',
        fontweight=300,
        horizontalalignment='left')
    
    plt.axis('off')


def show(dpi=150):
    print(f'\n\nResults >>>\n')

    score_te_avg = []
    dx0_te_avg = []
    dx1_te_avg = []
    dx2_te_avg = []

    score_sq_avg = []
    dx0_sq_avg = []
    dx1_sq_avg = []
    dx2_sq_avg = []

    fig = plt.figure(figsize=(9, 15))
    plt.subplots_adjust(wspace=0., hspace=0.05,
        left=0.05, bottom=0.05, right=0.95, top=0.9)

    num_cur = 1

    for num in range(1, len(LLAVA_NUMS)+1):
        print(f'\n>>> # {num} ({LLAVA_NUMS[num-1]})')

        c, score_te, dx0_te, dx1_te, dx2_te, out_base, out_te = show_method(
            num)

        c, score_sq, dx0_sq, dx1_sq, dx2_sq, out_base, out_sq = show_method(
            num, 'square')

        if score_te is None or score_sq is None:
            print('SKIP')
            continue

        score_te_avg.append(score_te)
        dx0_te_avg.append(dx0_te)
        dx1_te_avg.append(dx1_te)
        dx2_te_avg.append(dx2_te)

        score_sq_avg.append(score_sq)
        dx0_sq_avg.append(dx0_sq)
        dx1_sq_avg.append(dx1_sq)
        dx2_sq_avg.append(dx2_sq)

        img_te = get_image(num)
        img_sq = get_image(num, 'square')

        fig.add_subplot(len(LLAVA_NUMS), 4, num_cur)
        plot_text(score_te, dx1_te, dx2_te, out_te)
        num_cur += 1

        if num == 1:
            plt.text(1.32, 1.1, 'TETRADAT',
                fontfamily='monospace', fontsize=12,
                color='#000099',
                fontweight=1000,
                horizontalalignment='left')

        fig.add_subplot(len(LLAVA_NUMS), 4, num_cur)
        plt.imshow(img_te)
        # plt.title(titles[j][i], fontsize=9, color='#8b1d1d')
        plt.axis('off')
        num_cur += 1

        fig.add_subplot(len(LLAVA_NUMS), 4, num_cur)
        plt.imshow(img_sq)
        # plt.title(titles[j][i], fontsize=9, color='#8b1d1d')
        plt.axis('off')
        num_cur += 1

        fig.add_subplot(len(LLAVA_NUMS), 4, num_cur)
        plot_text(score_sq, dx1_sq, dx2_sq, out_sq)
        num_cur += 1

        if num == 1:
            plt.text(-0.65, 1.1, 'Square',
                fontfamily='monospace', fontsize=12,
                color='#000099',
                fontweight=500,
                horizontalalignment='left')

    count = len(score_te_avg)
    assert count == len(LLAVA_NUMS)
    
    score_te_avg = np.mean(score_te_avg)
    dx0_te_avg = np.mean(dx0_te_avg)
    dx1_te_avg = np.mean(dx1_te_avg)
    dx2_te_avg = np.mean(dx2_te_avg)

    score_sq_avg = np.mean(score_sq_avg)
    dx0_sq_avg = np.mean(dx0_sq_avg)
    dx1_sq_avg = np.mean(dx1_sq_avg)
    dx2_sq_avg = np.mean(dx2_sq_avg)

    print('\n\n' + '='*70 + '\n')
    print('TOTAL (average)   :')
    print(f'count             : {count}')
    print(f'score tetradat    : {score_te_avg}')
    print(f'score square      : {score_sq_avg}')
    print(f'dx0   tetradat    : {dx0_te_avg}')
    print(f'dx0   square      : {dx0_sq_avg}')
    print(f'dx1   tetradat    : {dx1_te_avg}')
    print(f'dx1   square      : {dx1_sq_avg}')
    print(f'dx2   tetradat    : {dx2_te_avg}')
    print(f'dx2   square      : {dx2_sq_avg}')
    print('\n' + '='*70 + '\n')

    os.makedirs(f'{ROOT}/_show', exist_ok=True)
    plt.savefig(f'{ROOT}/_show/llava.png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def show_method(num, bs=None, title=True):
    try:
        r = load_data(num, bs)[LLAVA_NUMS[num-1]]
    except Exception as e:
        return None, None, None, None, None, None, None

    if not 'score' in r:
        return None, None, None, None, None, None, None

    c = r['c']
    dx0 = r['changes']
    dx1 = r['dx1']
    dx2 = r['dx2']
    score = r['score']
    name = 'tetradat' if bs is None else f'{bs}'
    out_base = r['out_base']
    out = r['out']

    text = ''
    text += name + ' '*max(0, 10-len(name)) + ' >>> '
    text += f'score: {score*100:-6.2f}% | '
    text += f'changes: {dx0:-6.0f} | '
    text += f'dx1: {dx1:-8.1f} | '
    text += f'dx2: {dx2:-8.1f}'
    #text += f'\n out: {out}'
    #if bs is None:
    #    text += f'\n out_base: {out_base}'
    print(text)

    return c, score, dx0, dx1, dx2, out_base, out


if __name__ == '__main__':
    show()