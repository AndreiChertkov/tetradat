"""Script to run computations on zhores cluster.

Notes:
- To check status, use "squeue" command.
- Delete the running task as "scancel NUMBER".

Do manually before the run (?): # TODO: move it into the code

# module avail
# module load python/anaconda3
# conda info --envs
# conda activate && conda remove --name tetradat --all -y
# conda create --name tetradat python=3.8 -y
# conda activate tetradat
# conda list
# pip install teneva_opti==0.5.1 torch==1.12.1+cu113 torchvision==0.13.1+cu113 matplotlib requests urllib3 torchattacks==3.4.0 --extra-index-url https://download.pytorch.org/whl/cu113 && pip install triton
# conda list

"""
from copy import deepcopy as copy
import os
import subprocess
import sys
import time


OPTIONS = {
    'args': {
        'task': 'attack_target',
        'kind': 'attr',
        'data': 'imagenet',
        'model_attr': 'alexnet',
    },
    'opts': {
        'env': 'tetradat',
        'file': 'manager',
        'days': 2,
        'hours': 0,
        'memory': 15,
        'out': 'zhores_out',
        'gpu': True,
    }
}


TASKS = {
    '11-tet': {
        'args': {
            'model': 'googlenet',
        },
        'opts': {
            'out': 'result/imagenet-googlenet/attack_target-attr-alexnet'
        }
    },
    '12-tet': {
        'args': {
            'model': 'inception',
        },
        'opts': {
            'out': 'result/imagenet-inception/attack_target-attr-alexnet'
        }
    },
    '13-tet': {
        'args': {
            'model': 'mobilenet',
        },
        'opts': {
            'out': 'result/imagenet-mobilenet/attack_target-attr-alexnet'
        }
    },
    '14-tet': {
        'args': {
            'model': 'resnet',
        },
        'opts': {
            'out': 'result/imagenet-resnet/attack_target-attr-alexnet'
        }
    },
    '15-tet': {
        'args': {
            'model': 'vgg',
        },
        'opts': {
            'out': 'result/imagenet-vgg/attack_target-attr-alexnet'
        }
    },
    '21-tet': {
        'args': {
            'model': 'googlenet',
            'kind': 'bs_onepixel',
        },
        'opts': {
            'out': 'result/imagenet-googlenet/attack_target-bs_onepixel'
        }
    },
    '22-tet': {
        'args': {
            'model': 'inception',
            'kind': 'bs_onepixel',
        },
        'opts': {
            'out': 'result/imagenet-inception/attack_target-bs_onepixel'
        }
    },
    '23-tet': {
        'args': {
            'model': 'mobilenet',
            'kind': 'bs_onepixel',
        },
        'opts': {
            'out': 'result/imagenet-mobilenet/attack_target-bs_onepixel'
        }
    },
    '24-tet': {
        'args': {
            'model': 'resnet',
            'kind': 'bs_onepixel',
        },
        'opts': {
            'out': 'result/imagenet-resnet/attack_target-bs_onepixel'
        }
    },
    '25-tet': {
        'args': {
            'model': 'vgg',
            'kind': 'bs_onepixel',
        },
        'opts': {
            'out': 'result/imagenet-vgg/attack_target-bs_onepixel'
        }
    },
    '31-tet': {
        'args': {
            'model': 'googlenet',
            'kind': 'bs_pixle',
        },
        'opts': {
            'out': 'result/imagenet-googlenet/attack_target-bs_pixle'
        }
    },
    '32-tet': {
        'args': {
            'model': 'inception',
            'kind': 'bs_pixle',
        },
        'opts': {
            'out': 'result/imagenet-inception/attack_target-bs_pixle'
        }
    },
    '33-tet': {
        'args': {
            'model': 'mobilenet',
            'kind': 'bs_pixle',
        },
        'opts': {
            'out': 'result/imagenet-mobilenet/attack_target-bs_pixle'
        }
    },
    '34-tet': {
        'args': {
            'model': 'resnet',
            'kind': 'bs_pixle',
        },
        'opts': {
            'out': 'result/imagenet-resnet/attack_target-bs_pixle'
        }
    },
    '35-tet': {
        'args': {
            'model': 'vgg',
            'kind': 'bs_pixle',
        },
        'opts': {
            'out': 'result/imagenet-vgg/attack_target-bs_pixle'
        }
    },
    '41-tet': {
        'args': {
            'model': 'googlenet',
            'kind': 'bs_square',
        },
        'opts': {
            'out': 'result/imagenet-googlenet/attack_target-bs_square'
        }
    },
    '42-tet': {
        'args': {
            'model': 'inception',
            'kind': 'bs_square',
        },
        'opts': {
            'out': 'result/imagenet-inception/attack_target-bs_square'
        }
    },
    '43-tet': {
        'args': {
            'model': 'mobilenet',
            'kind': 'bs_square',
        },
        'opts': {
            'out': 'result/imagenet-mobilenet/attack_target-bs_square'
        }
    },
    '44-tet': {
        'args': {
            'model': 'resnet',
            'kind': 'bs_square',
        },
        'opts': {
            'out': 'result/imagenet-resnet/attack_target-bs_square'
        }
    },
    '45-tet': {
        'args': {
            'model': 'vgg',
            'kind': 'bs_square',
        },
        'opts': {
            'out': 'result/imagenet-vgg/attack_target-bs_square'
        }
    },
}


def zhores(kind='main'):
    if kind != 'main':
        raise NotImplementedError
    else:
        tasks = TASKS

    for name, task in tasks.items():
        opts = {**OPTIONS.get('opts', {}), **task.get('opts', {})}
        args = {**OPTIONS.get('args', {}), **task.get('args', {})}

        text = '#!/bin/bash -l\n'

        text += f'\n#SBATCH --job-name={name}'

        fold = opts['out']
        file = f'zhores_out_{name}.txt'
        text += f'\n#SBATCH --output={fold}/{file}'
        os.makedirs(fold, exist_ok=True)

        d = str(opts['days'])
        h = str(opts['hours'])
        h = '0' + h if len(h) == 1 else h
        text += f'\n#SBATCH --time={d}-{h}:00:00'

        with_gpu = True
        if opts['gpu']:
            text += '\n#SBATCH --partition gpu'
        else:
            text += '\n#SBATCH --partition mem'

        text += '\n#SBATCH --nodes=1'

        if opts['gpu']:
            text += '\n#SBATCH --gpus=1'

        mem = str(opts['memory'])
        text += f'\n#SBATCH --mem={mem}GB'

        text += '\n\nmodule rm *'

        text += '\nmodule load python/anaconda3'

        if opts['gpu']:
            text += '\nmodule load gpu/cuda-11.3'

        env = opts['env']
        text += f'\nsource activate {env}'
        text += f'\nconda activate {env}'

        text += '\n\n' + f'srun python3 {opts["file"]}.py'
        for name, value in args.items():
            text += f' --{name} {value}'

        text += '\n\n' + 'exit 0'

        with open('___zhores_run.sh', 'w') as f:
            f.write(text)

        prc = subprocess.getoutput('sbatch ___zhores_run.sh')
        time.sleep(0.5)
        os.remove('___zhores_run.sh')

        if 'command not found' in prc:
            print('!!! Error: can not run "sbatch"')
        else:
            print(prc)


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'main'
    zhores(kind)
