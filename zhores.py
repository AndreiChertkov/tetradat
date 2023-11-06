"""Script to run computations on zhores cluster.

Run this script as `python zhores.py` or `python zhores.py init`.
To check status, use "squeue"; to delete the running task do "scancel NUMBER".

Please, install packages within the environment before the run of this script:
$ module rm *
$ module load python/anaconda3
$ module load gpu/cuda-11.3
$ conda remove --name tetradat --all -y
$ conda create --name tetradat python=3.8 -y
$ conda activate tetradat
$ pip install teneva_opti==0.5.1 torch==1.12.1+cu113 torchvision==0.13.1+cu113 matplotlib requests urllib3 torchattacks==3.4.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install triton

"""
import os
import subprocess
import sys


MODEL_ATTR = 'alexnet'
MODELS = ['googlenet', 'inception', 'mobilenet', 'resnet', 'vgg']
BASELINES = ['onepixel', 'pixle', 'square']


OPTIONS = {
    'args': {
        'task': 'attack',
        'kind': 'attr',
        'data': 'imagenet'
    },
    'opts': {
        'env': 'tetradat',
        'file': 'manager',
        'days': 3,
        'hours': 0,
        'memory': 30,
        'out': 'zhores_out',
        'gpu': True
    }
}
OPTIONS_INIT = {
    'args': {
        'task': 'check',
        'kind': 'model',
        'data': 'imagenet'
    },
    'opts': {
        'env': 'tetradat',
        'file': 'manager',
        'days': 0,
        'hours': 3,
        'memory': 15,
        'out': 'zhores_out',
        'gpu': True
    }
}


TASKS = {}
for i, model in enumerate(MODELS, 1):
    TASKS[f'1{i}-tet'] = {
        'args': {
            'model': model,
            'model_attr': MODEL_ATTR
        },
        'opts': {
            'out': f'result/imagenet-{model}/attack_target-attr-{MODEL_ATTR}'
        }
    }
for j, bs in enumerate(BASELINES, 1):
    for i, model in enumerate(MODELS, 1):
        TASKS[f'{j+1}{i}-tet'] = {
            'args': {
                'model': model,
                'kind': f'bs_{bs}'
            },
            'opts': {
                'out': f'result/imagenet-{model}/attack_target-bs_{bs}'
            }
        }


TASKS_INIT = {'test': {'args': [{'kind': 'data'}]}}
for i, model in enumerate(MODELS, 1):
    TASKS_INIT['test']['args'].append({'model': model})


def zhores(kind='main'):
    if kind == 'main':
        options, tasks = OPTIONS, TASKS
    elif kind == 'init':
        options, tasks = OPTIONS_INIT, TASKS_INIT
    else:
        raise NotImplementedError

    for task_name, task in tasks.items():
        opts = {**options.get('opts', {}), **task.get('opts', {})}

        text = '#!/bin/bash -l\n'

        text += f'\n#SBATCH --job-name={task_name}'

        fold = opts['out'] or '.'
        file = f'zhores_out_{task_name}.txt'
        text += f'\n#SBATCH --output={fold}/{file}'
        os.makedirs(fold, exist_ok=True)

        d = str(opts['days'])
        h = str(opts['hours'])
        h = '0' + h if len(h) == 1 else h
        text += f'\n#SBATCH --time={d}-{h}:00:00'

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
        text += '\n'

        args_list = task.get('args', {})
        if isinstance(args_list, dict):
            args_list = [args_list]

        for args in args_list:
            text += f'\nsrun python3 {opts["file"]}.py'
            args = {**options.get('args', {}), **args}
            for name, value in args.items():
                if isinstance(value, bool):
                    text += f' --{name}'
                elif value is not None:
                    text += f' --{name} {value}'

        text += '\n\n' + 'exit 0'

        with open(f'___zhores_run_{task_name}.sh', 'w') as f:
            f.write(text)

        prc = subprocess.getoutput(f'sbatch ___zhores_run_{task_name}.sh')
        os.remove(f'___zhores_run_{task_name}.sh')

        if 'command not found' in prc:
            print('!!! Error: can not run "sbatch"')
        else:
            print(prc)


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'main'
    zhores(kind)
