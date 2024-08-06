import subprocess
import time


FOLDER         = '/home/jovyan/chertkov/tetradat'
NAME_PREFIX    = 'chertkov_tetradat_mooa'
CONDA_ENV      = 'chertkov_tetradat'
MODELS         = [
    'adv_inception',
    'adv_inception_resnet',
    'alexnet',
    'googlenet',
    'inception',
    'mobilenet',
    'resnet']
ARGS           = {
    'task': 'attack',
    'kind': 'bs_mooa',
    'data': 'imagenet',
    'model_attr': 'vgg',
    'model': None,
    'img_portion': None,
    'gpu': None}


class Tmuxmanager:
    def __init__(self, session, rewrite=False):
        # Note that we always work in a new session:
        self.create_session(session, rewrite)

    def create_session(self, session, rewrite=False):
        if ' ' in session:
            raise ValueError('Input can not contain spaces')

        prc = subprocess.getoutput('tmux ls')
        if session + ':' in prc:
            if rewrite:
                act = 'y'
            else:
                msg = f'Session "{session}" already exists. Remove? [y/n] '
                act = input(msg)
            if act == 'y':
                self.delete_session(session)
            else:
                raise ValueError('Session is already exists')

        prc = subprocess.getoutput(f'tmux new -d -s {session}')
        if prc:
            msg = 'Can not create session. Output:\n\n' + str(prc) + '\n\n'
            raise ValueError(msg)

        self.session = session

    def create_window(self, window, sleep=2):
        if ' ' in window:
            raise ValueError('Input can not contain spaces')
        
        self._cmd(f'tmux new-window -n {window}')
        self._cmd(f'tmux select-window -t {window}')
        time.sleep(sleep) # To make sure that the window has time to be created

    def delete_session(self, session):
        if ' ' in session:
            raise ValueError('Input can not contain spaces')

        prc = subprocess.getoutput(f'tmux kill-session -t {session}')
        if prc:
            msg = 'Can not delete session. Output:\n\n' + str(prc) + '\n\n'
            raise ValueError(msg)

    def run_python(self, name, folder, script, args={}, conda_env=None):
        if ' ' in name:
            raise ValueError('Input "name" can not contain spaces')
        if ' ' in folder:
            raise ValueError('Input "folder" can not contain spaces')
        if ' ' in script:
            raise ValueError('Input "script" can not contain spaces')
        if conda_env and ' ' in conda_env:
            raise ValueError('Input "conda_env" can not contain spaces')

        args_str = []
        for uid, value in args.items():
            uid = str(uid)
            value = str(value)
            if ' ' in uid or ' ' in value:
                raise ValueError('Input "args" can not contain spaces')
            args_str.append(f'--{uid} {value}')
        args_str = ' '.join(args_str)
        if args_str:
            args_str = ' ' + args_str

        self.create_window(name)
        self._cmd(f'cd {folder}', name)
        if conda_env:
            self._cmd(f'source ~/miniconda3/etc/profile.d/conda.sh', name)
            self._cmd(f'conda activate {conda_env}', name)
        self._cmd(f'python {script}{args_str}', name)

    def _cmd(self, command, window=None, sleep=0.5):
        if window is None:
            window = '0'
        
        command_full = f'tmux send-keys '
        command_full += f'-t {self.session}:{window} Space '
        command_full += ' Space '.join(command.split(' ')) + ' Enter'
        
        prc = subprocess.getoutput(command_full)
        if prc:
            msg = f'Error for command "{command_full}". '
            msg += 'Output:\n\n' + str(prc) + '\n\n'
            raise ValueError(msg)

        time.sleep(sleep)


def run():
    name = 'simple_digital'
    name_short = 'digital'
    
    tm = Tmuxmanager(NAME_PREFIX)

    for i, model in enumerate(MODELS):
        for img_portion in range(1, 11):
            ARGS['model'] = model
            ARGS['gpu'] = i
            ARGS['img_portion'] = img_portion

            tm.run_python(name=f'{model}_{img_portion}',
                folder=FOLDER, script='manager.py',
                args=ARGS, conda_env=CONDA_ENV)


if __name__ == '__main__':
    run()