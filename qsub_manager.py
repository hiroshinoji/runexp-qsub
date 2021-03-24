from hashlib import md5
import logging
import numpy as np
import os
import itertools
from pathlib import Path
from runexp import Workflow
import signal
from subprocess import Popen
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

"""Usage:

with runexp.py

# add jobs under some common configuration

template = '''#!/bin/bash

#$-l __MACHINE__=1
#$-l h_rt=__TIME__
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source __VENV__/bin/activate  # __XXX__ is a placeholder, which can be filled by variable `xxx` given to `new_env` argument.
# conda activate __CONDA__  # In this case, variable `conda` will replace __CONDA__.

cd __DIR__

__CMD__
'''

with QsubWorkflow(template=template) as workflow:
    with workflow.new_env(time='5:00', venv='/path/venv') as env:
        env.add_task('cmd1')  # this will regist new qsub script (might be deleted later)
        env.add_task('cmd2')
    with workflow.new_env(time='3:00', venv='/path/venv') as env:
        env.add_task('cmd3')
    workflow.run()  # wrap exp.run()
"""

def run_qsub(add_task_fns, template, dry_run=False, num_jobs=128, time='5:00:00', group='gac50547',
             sync=True, machine='rt_G.small', keep_going=True, **env_args):
    if not isinstance(add_task_fns, list):
        add_task_fns = [add_task_fns]
    with QsubWorkflow(template=template, dry_run=dry_run, num_jobs=num_jobs, keep_going=keep_going,
                      qsub_dir='qsubs') as workflow:
        with workflow.new_env(machine=machine, time=time, delete_qsub=False,
                              sync=sync, conda=conda, group=group, **env_args) as env:
            for add_task_fn in add_task_fns:
                add_task_fn(env)
            workflow.run()

class QsubTask:
    def __init__(self, env, cmd, src, tgt):
        self.env = env
        self.cmd = cmd
        self.src = src or []
        self.tgt = tgt or []

class QsubEnv:

    def __init__(self,
                 workflow,
                 time='5:00:00',
                 group='gac50428',
                 machine='rt_G.small',
                 workdir='./',
                 delete_qsub=True,
                 sync=True,
                 **template_args):
        self.workflow = workflow
        self.time = time
        self.group = group
        self.machine = machine
        self.workdir = workdir
        self.delete_qsub = delete_qsub
        self.sync = 'y' if sync else 'n'
        self.template_args = dict([('__{}__'.format(k.upper()), v) for k, v in template_args.items()])

    def add_task(self, cmd, src=None, tgt=None):
        self.workflow.add_task(QsubTask(self, cmd, src, tgt))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class QsubWorkflow:

    def __init__(self, qsub_dir = "./", template = None, **args):
        self.tasks = []
        self.qsub_paths = []
        self.qsub_dir = Path(qsub_dir)

        self.exp = Workflow(args=[])
        self.exp.set_options(**args)

        self.template = template or '''
#!/bin/bash

#$-l __MACHINE__=1
#$-l h_rt=__TIME__
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source __CUDA_MODULE__
. /home/aaa10317sm/anaconda3/etc/profile.d/conda.sh
conda activate __CONDA__

cd __DIR__

__CMD__'''

    def new_env(self, **args):
        env = QsubEnv(self, **args)
        return env

    def add_task(self, task):
        qsub_content = self._mk_qsub_content(task)
        code = md5(qsub_content.encode('utf-8')).hexdigest()

        if not self.qsub_dir.exists():
            self.qsub_dir.mkdir()
        qsub_path = self.qsub_dir / 'qsub-{}.sh'.format(code)
        with qsub_path.open('wt') as f:
            f.write(qsub_content)

        self.tasks.append(task)
        self.qsub_paths.append(qsub_path)

        self.exp(name = task.tgt, source=task.src, target=task.tgt,
                 rule='qsub -g {} -sync {} {}'.format(
                     task.env.group, task.env.sync, qsub_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for task, qsub_path in zip(self.tasks, self.qsub_paths):
            if task.env.delete_qsub and qsub_path.exists():
                qsub_path.unlink()

    def run(self):
        self.exp.run()

    def _mk_qsub_content(self, task):
        env = task.env
        workdir = Path(os.getcwd()) / env.workdir
        replace_dict = {
            **env.template_args,
            **{'__MACHINE__': env.machine,
               '__TIME__': env.time,
               '__DIR__': workdir,
               '__CMD__': task.cmd,
            }}
        template = self.template
        for k, v in replace_dict.items():
            template = template.replace(str(k), str(v))
        return template


class ParamSet:

    def __init__(self):
        self.params = []

    def add_list(self, params = []):
        self.params.extend(params)

    def add_grid_search(self, parameters = {}):
        self.params.extend(self._product(parameters))

    def add_random_samples(self, param_to_generator, num_samples):
        for i in range(num_samples):
            param = dict([(k, param_to_generator[k]()) for k in param_to_generator])
            self.params.append(param)

    def sorted_params(self):
        return [sorted(p.items()) for p in self.params]

    def _product(self, parameters):
        '''
        >>> product({'x': [0, 1, 2], 'y': [1, 3, 5]})
        [{'x': 0, 'y': 1}, {'x': 0, 'y': 3}, {'x': 0, 'y': 5},
        {'x': 1, 'y': 1}, {'x': 1, 'y': 3}, {'x': 1, 'y': 5},
        {'x': 2, 'y': 1}, {'x': 2, 'y': 3}, {'x': 2, 'y': 5}]
        '''
        keys = sorted(parameters)
        values = [parameters[key] for key in keys]
        values_product = itertools.product(*values)
        return [dict(zip(keys, vals)) for vals in values_product]


class RandGen:

    @staticmethod
    def categorical(choices=[]):
        return lambda: np.random.choice(choices)

    @staticmethod
    def uniform(low, high, precision=10):
        """los <= x < high"""
        return lambda: round(np.random.uniform(low, high), precision)

    @staticmethod
    def loguniform(low, high, precision=10):
        return lambda: round(np.exp(np.random.uniform(np.log(low), np.log(high))), precision)

    @staticmethod
    def uniform_int(low, high):
        """low <= x < high"""
        return lambda: np.random.randint(low, high)

    @staticmethod
    def store_true():
        return categorical(['store_true_yes', 'store_true_no'])


class QsubJob:

    def __init__(self,
                 cmd,
                 time='5:00:00',
                 conda='torch-1.7',
                 group='gac50428',
                 cuda_module='~/load_cuda_moduels.sh',
                 machine='rt_G.small',
                 delete_qsub=True,
                 sync=True,
                 dry_run=False):
        self.cmd = cmd
        self.time = time
        self.conda = conda
        self.group = group
        self.cuda_module = cuda_module
        self.machine = machine
        self.delete_qsub = delete_qsub
        self.sync = 'y' if sync else 'n'
        self.dry_run = dry_run

        self.qsub_path = None

    def run(self):
        assert self.qsub_path is not None
        cmd = 'qsub -g {} -sync {} {}'.format(self.group, self.sync, self.qsub_path)
        logger.info('Run: {} ({})'.format(cmd, self.cmd))
        if not self.dry_run:
            try:
                proc = Popen(cmd, shell=True, close_fds=True, preexec_fn=os.setpgrp)
                ret = proc.wait()
            except  KeyboardInterrupt as e:
                logger.info('Terminating the running task...')
                os.killpg(proc.pid, signal.SIGTERM)
                return 1
            except Exception as e:
                logger.info('ExecCommand: Unknown error raised')
                os.killpg(proc.pid, signal.SIGTERM)
                return 1
            return ret

    def __enter__(self):
        qsub_content = self._mk_content()
        code = md5(qsub_content.encode('utf-8')).hexdigest()
        qsub_path = 'qsub-{}.sh'.format(code)
        with Path(qsub_path).open('wt') as f:
            f.write(qsub_content)
        self.qsub_path = qsub_path
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.delete_qsub and Path(self.qsub_path).exists():
            Path(self.qsub_path).unlink()
        self.qsub_path is None

    def _mk_content(self):
        return '''#!/bin/bash

#$-l {}=1
#$-l h_rt={}
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source {}
. /home/aaa10317sm/anaconda3/etc/profile.d/conda.sh
conda activate {}

cd {}

{}
'''.format(self.machine, self.time, self.cuda_module, self.conda, os.getcwd(), self.cmd)

def run_command(cmd, **args):
    with QsubJob(cmd, **args) as j:
        j.run()

if __name__ == '__main__':

    env_args = {'conda': 'torch-1.7', 'cuda_module': '~/load_cuda_moduels.sh'}
    with QsubWorkflow(dry_run=True, keep_going=True, num_jobs=1) as workflow:
        with workflow.new_env(machine='rt_C.small', **env_args, delete_qsub=False) as env:
            env.add_task('ls -alh > ls1.txt', tgt='ls1.txt')
            env.add_task('ls -al > ls2.txt', tgt='ls2.txt')
        with workflow.new_env(time='0:10:00', **env_args) as env:
            env.add_task('ls > ls3.txt', tgt='ls3.txt')

        workflow.run()
