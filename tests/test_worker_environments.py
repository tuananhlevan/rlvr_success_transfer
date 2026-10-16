"""Launch real train/infer subprocesses through directory and executable symlinks."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import venv

from test_portability import driver


class WorkerEnvironmentTests(unittest.TestCase):
    def test_main_launches_workers_in_their_virtual_environments(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);project=root/'project';(project/'.venv').mkdir(parents=True)
            interpreters={}
            for phase in ['train','infer']:
                environment=root/'runtime/envs'/phase
                venv.EnvBuilder(with_pip=False,symlinks=True).create(environment)
                python=environment/'bin/python'
                self.assertTrue(python.is_symlink())
                paths=json.loads(subprocess.check_output([str(python),'-c',
                    'import json,sysconfig; print(json.dumps(sysconfig.get_paths()))'],text=True))
                Path(paths['purelib'],'rlvr_worker_env_probe.py').write_text(f'IDENTITY={phase!r}\n')
                (project/'.venv'/phase).symlink_to(environment,target_is_directory=True)
                interpreters[phase]=str(project/f'.venv/{phase}/bin/python')
            worker=root/'fixture_worker.py'
            worker.write_text('import json,sys,rlvr_worker_env_probe\n'
                'print(json.dumps(dict(mode=sys.argv[1],identity=rlvr_worker_env_probe.IDENTITY,prefix=sys.prefix)))\n')
            run=root/'run';real_sleep=time.sleep
            def launch(*args):
                driver.subprocess_phase(driver.TRAIN_PYTHON,['update'],'fixture optimization',10)
                driver.subprocess_phase(driver.INFER_PYTHON,['generate'],'fixture generation',10)
            arguments=['driver','run','--run-dir',str(run),'--train-python',interpreters['train'],
                '--infer-python',interpreters['infer']]
            with patch.multiple(driver,RUN=run,SCRATCH=run/'scratch',TRAIN_PYTHON=sys.executable,
                    INFER_PYTHON=sys.executable,PERSIST_DIR=None,SESSION_END_EPOCH=0,RUN_DEADLINE=None,__file__=str(worker)),\
                    patch.dict(driver.CONFIG,dict(driver.CONFIG)),patch.object(sys,'argv',arguments),\
                    patch.object(driver,'driver',side_effect=launch),\
                    patch.object(driver.time,'sleep',side_effect=lambda _:real_sleep(.01)):
                driver.main()
                self.assertEqual(driver.TRAIN_PYTHON,interpreters['train'])
                self.assertEqual(driver.INFER_PYTHON,interpreters['infer'])
            output=[json.loads(line) for line in (run/'worker.log').read_text().splitlines()]
            self.assertEqual([(r['mode'],r['identity']) for r in output],[('update','train'),('generate','infer')])
            for record in output:
                self.assertEqual(Path(record['prefix']).resolve(),root/'runtime/envs'/record['identity'])

    def test_python_lookup_expands_paths_without_dereferencing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'python-worker').symlink_to(sys.executable)
            with patch.dict(os.environ,{'PATH':str(root)}):
                self.assertEqual(driver.worker_python('python-worker'),str(root/'python-worker'))
            with patch.object(driver.os.path,'expanduser',return_value=str(root/'python-worker')):
                self.assertEqual(driver.worker_python('~/python-worker'),str(root/'python-worker'))


if __name__=='__main__':unittest.main()
