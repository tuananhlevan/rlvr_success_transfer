"""Check visible subprocess progress, failures and safe setup retries."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_portability import load

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from setup_progress import SetupReporter,setup_process_kind
bootstrap=load('progress_bootstrap','scripts/bootstrap_notebook.py')


class SetupProgressTests(unittest.TestCase):
    def test_streams_child_stdout_stderr_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()) as visible:
            reporter=SetupReporter(temp)
            reporter.run('small child',[sys.executable,'-c',"import sys; print('visible output'); print('visible error',file=sys.stderr)"])
            self.assertIn('visible output',visible.getvalue())
            self.assertIn('visible error',Path(temp,'setup.log').read_text())
            self.assertEqual(json.loads(Path(temp,'setup_status.json').read_text())['status'],'COMPLETED')
            with self.assertRaises(subprocess.CalledProcessError):
                reporter.run('failed child',[sys.executable,'-c','raise SystemExit(7)'])
            self.assertEqual(json.loads(Path(temp,'setup_status.json').read_text())['status'],'FAILED')

    def test_heartbeat_and_interrupt_cleanup(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            reporter=SetupReporter(temp,heartbeat_seconds=.01)
            reporter.run('silent child',[sys.executable,'-c','import time; time.sleep(.1)'])
            self.assertIn('still running',Path(temp,'setup.log').read_text())
            child=unittest.mock.MagicMock()
            child.stdout.__iter__.side_effect=KeyboardInterrupt
            child.poll.return_value=None;child.pid=123456
            with patch('setup_progress.subprocess.Popen',return_value=child),patch('setup_progress.os.killpg') as terminate:
                with self.assertRaises(KeyboardInterrupt):reporter.run('interrupt',['fixture'])
                terminate.assert_called_once();child.wait.assert_called_once()
            self.assertEqual(json.loads(Path(temp,'setup_status.json').read_text())['status'],'INTERRUPTED')

    def test_restart_matcher_excludes_training_and_other_checkouts(self):
        root=Path('/fixture')
        script=str(root/'studies/rlvr_success_transfer/reproduce_failure.py')
        self.assertEqual(setup_process_kind(['python',script,'prepare'],root),'prepare')
        self.assertIsNone(setup_process_kind(['python',script,'run'],root))
        self.assertIsNone(setup_process_kind(['python','/other/scripts/bootstrap_notebook.py'],root))
        self.assertEqual(setup_process_kind(['python',str(root/'scripts/artifacts.py'),'restore'],root),'restore')

    def test_completed_install_reuses_packages_but_runs_gpu_doctor(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp);work=root/'runtime';work.mkdir()
            for phase in ['train','infer']:
                python=work/f'envs/{phase}/bin/python';python.parent.mkdir(parents=True);python.touch()
                for suffix in ['txt','lock.txt']:
                    path=root/f'requirements/{phase}.{suffix}';path.parent.mkdir(exist_ok=True);path.write_text('fixture==1\n')
            with patch.object(bootstrap,'PROJECT',root):
                fingerprint=bootstrap.installation_fingerprint('float16',False)
                receipt=work/'setup_receipt.json'
                receipt.write_text(json.dumps(dict(fingerprint=fingerprint,installation_completed=True)))
                with patch.object(bootstrap,'hardware'),patch.object(bootstrap.subprocess,'run') as execute,patch.object(sys,'argv',['bootstrap','--work-root',str(work),'--precision','float16']):
                    bootstrap.main()
                execute.assert_called_once()
                self.assertIn(str(root/'scripts/doctor.py'),execute.call_args.args[0])
                self.assertTrue(json.loads(receipt.read_text())['gpu_execution_tested'])
                with patch.object(bootstrap,'hardware'),patch.object(bootstrap.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'doctor')),patch.object(sys,'argv',['bootstrap','--work-root',str(work),'--precision','float16']):
                    with self.assertRaises(subprocess.CalledProcessError):bootstrap.main()
                self.assertFalse(json.loads(receipt.read_text())['gpu_execution_tested'])
                (root/'requirements/train.lock.txt').write_text('fixture==2\n')
                self.assertNotEqual(fingerprint,bootstrap.installation_fingerprint('float16',False))


if __name__=='__main__':unittest.main()
