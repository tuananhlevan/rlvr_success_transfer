"""Expose the actual worker error without replaying historical worker output."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_portability import driver


class WorkerDiagnosticTests(unittest.TestCase):
    def test_tail_is_bounded_and_excludes_previous_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'worker.log'
            old=b'previous error\n';current=('worker details\n'*1000+'actual failure\n').encode()
            path.write_bytes(old+current)
            tail=driver.worker_error_tail(path,len(old),max_bytes=200,max_lines=3)
            self.assertNotIn('previous error',tail)
            self.assertIn('actual failure',tail)
            self.assertLessEqual(len(tail.splitlines()),3)

    def test_failed_worker_prints_its_error_in_driver_output(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()) as visible:
            root=Path(temp);(root/'worker.log').write_text('old unrelated traceback\n')
            def launch(command,**kwargs):
                kwargs['stdout'].write('Traceback (most recent call last):\nRuntimeError: fixture optimization error\n')
                kwargs['stdout'].flush()
                return SimpleNamespace(pid=123,returncode=1,poll=lambda:1)
            with patch.object(driver,'RUN',root),patch.object(driver,'RUN_DEADLINE',None),patch.object(driver,'check_session_margin'),patch.object(driver.subprocess,'Popen',side_effect=launch):
                with self.assertRaisesRegex(RuntimeError,'failed with exit code 1'):
                    driver.subprocess_phase('fixture',['update'],'test optimization',60)
            self.assertIn('fixture optimization error',visible.getvalue())
            self.assertNotIn('old unrelated traceback',visible.getvalue())


if __name__=='__main__':unittest.main()
