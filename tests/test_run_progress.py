"""ETA calibration must use current-invocation, complete-cycle evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_progress


class RunProgressTests(unittest.TestCase):
    def test_estimate_ignores_other_machines_and_uses_recent_complete_steps(self):
        records=[dict(measurement_id='old',seconds=999999)]+[
            dict(measurement_id='current',seconds=value) for value in [99999,100,200,300,400,500]]
        estimate=run_progress.training_estimate(records,'current',110,120)
        self.assertEqual(estimate['samples'],5)
        self.assertEqual(estimate['remaining_seconds'],3000)
        self.assertIsNone(run_progress.training_estimate(records,'new-runtime',110,120))
        self.assertIsNone(run_progress.training_estimate(records,'current',120,120))

    def test_visible_calibration_session_budget_and_conservative_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'seed_1/latest_checkpoint';folder.mkdir(parents=True)
            (folder/'commit.json').write_text(json.dumps(dict(step=1)))
            output=[]
            with patch.object(run_progress.time,'time',return_value=100000):
                monitor=run_progress.RunProgress(root,1,6,120,'current',output.append)
                monitor.report()
            self.assertIn('calibrating',''.join(output))
            (folder.parent/'step_timings.jsonl').write_text(json.dumps(dict(measurement_id='current',seconds=3600))+'\n{incomplete')
            output.clear()
            with patch.object(run_progress.time,'time',return_value=100000+5*3600):monitor.report()
            text=''.join(output)
            self.assertIn('Pause margin reached',text)
            self.assertIn('~4d 23h 0m',text)
            self.assertIn('excludes validation',text)
            self.assertIn('additional resumed sessions',text)
            self.assertIn('Full experiment finish: not yet estimated',text)
            (folder/'commit.json').write_text(json.dumps(dict(step=120)))
            output.clear();monitor.report()
            self.assertIn('Evaluation/analysis may still be running',''.join(output))


if __name__=='__main__':unittest.main()
