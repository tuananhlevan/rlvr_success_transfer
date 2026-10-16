"""Restore or start one bounded notebook slice, keeping a verified vault."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

from session_store import candidates,publish,restore

PROJECT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--vault',type=Path,required=True)
    parser.add_argument('--precision',choices=['bfloat16','float16'],required=True)
    parser.add_argument('--hours',type=float,default=6.)
    parser.add_argument('--pause-margin-minutes',type=float,default=120.)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--migration',action='store_true',help='Restore the exported BF16 step-1 checkpoint if no saved session exists')
    parser.add_argument('--artifact-dir',type=Path,default=PROJECT/'artifacts')
    parser.add_argument('--mode',choices=['run','benchmark'],default='run')
    args=parser.parse_args();run=args.run_dir.resolve();vault=args.vault.resolve()
    if args.mode=='benchmark' and args.migration:parser.error('Use a separate vault and run directory for the benchmark')
    if candidates(vault) and not (run.exists() and any(run.iterdir())):
        print('Restoring the saved notebook session...',flush=True);restore(vault,run)
    elif not (run/'inputs.json').exists() and args.migration:
        # LFS files must already be materialized or attached as a Kaggle dataset.
        subprocess.run([sys.executable,str(PROJECT/'scripts/artifacts.py'),'--artifact-dir',str(args.artifact_dir),
            'restore','--run-dir',str(run)],check=True)
        if args.precision=='float16':
            print('WARNING: continuing BF16 step 1 under FP16 creates a mixed-precision continuation variant; keep it separate from fresh FP16 seeds.',flush=True)
            (run/'precision_transition.json').write_text(json.dumps(dict(from_precision='bfloat16',to_precision='float16',at_outer_step=1))+'\n')
    run.mkdir(parents=True,exist_ok=True)
    command=[str(PROJECT/'.venv/train/bin/python'),'-B','-u',str(PROJECT/'studies/rlvr_success_transfer/reproduce_failure.py'),
        args.mode,'--seed',str(args.seed),'--run-dir',str(run),'--persist-dir',str(vault),
        '--max-hours',str(args.hours),'--pause-margin-minutes',str(args.pause_margin_minutes),
        '--precision',args.precision,'--physical-batch-size','1','--generation-chunk-responses','512',
        '--max-num-seqs','16','--max-num-batched-tokens','2048','--grading-workers','2']
    if args.precision=='float16':command.append('--enforce-eager')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',
        RLVR_TRAIN_PYTHON=str(PROJECT/'.venv/train/bin/python'),
        RLVR_INFER_PYTHON=str(PROJECT/'.venv/infer/bin/python'),HF_HUB_DISABLE_XET='1',RLVR_CPU_THREADS='2')
    print('Launching a bounded slice; scientific target remains 120 outer steps.',flush=True)
    with (run/'live.log').open('a') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        try:
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        except KeyboardInterrupt:
            status_path=run/'status.json'
            if status_path.exists():
                status=json.loads(status_path.read_text());worker=status.get('pid')
                if worker:
                    try:
                        # Only terminate a worker belonging to this driver.
                        parent=int(Path(f'/proc/{worker}/stat').read_text().split(') ')[1].split()[1])
                        if parent==process.pid:os.killpg(worker,signal.SIGTERM)
                    except (OSError,ValueError,IndexError):pass
            process.terminate()
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if code:raise SystemExit(f'Worker failed ({code}); inspect {run}/worker.log. The last published snapshot remains available.')
    status=json.loads((run/'status.json').read_text())
    print(json.dumps(status,indent=2))
    if status.get('status')=='PAUSED':print('Session slice saved. Start the same notebook again to continue; this is not a completed result.')


if __name__=='__main__':main()
