"""Clone -> setup.py -> run.py, with automatic cloud/GPU configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from session_store import atomic_json,candidates,publish,restore
from setup_progress import SetupReporter,stop_previous_setup

PROJECT=Path(__file__).resolve().parents[1]


def detect_platform():
    if Path('/kaggle/working').exists() or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):return 'kaggle'
    if Path('/content').exists() or any(name.startswith('COLAB_') for name in os.environ):return 'colab'
    return 'local'


def detect_precision():
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader,nounits'],text=True)
        capability=float(output.splitlines()[0].strip())
    except (OSError,ValueError,subprocess.CalledProcessError,IndexError) as error:
        raise RuntimeError('Select a GPU runtime, then rerun setup.py.') from error
    if capability<7.5:raise RuntimeError('This stack needs T4/Turing or newer.')
    return 'bfloat16' if capability>=8 else 'float16'


def mount_drive():
    if Path('/content/drive/MyDrive').exists():return
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as error:
        raise RuntimeError('Run `%run setup.py` in a Colab code cell so Drive authorization can open in the notebook.') from error
    if not Path('/content/drive/MyDrive').exists():raise RuntimeError('Google Drive did not mount; setup has stopped before training.')


def make_config(platform,precision,seed,work_root=None,vault=None,start_from_export=None):
    work=Path(work_root).expanduser().resolve() if work_root else (
        Path('/content/rlvr_runtime') if platform=='colab' else
        Path('/tmp/rlvr_runtime') if platform=='kaggle' else PROJECT/'.runtime')
    label=f'{precision}_seed{seed}'
    persistent=Path(vault).expanduser().resolve() if vault else (
        Path('/content/drive/MyDrive/rlvr_success_transfer')/f'vault_{label}' if platform=='colab' else
        Path('/kaggle/working')/f'vault_{label}' if platform=='kaggle' else PROJECT/'runs'/f'vault_{label}')
    if start_from_export is None:start_from_export=precision=='bfloat16' and seed==1
    if start_from_export and seed!=1:raise ValueError('The exported checkpoint belongs to seed 1; use --fresh for another seed.')
    return dict(format=1,platform=platform,precision=precision,seed=seed,work_root=str(work),
        run_dir=str(work/f'run_{label}'),vault=str(persistent),hf_home=str(work/'huggingface'),
        start_from_export=start_from_export,hours=6.,pause_margin_minutes=120.)


def execution_env(config):
    env=os.environ.copy()
    env.update(HF_HOME=config['hf_home'],HF_HUB_DISABLE_XET='1',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    return env


def initialize_data(config,artifact_dir=None,resume_from=None,reporter=None):
    run=Path(config['run_dir']);vault=Path(config['vault'])
    if (run/'inputs.json').exists():
        if reporter:reporter.output('[setup] Reusing existing prepared inputs and local progress.')
        return 'existing local progress'
    def execute(label,command):
        if reporter:reporter.run(label,command,env=execution_env(config))
        else:subprocess.run(command,check=True,env=execution_env(config))
    def snapshot():
        if reporter:
            with reporter.stage('Save initial session snapshot to persistent storage'):
                publish(run,vault,progress=True)
        else:publish(run,vault)
    source=vault if candidates(vault) else Path(resume_from).expanduser().resolve() if resume_from else None
    if source:
        if reporter:
            with reporter.stage('Restore saved session from persistent storage'):restore(source,run)
        else:restore(source,run)
        if source!=vault:snapshot()
        return 'restored saved session'
    if config['start_from_export']:
        artifacts=Path(artifact_dir).resolve() if artifact_dir else PROJECT/'artifacts'
        command=[sys.executable,str(PROJECT/'scripts/artifacts.py'),'--artifact-dir',str(artifacts)]
        if not artifact_dir:execute('Download exported checkpoint (about 5.5 GB)',[*command,'download'])
        execute('Verify and extract checkpoint (about 6 GB)',[*command,'restore','--run-dir',str(run)])
        if config['precision']=='float16':
            atomic_json(run/'precision_transition.json',dict(from_precision='bfloat16',to_precision='float16',at_outer_step=1))
        origin='exported step-1 checkpoint and pending step-2 rollouts'
    else:
        execute('Download pinned base model and prepare dataset',[str(PROJECT/'.venv/train/bin/python'),str(PROJECT/'studies/rlvr_success_transfer/reproduce_failure.py'),
            'prepare','--run-dir',str(run),'--precision',config['precision']])
        origin='fresh pinned model and dataset'
    snapshot()
    return origin


def setup_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform',choices=['colab','kaggle','local'],default=None)
    parser.add_argument('--precision',choices=['auto','bfloat16','float16'],default='auto')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--work-root',type=Path);parser.add_argument('--vault',type=Path)
    parser.add_argument('--resume-from',type=Path,help='A previous Kaggle output vault attached as input')
    parser.add_argument('--artifact-dir',type=Path,help='Attached migration artifacts instead of another LFS download')
    parser.add_argument('--restart',action='store_true',help='Stop leftover setup installers/downloads for this checkout, retaining caches')
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--fresh',action='store_true');choice.add_argument('--from-checkpoint',action='store_true')
    args=parser.parse_args()
    platform=args.platform or detect_platform()
    precision=detect_precision() if args.precision=='auto' else args.precision
    if platform=='colab' and args.vault is None:mount_drive()
    start=False if args.fresh else True if args.from_checkpoint else None
    config=make_config(platform,precision,args.seed,args.work_root,args.vault,start)
    if args.restart:
        stopped=stop_previous_setup(PROJECT)
        print(f'Stopped leftover setup processes: {stopped}; completed downloads are retained.',flush=True)
    folder=PROJECT/'.cloud';folder.mkdir(exist_ok=True)
    marker=folder/'setup_in_progress.json'
    reporter=SetupReporter(folder)
    atomic_json(marker,dict(platform=platform,precision=precision))
    print(f'Setting up {platform}; GPU 0; precision={precision}',flush=True)
    if precision=='float16':print('T4/FP16 numerical variant: kept separate from BF16 reproduction.',flush=True)
    command=[sys.executable,str(PROJECT/'scripts/bootstrap_notebook.py'),
        '--work-root',config['work_root'],'--precision',precision]
    if precision=='bfloat16':command.append('--prebuilt-kernels')
    reporter.output(f'[setup] Detailed log: {folder/"setup.log"}')
    reporter.run('Install dependencies and check GPU',command,env=execution_env(config))
    origin=initialize_data(config,args.artifact_dir,args.resume_from,reporter=reporter)
    config.update(ready=True,origin=origin)
    atomic_json(folder/'session.json',config);marker.unlink()
    print(f'Setup complete: {origin}.\nCheckpoints: {config["vault"]}\nRun: python run.py',flush=True)


def run_main():
    parser=argparse.ArgumentParser(description='Run or resume the setup.py configuration.')
    parser.add_argument('--hours',type=float,help='Override this session\'s training time budget')
    args=parser.parse_args();folder=PROJECT/'.cloud'
    if (folder/'setup_in_progress.json').exists():raise RuntimeError('Setup is incomplete. Rerun setup.py before starting training.')
    if not (folder/'session.json').exists():raise RuntimeError('Run setup.py first (Colab/Kaggle: `%run setup.py`).')
    config=json.loads((folder/'session.json').read_text())
    if not config.get('ready'):raise RuntimeError('Setup did not complete. Rerun setup.py.')
    for phase in ['train','infer']:
        if not (PROJECT/f'.venv/{phase}/bin/python').exists():raise RuntimeError('Runtime environments are missing. Rerun setup.py in this runtime.')
    hours=args.hours if args.hours is not None else config['hours']
    if hours<=config['pause_margin_minutes']/60:parser.error('Session hours must exceed the two-hour checkpoint margin.')
    command=[sys.executable,str(PROJECT/'scripts/notebook_session.py'),'--run-dir',config['run_dir'],
        '--vault',config['vault'],'--precision',config['precision'],'--seed',str(config['seed']),
        '--hours',str(hours),'--pause-margin-minutes',str(config['pause_margin_minutes'])]
    subprocess.run(command,check=True,env=execution_env(config))
