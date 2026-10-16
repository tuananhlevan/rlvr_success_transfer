"""Install isolated Python 3.11 environments inside a disposable runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]


def hardware(precision):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,compute_cap',
        '--format=csv,noheader,nounits'],text=True)
    first=output.splitlines()[0].split(',')
    name=first[0].strip();memory=int(first[1]);capability=float(first[2])
    if capability<7.5:raise RuntimeError('T4/Turing or newer is required by this stack.')
    if precision=='bfloat16' and capability<8:
        raise RuntimeError('This GPU has no native BF16. Select L4/A100, or explicitly choose PRECISION="float16" for the T4 numerical variant.')
    if memory<14000:raise RuntimeError('This full-optimizer recipe needs approximately 16 GB GPU memory; small slices do not reduce optimizer memory.')
    if not shutil.which('c++'):raise RuntimeError('A C++ compiler is required by runtime kernels.')
    print(f'GPU 0: {name}, {memory} MiB, capability {capability}; selected precision={precision}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=['bfloat16','float16'],required=True)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--prebuilt-kernels',action='store_true',help='Install pinned FlashInfer caches for BF16 GPUs')
    args=parser.parse_args();hardware(args.precision)
    work=args.work_root.resolve();work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(work).free<25*1024**3:raise RuntimeError('Need at least 25 GiB free disposable disk for environments, model and checkpoints.')
    tools=work/'tools';tools.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','install','--target',str(tools),'uv==0.9.5'],check=True)
    uv=tools/'bin/uv'
    env=os.environ.copy();env.update(UV_PYTHON_INSTALL_DIR=str(work/'python'),
        UV_CACHE_DIR=str(work/'uv-cache'),UV_LINK_MODE='hardlink',HF_HUB_DISABLE_XET='1')
    def run(*arguments):subprocess.run([str(uv),*arguments],check=True,env=env)
    run('python','install','3.11.14')
    venv_root=work/'envs';(PROJECT/'.venv').mkdir(exist_ok=True)
    for phase in ['train','infer']:
        venv=venv_root/phase
        if not venv.exists():run('venv','--python','3.11.14','--seed',str(venv))
        link=PROJECT/'.venv'/phase
        if link.is_symlink() and link.resolve()!=venv:link.unlink()
        if not link.exists():link.symlink_to(venv,target_is_directory=True)
        if link.resolve()!=venv:raise RuntimeError(f'{link} exists outside this bootstrap; choose a clean checkout.')
        python=str(venv/'bin/python')
        torch=['torch==2.9.1'] if phase=='train' else ['torch==2.9.0','torchvision==0.24.0','torchaudio==2.9.0']
        run('pip','install','--python',python,*torch,'--index-url','https://download.pytorch.org/whl/cu128')
        run('pip','install','--python',python,'-r',str(PROJECT/f'requirements/{phase}.lock.txt'))
        run('pip','check','--python',python)
    if args.prebuilt_kernels and args.precision=='bfloat16':
        python=str(venv_root/'infer/bin/python')
        run('pip','install','--python',python,'flashinfer-cubin==0.5.2')
        run('pip','install','--python',python,'flashinfer-jit-cache==0.5.2',
            '--index-url','https://flashinfer.ai/whl/cu128')
    receipt=dict(python='3.11.14',precision=args.precision,work_root=str(work),
        prebuilt_kernels=args.prebuilt_kernels,installation_completed=True,gpu_execution_tested=False)
    (work/'setup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    subprocess.run([str(PROJECT/'.venv/train/bin/python'),str(PROJECT/'scripts/doctor.py'),
        '--gpu','--precision',args.precision],check=True)
    print('Isolated notebook environments and GPU checks completed. Training has not started.',flush=True)


if __name__=='__main__':main()
