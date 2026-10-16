"""Install isolated Python 3.11 environments inside a disposable runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]


def installation_fingerprint(precision,prebuilt_kernels):
    settings=dict(python='3.11.14',uv='0.9.5',precision=precision,prebuilt_kernels=prebuilt_kernels)
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode())
    for phase in ['train','infer']:
        for suffix in ['txt','lock.txt']:
            digest.update((PROJECT/f'requirements/{phase}.{suffix}').read_bytes())
    return digest.hexdigest()


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
    receipt_path=work/'setup_receipt.json'
    fingerprint=installation_fingerprint(args.precision,args.prebuilt_kernels)
    try:previous=json.loads(receipt_path.read_text())
    except (OSError,ValueError):previous={}
    reusable=(previous.get('fingerprint')==fingerprint and previous.get('installation_completed')
        and all((work/f'envs/{phase}/bin/python').exists() for phase in ['train','infer']))
    if not reusable and shutil.disk_usage(work).free<25*1024**3:
        raise RuntimeError('Need at least 25 GiB free disposable disk for environments, model and checkpoints.')
    tools=work/'tools';tools.mkdir(exist_ok=True)
    uv=tools/'bin/uv'
    if not reusable:
        print('[dependencies] Install/check uv package manager',flush=True)
        if not uv.exists() or subprocess.check_output([str(uv),'--version'],text=True).strip()!='uv 0.9.5':
            subprocess.run([sys.executable,'-m','pip','install','--upgrade','--target',str(tools),'uv==0.9.5'],check=True)
    env=os.environ.copy();env.update(UV_PYTHON_INSTALL_DIR=str(work/'python'),
        UV_CACHE_DIR=str(work/'uv-cache'),UV_LINK_MODE='hardlink',HF_HUB_DISABLE_XET='1')
    def run(*arguments):subprocess.run([str(uv),*arguments],check=True,env=env)
    if not reusable:
        print('[dependencies] Install/check Python 3.11.14',flush=True)
        run('python','install','3.11.14')
    else:print('[dependencies] Reusing completed installation; GPU checks will run again.',flush=True)
    venv_root=work/'envs';(PROJECT/'.venv').mkdir(exist_ok=True)
    for phase in ['train','infer']:
        venv=venv_root/phase
        if not venv.exists():run('venv','--python','3.11.14','--seed',str(venv))
        link=PROJECT/'.venv'/phase
        if link.is_symlink() and link.resolve()!=venv:link.unlink()
        if not link.exists():link.symlink_to(venv,target_is_directory=True)
        if link.resolve()!=venv:raise RuntimeError(f'{link} exists outside this bootstrap; choose a clean checkout.')
        python=str(venv/'bin/python')
        if reusable:continue
        torch=['torch==2.9.1'] if phase=='train' else ['torch==2.9.0','torchvision==0.24.0','torchaudio==2.9.0']
        print(f'[dependencies] {phase}: install/check CUDA Torch wheels',flush=True)
        run('pip','install','--python',python,*torch,'--index-url','https://download.pytorch.org/whl/cu128')
        print(f'[dependencies] {phase}: install/check pinned dependencies',flush=True)
        run('pip','install','--python',python,'-r',str(PROJECT/f'requirements/{phase}.lock.txt'))
        run('pip','check','--python',python)
    if not reusable and args.prebuilt_kernels and args.precision=='bfloat16':
        print('[dependencies] Inference: install/check prebuilt FlashInfer kernels (about 1 GB cache wheel)',flush=True)
        python=str(venv_root/'infer/bin/python')
        run('pip','install','--python',python,'flashinfer-cubin==0.5.2')
        run('pip','install','--python',python,'flashinfer-jit-cache==0.5.2',
            '--index-url','https://flashinfer.ai/whl/cu128')
    receipt=dict(fingerprint=fingerprint,python='3.11.14',precision=args.precision,work_root=str(work),
        prebuilt_kernels=args.prebuilt_kernels,installation_completed=True,gpu_execution_tested=False)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print('[dependencies] Validate packages and execute GPU kernel checks',flush=True)
    subprocess.run([str(PROJECT/'.venv/train/bin/python'),str(PROJECT/'scripts/doctor.py'),
        '--gpu','--precision',args.precision],check=True)
    receipt['gpu_execution_tested']=True
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print('Isolated notebook environments and GPU checks completed. Training has not started.',flush=True)


if __name__=='__main__':main()
