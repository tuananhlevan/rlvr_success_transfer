"""Check installation and optional GPU execution without starting training."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]


def source_check():
    root=PROJECT/'resources/rlvr_success_transfer/source'
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError(f'Pinned source changed: {name}')
    print(f'Pinned source: {len(manifest["files"])} hashes verified')


def package_check(requirements):
    for line in requirements.read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,version=line.split('==')
        actual=metadata.version(name).split('+')[0]
        if actual!=version:raise RuntimeError(f'{name}: expected {version}, found {actual}')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)


def gpu_check(infer=False,precision='bfloat16'):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Check driver, CUDA_VISIBLE_DEVICES and container GPU passthrough.')
    if precision=='bfloat16' and torch.cuda.get_device_capability(0)[0]<8:
        raise RuntimeError('Native BF16 needs Ampere or newer. T4 requires --precision float16 (a numerical variant).')
    x=torch.ones((16,16),device='cuda',dtype=getattr(torch,precision))
    assert bool(torch.isfinite(x@x).all())
    print(f'GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, Torch {torch.__version__}')
    if infer:
        from vllm import _custom_ops as ops
        y=torch.empty_like(x)
        ops.rms_norm(y,x,torch.ones(16,device='cuda',dtype=x.dtype),1e-6)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(y).all())
        print('vLLM native CUDA RMSNorm executed')
        if torch.cuda.get_device_capability(0)[0]<8:
            from xformers.ops import memory_efficient_attention
            q=torch.randn((1,16,2,64),device='cuda',dtype=x.dtype)
            z=memory_efficient_attention(q,q,q)
            print('T4 xFormers attention executed; FlashInfer is not used on this path')
        else:
            from flashinfer.norm import rmsnorm
            z=rmsnorm(x,torch.ones(16,device='cuda',dtype=x.dtype),eps=1e-6)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(z).all())
        print('Attention/kernel backend check passed')
    else:
        from torch.nn.utils.stateless import _reparametrize_module
        assert callable(_reparametrize_module)
        print('Training precision adapter API present')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--source-only',action='store_true')
    p.add_argument('--precision',choices=['bfloat16','float16'],default='bfloat16')
    p.add_argument('--infer-worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    source_check()
    if args.source_only:return
    phase='infer' if args.infer_worker else 'train'
    package_check(PROJECT/f'requirements/{phase}.txt')
    if args.gpu:gpu_check(args.infer_worker,args.precision)
    if not args.infer_worker:
        command=[str(PROJECT/'.venv/infer/bin/python'),str(Path(__file__).resolve()),'--infer-worker','--precision',args.precision]
        if args.gpu:command.append('--gpu')
        subprocess.run(command,check=True)
    print(f'{phase} checks passed; no model training started')


if __name__=='__main__':main()
