"""Pack/restore the trusted migration checkpoint using checksummed LFS parts."""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / 'artifacts'
SNAPSHOT = PROJECT / 'studies/rlvr_success_transfer/reproduction'
CHUNK_BYTES = 1024 ** 3


def download(repository='tuananhlevan/rlvr_success_transfer'):
    """Fetch existing LFS objects over HTTPS without installing Git LFS."""
    if len(repository.split('/'))!=2 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./' for c in repository):
        raise ValueError('Expected repository in owner/name form')
    manifest=read_json(ARTIFACTS/'manifest.json')
    token=os.environ.get('RLVR_GITHUB_TOKEN') or os.environ.get('GITHUB_TOKEN')
    headers={'Content-Type':'application/vnd.git-lfs+json','Accept':'application/vnd.git-lfs+json'}
    if token:headers['Authorization']='Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode()
    for part in manifest['parts']:
        path=ARTIFACTS/part['name']
        if path.exists() and path.stat().st_size==part['bytes'] and digest(path)==part['sha256']:
            print(f'Reusing verified {part["name"]}',flush=True);continue
        payload=dict(operation='download',transfers=['basic'],objects=[dict(oid=part['sha256'],size=part['bytes'])])
        req=urllib.request.Request(f'https://github.com/{repository}.git/info/lfs/objects/batch',
            data=json.dumps(payload).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=60) as response:record=json.load(response)['objects'][0]
        if record.get('error'):raise RuntimeError(f'LFS object unavailable ({part["name"]}); check repository access and LFS quota')
        action=record['actions']['download'];temporary=path.with_suffix(path.suffix+'.download')
        offset=temporary.stat().st_size if temporary.exists() else 0
        if offset>=part['bytes']:temporary.unlink();offset=0
        download_headers=dict(action.get('header',{}))
        if offset:download_headers['Range']=f'bytes={offset}-'
        request=urllib.request.Request(action['href'],headers=download_headers)
        with urllib.request.urlopen(request,timeout=120) as response:
            mode='ab' if offset and response.status==206 else 'wb'
            done=offset if mode=='ab' else 0
            start=time.monotonic();last=start;initial=done
            print(f'Downloading {part["name"]}: {done/1024**2:.0f}/{part["bytes"]/1024**2:.0f} MiB',flush=True)
            with temporary.open(mode) as output:
                while block:=response.read(8*1024**2):
                    output.write(block);done+=len(block);now=time.monotonic()
                    if now-last>=10:
                        print(f'{part["name"]}: {done/1024**2:.0f}/{part["bytes"]/1024**2:.0f} MiB, {(done-initial)/1024**2/(now-start):.1f} MiB/s',flush=True)
                        last=now
        print(f'Verifying {part["name"]}',flush=True)
        if temporary.stat().st_size!=part['bytes'] or digest(temporary)!=part['sha256']:
            raise ValueError(f'Incomplete/corrupt download: {part["name"]}; rerun to resume it')
        temporary.replace(path)
        print(f'Downloaded and verified {part["name"]}',flush=True)
    verify(manifest)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def pack(checkpoint,buffer,request):
    checkpoint=Path(checkpoint).resolve();ARTIFACTS.mkdir(exist_ok=True)
    manifest_path=ARTIFACTS/'manifest.json'
    if manifest_path.exists():raise FileExistsError('Artifact manifest already exists; refusing to replace it.')
    commit=read_json(checkpoint/'commit.json')
    manifest=dict(format_version=1,seed=1,committed_state=commit,
        trust='Only restore artifacts from a repository and commit you trust. Optimizer loading uses torch pickle.',
        checkpoint_files={},parts=[],archive_root='latest_checkpoint')
    for p in sorted(checkpoint.rglob('*')):
        if p.is_file():manifest['checkpoint_files'][str(p.relative_to(checkpoint))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    archive=ARTIFACTS/'.checkpoint.tar.gz'
    try:
        with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(checkpoint,arcname='latest_checkpoint')
        manifest['archive_sha256']=digest(archive)
        with archive.open('rb') as f:
            index=0
            while block:=f.read(CHUNK_BYTES):
                name=f'checkpoint_step_{commit["step"]:04d}.tar.gz.part{index:03d}'
                p=ARTIFACTS/name;p.write_bytes(block)
                manifest['parts'].append(dict(name=name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
                print(f'Packed {name}: {len(block)} bytes',flush=True);index+=1
    finally:archive.unlink(missing_ok=True)
    if read_json(checkpoint/'commit.json')!=commit:raise RuntimeError('Checkpoint changed during packing')
    req=read_json(request)
    assert req['next_state']['step']==commit['step']+1
    pending=ARTIFACTS/'pending_update_buffer.json.gz'
    with Path(buffer).open('rb') as src,gzip.open(pending,'wb',compresslevel=1) as out:shutil.copyfileobj(src,out)
    manifest['pending_update']=dict(name=pending.name,bytes=pending.stat().st_size,
        sha256=digest(pending),buffer_sha256=digest(buffer),seed=req['seed'],
        next_state=req['next_state'],prepared_from_step=commit['step'])
    write_json(manifest_path,manifest)
    print('Artifact manifest written; model, AdamW state and pending step preserved.',flush=True)


class PartReader:
    def __init__(self,paths):self.paths=iter(paths);self.file=None
    def read(self,n=-1):
        if n<0:raise ValueError('Streaming reader requires a bounded read')
        out=[]
        while n:
            if self.file is None:
                try:self.file=next(self.paths).open('rb')
                except StopIteration:break
            b=self.file.read(n)
            if not b:self.file.close();self.file=None;continue
            out.append(b);n-=len(b)
        return b''.join(out)
    def close(self):
        if self.file:self.file.close()


def verify(manifest=None):
    manifest=manifest or read_json(ARTIFACTS/'manifest.json')
    h=hashlib.sha256()
    for part in manifest['parts']:
        p=ARTIFACTS/part['name']
        if not p.exists() or p.stat().st_size!=part['bytes'] or digest(p)!=part['sha256']:
            raise ValueError(f'Missing/corrupt LFS artifact: {p}. Run git lfs pull.')
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    if h.hexdigest()!=manifest['archive_sha256']:raise ValueError('Combined archive hash mismatch')
    pending=manifest['pending_update'];p=ARTIFACTS/pending['name']
    if not p.exists() or digest(p)!=pending['sha256']:raise ValueError('Pending update hash mismatch')
    print('All artifact parts and pending buffer verified.',flush=True)
    return manifest


def restore(run_dir):
    run_dir=Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} is not empty. Use a new run directory to protect existing progress.')
    manifest=verify();run_dir.mkdir(parents=True,exist_ok=True)
    seed_dir=run_dir/f'seed_{manifest["seed"]}';seed_dir.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.restore-',dir=run_dir))
    reader=PartReader([ARTIFACTS / p['name'] for p in manifest['parts']])
    try:
        with tarfile.open(fileobj=reader,mode='r|gz') as archive:
            for member in archive:
                target=(stage/member.name).resolve()
                if not target.is_relative_to(stage.resolve()):raise ValueError('Unsafe archive path')
                if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                if not member.isfile():raise ValueError('Only regular files and directories are accepted')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
        checkpoint=stage/'latest_checkpoint'
        for name,record in manifest['checkpoint_files'].items():
            p=checkpoint/name
            if p.stat().st_size!=record['bytes'] or digest(p)!=record['sha256']:
                raise ValueError(f'Restored checkpoint hash mismatch: {name}')
        checkpoint.rename(seed_dir/'latest_checkpoint')
    finally:reader.close();shutil.rmtree(stage)
    # Historical results stay immutable in studies/. A continuation gets its
    # own copies of committed curves and exact input population.
    for name in ['inputs.json','protocol.json']:
        shutil.copyfile(SNAPSHOT/name,run_dir/name)
    for name in ['state.json','optimization_curve.jsonl','prompt_reward_stats.csv',
                 'sampling_journal.json','validation_curve.jsonl','validation_evaluation.json']:
        p=SNAPSHOT/f'seed_{manifest["seed"]}'/name
        if p.exists():shutil.copyfile(p,seed_dir/name)
    pending=manifest['pending_update'];folder=seed_dir/'pending_update';folder.mkdir()
    buffer=folder/'buffer.json'
    with gzip.open(ARTIFACTS/pending['name'],'rb') as src,buffer.open('wb') as out:shutil.copyfileobj(src,out)
    if digest(buffer)!=pending['buffer_sha256']:raise ValueError('Decompressed pending buffer hash mismatch')
    write_json(folder/'request.json',dict(seed=pending['seed'],next_state=pending['next_state'],
        prepared_from_step=pending['prepared_from_step'],buffer_sha256=pending['buffer_sha256']))
    write_json(run_dir/'status.json',dict(status='RESTORED_NOT_RUNNING',committed_step=manifest['committed_state']['step'],
        pending_step=pending['next_state']['step'],source_manifest_sha256=digest(ARTIFACTS/'manifest.json')))
    print(f'Restored step {manifest["committed_state"]["step"]}; step {pending["next_state"]["step"]} rollouts ready in {run_dir}',flush=True)


def main():
    global ARTIFACTS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,default=ARTIFACTS)
    sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('pack');a.add_argument('--checkpoint',required=True);a.add_argument('--buffer',required=True);a.add_argument('--request',required=True)
    sub.add_parser('verify');a=sub.add_parser('restore');a.add_argument('--run-dir',default=str(PROJECT/'runs/reproduction'))
    a=sub.add_parser('download');a.add_argument('--repository',default='tuananhlevan/rlvr_success_transfer')
    args=p.parse_args();ARTIFACTS=args.artifact_dir.resolve()
    if args.mode=='pack':pack(args.checkpoint,args.buffer,args.request)
    elif args.mode=='download':download(args.repository)
    elif args.mode=='verify':verify()
    else:restore(args.run_dir)


if __name__=='__main__':main()
