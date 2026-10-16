"""Verified snapshots for ephemeral notebook runtimes (standard library only).

One manifest references immutable SHA-256 blobs. A failed publish retains the
previous manifest; restore verifies bytes before accepting a snapshot. Use one
writer per vault. Mounted Drive does not provide local-filesystem guarantees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


class SnapshotRestoreError(ValueError):
    def __init__(self,errors,blob_issues):
        super().__init__('No complete snapshot found: '+'; '.join(errors))
        self.blob_issues=blob_issues


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,sort_keys=True)+'\n');temp.replace(path)


def safe_path(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError(f'Unsafe snapshot path: {name}')
    return path


def selected_files(run):
    skipped={'worker.log','live.log','driver.log','status.json','driver_process.json',
        'generation_request.json','runtime_request.json','.snapshot_cache.json'}
    for path in sorted(run.rglob('*')):
        relative=path.relative_to(run)
        if any(part.startswith('.') for part in relative.parts):continue
        if path.is_symlink():raise ValueError(f'Snapshot symlink refused: {relative}')
        if not path.is_file() or path.name in skipped or path.name.endswith('.tmp'):continue
        if 'pending_update' in relative.parts and path.name not in {'request.json','buffer.json'}:continue
        yield path,relative.as_posix()


def publish(run,vault,progress=False):
    run=Path(run).resolve();vault=Path(vault).resolve()
    if vault.is_relative_to(run) or run.is_relative_to(vault):raise ValueError('Run and vault must be separate directories')
    if not (run/'inputs.json').exists():return None
    blobs=vault/'blobs';blobs.mkdir(parents=True,exist_ok=True)
    manifests=vault/'manifests';manifests.mkdir(exist_ok=True)
    cache_path=run/'.snapshot_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    records={};new_cache={}
    for path,name in selected_files(run):
        before=path.stat();signature=[before.st_size,before.st_mtime_ns]
        cached=cache.get(name,{})
        digest=cached.get('sha256') if cached.get('signature')==signature else sha(path)
        blob=blobs/digest
        if not blob.exists() or blob.stat().st_size!=before.st_size:
            if progress:print(f'[snapshot] Copying {name} ({before.st_size/1024**2:.1f} MiB)',flush=True)
            temp=blob.with_suffix('.tmp');shutil.copyfile(path,temp)
            if progress:print(f'[snapshot] Verifying {name}',flush=True)
            if sha(temp)!=digest:raise ValueError(f'Copy hash mismatch: {name}')
            temp.replace(blob)
        after=path.stat()
        if [after.st_size,after.st_mtime_ns]!=signature:raise RuntimeError(f'File changed during snapshot: {name}')
        records[name]=dict(sha256=digest,bytes=before.st_size)
        new_cache[name]=dict(sha256=digest,signature=signature)
    manifest=dict(format=1,created_ns=time.time_ns(),files=records)
    name=f'{manifest["created_ns"]}.json'
    atomic_json(manifests/name,manifest)
    atomic_json(vault/'CURRENT.json',dict(manifest=name,sha256=sha(manifests/name)))
    atomic_json(cache_path,new_cache)
    # Keep the newest valid snapshot; unchanged large files share blob objects.
    for old in manifests.glob('*.json'):
        if old.name!=name:old.unlink()
    retained={record['sha256'] for record in records.values()}
    for blob in blobs.iterdir():
        if blob.name not in retained:blob.unlink()
    return manifest


def candidates(vault):
    pointer=vault/'CURRENT.json';preferred=None
    if pointer.exists():
        try:
            record=json.loads(pointer.read_text());p=safe_path(vault/'manifests',record['manifest'])
            if p.exists() and sha(p)==record['sha256']:preferred=p
        except (ValueError,KeyError,json.JSONDecodeError):pass
    paths=sorted((vault/'manifests').glob('*.json'),reverse=True)
    return ([preferred] if preferred else [])+[p for p in paths if p!=preferred]


def restore(vault,run):
    vault=Path(vault).resolve();run=Path(run).resolve()
    if run.exists() and any(run.iterdir()):raise FileExistsError('Restore requires an empty local run directory')
    run.parent.mkdir(parents=True,exist_ok=True)
    errors=[];first_issues=[]
    for candidate in candidates(vault):
        stage=Path(tempfile.mkdtemp(prefix='rlvr-restore-',dir=run.parent))
        issues=[]
        try:
            manifest=json.loads(candidate.read_text())
            # Check missing objects before spending minutes copying large files.
            for name,record in manifest['files'].items():
                safe_path(stage,name)
                digest=record['sha256']
                if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid blob hash')
                blob=vault/'blobs'/digest
                if not blob.exists() or blob.stat().st_size!=record['bytes']:
                    issues.append(dict(name=name,**record))
            if issues:raise ValueError('Missing or wrong-size snapshot blobs: '+', '.join(r['name'] for r in issues))
            for name,record in manifest['files'].items():
                target=safe_path(stage,name);target.parent.mkdir(parents=True,exist_ok=True)
                digest=record['sha256']
                if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid blob hash')
                blob=vault/'blobs'/digest
                shutil.copyfile(blob,target)
                # Hash the local copy: verifies the bytes actually restored and
                # avoids reading every multi-GB object twice over Drive.
                if target.stat().st_size!=record['bytes'] or sha(target)!=digest:
                    issues.append(dict(name=name,**record))
                    raise ValueError(f'Corrupt snapshot blob: {name}')
            if run.exists():run.rmdir()
            stage.rename(run)
            return manifest
        except (ValueError,KeyError,OSError,json.JSONDecodeError) as error:
            if not errors:first_issues=issues
            errors.append(str(error))
        finally:shutil.rmtree(stage,ignore_errors=True)
    raise SnapshotRestoreError(errors,first_issues)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['publish','restore']);parser.add_argument('--vault',required=True)
    parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    result=publish(args.run_dir,args.vault) if args.mode=='publish' else restore(args.vault,args.run_dir)
    print(json.dumps(dict(files=len(result['files']) if result else 0,mode=args.mode)))


if __name__=='__main__':main()
