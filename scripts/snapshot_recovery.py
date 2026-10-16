"""Repair only byte-identical export objects; retain saved progress and manifests."""
import json
from pathlib import Path
import shutil
import tempfile

from session_store import sha


def recovery_plan(issues,artifact_manifest):
    if not issues:raise ValueError('Snapshot failure has no recoverable blob diagnostics.')
    manifest=json.loads(Path(artifact_manifest).read_text())
    known={record['sha256']:(Path(f'seed_{manifest["seed"]}/latest_checkpoint')/name,record['bytes'])
        for name,record in manifest['checkpoint_files'].items()}
    pending=manifest['pending_update']
    known[pending['buffer_sha256']]=(Path(f'seed_{manifest["seed"]}/pending_update/buffer.json'),None)
    plan={}
    for issue in issues:
        match=known.get(issue['sha256'])
        if not match or (match[1] is not None and match[1]!=issue['bytes']):
            raise ValueError(f'Cannot recover {issue["name"]} from the exported checkpoint: its exact hash is unavailable. Saved progress has been retained; no fallback to step 1 was performed.')
        relative,size=match
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe export path')
        plan[issue['sha256']]=dict(source=relative,bytes=issue['bytes'],name=issue['name'])
    return plan


def repair_blobs(vault,export,plan,emit=print):
    blobs=Path(vault)/'blobs';blobs.mkdir(parents=True,exist_ok=True)
    for digest,record in plan.items():
        source=Path(export)/record['source']
        if source.stat().st_size!=record['bytes'] or sha(source)!=digest:
            raise ValueError(f'Recovery source hash mismatch: {record["name"]}')
        emit(f'[recovery] Restoring verified blob for {record["name"]} ({record["bytes"]/1024**2:.0f} MiB)')
        temporary=None
        try:
            # A unique temporary name avoids accepting a previous partial copy.
            descriptor=tempfile.NamedTemporaryFile(prefix='.repair-',dir=blobs,delete=False)
            temporary=Path(descriptor.name);descriptor.close()
            shutil.copyfile(source,temporary)
            if temporary.stat().st_size!=record['bytes'] or sha(temporary)!=digest:
                raise ValueError(f'Recovered blob verification failed: {record["name"]}')
            temporary.replace(blobs/digest)
        finally:
            if temporary:temporary.unlink(missing_ok=True)
    emit('[recovery] Missing export blobs repaired; saved manifests and later training state were preserved.')
