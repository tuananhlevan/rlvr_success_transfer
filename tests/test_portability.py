"""Migration and transaction tests using small fixtures; no GPU required."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,PROJECT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


artifacts=load('artifacts','scripts/artifacts.py')
driver=load('driver','studies/rlvr_success_transfer/reproduce_failure.py')


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.store=self.root/'artifacts';self.snapshot=self.root/'snapshot'
        self.patches=patch.multiple(artifacts,ARTIFACTS=self.store,SNAPSHOT=self.snapshot,CHUNK_BYTES=128)
        self.patches.start();self.addCleanup(self.patches.stop)
        self.commit=dict(step=1,epoch=0,cursor=3072,generation_batches=3)
        checkpoint=self.root/'checkpoint'
        write(checkpoint/'commit.json',self.commit)
        write(checkpoint/'carry.json',[{'id':4}])
        (checkpoint/'model.safetensors').write_bytes(bytes(range(256))*4)
        write(self.root/'buffer.json',{'groups':[{'id':9}]})
        write(self.root/'request.json',dict(seed=2,next_state=dict(self.commit,step=2,cursor=5120,generation_batches=5)))
        for name in ['inputs.json','protocol.json']:write(self.snapshot/name,{'test':True})
        write(self.snapshot/'seed_1/state.json',self.commit)
        with redirect_stdout(io.StringIO()):artifacts.pack(checkpoint,self.root/'buffer.json',self.root/'request.json')

    def test_restore_checks_hashes_and_rebases_pending(self):
        run=self.root/'another-machine'/'run'
        with redirect_stdout(io.StringIO()):artifacts.restore(run)
        self.assertEqual(json.loads((run/'seed_1/latest_checkpoint/commit.json').read_text()),self.commit)
        metadata=json.loads((run/'seed_1/pending_update/request.json').read_text())
        self.assertEqual(metadata['next_state']['step'],2)
        self.assertNotIn('model_path',metadata)
        self.assertEqual((run/'seed_1/pending_update/buffer.json').read_bytes(),(self.root/'buffer.json').read_bytes())
        with self.assertRaises(FileExistsError):artifacts.restore(run)

    def test_corrupted_or_pointer_artifact_is_rejected(self):
        manifest=artifacts.read_json(self.store/'manifest.json')
        part=self.store/manifest['parts'][0]['name']
        part.write_text('version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError,'Missing/corrupt LFS'):artifacts.verify()

    def test_path_traversal_rejected_even_with_matching_hash(self):
        manifest=artifacts.read_json(self.store/'manifest.json')
        part=self.store/'evil.part'
        with tarfile.open(part,'w:gz') as t:
            member=tarfile.TarInfo('../outside');member.size=1
            t.addfile(member,io.BytesIO(b'x'))
        manifest['parts']=[dict(name=part.name,bytes=part.stat().st_size,sha256=artifacts.digest(part))]
        manifest['archive_sha256']=artifacts.digest(part)
        artifacts.write_json(self.store/'manifest.json',manifest)
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'Unsafe archive path'):
            artifacts.restore(self.root/'run')
        self.assertFalse((self.root/'run/outside').exists())


class PendingUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.seed=Path(self.temp.name)/'rebased-run/seed_1'
        self.seed.mkdir(parents=True)
        self.old=dict(step=1,epoch=0,cursor=3072,generation_batches=3)
        self.sampled=dict(self.old,cursor=5120,generation_batches=5)
        self.target=dict(self.sampled,step=2)
        write(self.seed/'latest_checkpoint/commit.json',self.old)
        write(self.seed/'latest_checkpoint/carry.json',[])
        write(self.seed/'state.json',self.old)
        driver.persist_update(self.seed,self.sampled,[{'id':9}],2)

    def worker(self,python,args,phase,timeout):
        req=json.loads(Path(args[2]).read_text())
        checkpoint=Path(req['checkpoint_path'])
        self.assertEqual(checkpoint,self.seed/'latest_checkpoint')
        self.assertEqual(Path(req['optimizer_path']),checkpoint/'optimizer.pt')
        self.assertEqual(Path(req['buffer_path']),self.seed/'pending_update/buffer.json')
        self.assertEqual(req['next_state'],self.target)
        write(checkpoint/'commit.json',req['next_state'])
        write(checkpoint/'carry.json',[{'id':77}])
        write(checkpoint/'update_receipt.json',dict(updates=[{'loss':1}],elapsed_seconds=1))

    def test_pending_update_is_rebased_and_committed_once(self):
        with patch.object(driver,'subprocess_phase',side_effect=self.worker) as worker:
            state,carry=driver.finish_pending_update(self.seed,self.old)
        self.assertEqual(worker.call_count,1)
        self.assertEqual(state,self.target);self.assertEqual(carry,[{'id':77}])
        self.assertFalse((self.seed/'pending_update').exists())
        driver.append_update_receipt(self.seed,2,{'updates':[]})
        self.assertEqual(len((self.seed/'optimization_curve.jsonl').read_text().splitlines()),1)

    def test_failed_worker_preserves_original_checkpoint_and_buffer(self):
        before=(self.seed/'pending_update/buffer.json').read_bytes()
        with patch.object(driver,'subprocess_phase',side_effect=RuntimeError('CUDA unavailable')):
            with self.assertRaises(RuntimeError):driver.finish_pending_update(self.seed,self.old)
        self.assertEqual((self.seed/'pending_update/buffer.json').read_bytes(),before)
        self.assertEqual(driver.read_json(self.seed/'latest_checkpoint/commit.json'),self.old)

    def test_commit_before_driver_crash_does_not_retrain(self):
        write(self.seed/'latest_checkpoint/commit.json',self.target)
        write(self.seed/'latest_checkpoint/update_receipt.json',{'updates':[]})
        with patch.object(driver,'subprocess_phase') as worker:
            state,_=driver.finish_pending_update(self.seed,self.target)
        worker.assert_not_called()
        self.assertEqual(driver.read_json(self.seed/'state.json'),self.target)
        self.assertEqual(state,self.target)
        self.assertEqual(len((self.seed/'optimization_curve.jsonl').read_text().splitlines()),1)

    def test_bad_buffer_does_not_start_worker(self):
        (self.seed/'pending_update/buffer.json').write_text('{}')
        with patch.object(driver,'subprocess_phase') as worker:
            with self.assertRaisesRegex(ValueError,'hash mismatch'):driver.finish_pending_update(self.seed,self.old)
        worker.assert_not_called()

    def test_replay_removes_only_uncommitted_csv_rows(self):
        path=self.seed/'prompt_reward_stats.csv'
        path.write_text('step,prompt_id\n1,10\n2,11\n2,12\n')
        driver.rollback_uncommitted_stats(self.seed,1)
        self.assertEqual(path.read_text().splitlines(),['step,prompt_id','1,10'])


if __name__=='__main__':unittest.main()
