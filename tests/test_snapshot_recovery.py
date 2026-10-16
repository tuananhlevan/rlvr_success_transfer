"""Recover a missing export blob while preserving the saved experiment."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cloud_cli as cloud
import session_store as store
import snapshot_recovery as recovery


class SnapshotRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'runtime/run';self.vault=self.root/'vault'
        checkpoint=self.run/'seed_1/latest_checkpoint';checkpoint.mkdir(parents=True)
        (checkpoint/'optimizer.pt').write_bytes(b'exported optimizer')
        (checkpoint/'model.bin').write_bytes(b'saved model')
        (checkpoint/'commit.json').write_text('{"step": 1}')
        (self.run/'inputs.json').write_text('{"exact_population": [1, 2]}')
        (self.run/'seed_1/sampling_progress.json').write_text('{"preserved": "new sampling evidence"}')
        manifest=store.publish(self.run,self.vault)
        self.record=manifest['files']['seed_1/latest_checkpoint/optimizer.pt']
        self.digest=self.record['sha256'];self.blob=self.vault/'blobs'/self.digest
        self.pointer=(self.vault/'CURRENT.json').read_bytes()
        self.artifacts=self.root/'artifacts';self.artifacts.mkdir()
        self.artifact_manifest=self.artifacts/'manifest.json'
        self.artifact_manifest.write_text(json.dumps(dict(seed=1,checkpoint_files={'optimizer.pt':self.record},
            pending_update=dict(buffer_sha256='a'*64))))
        shutil.rmtree(self.run)

    def test_setup_repairs_exact_blob_and_preserves_saved_progress(self):
        self.blob.unlink();commands=[]
        def execute(command,**kwargs):
            commands.append(command)
            if 'restore' in command:
                exported=Path(command[command.index('--run-dir')+1])/'seed_1/latest_checkpoint'
                exported.mkdir(parents=True);(exported/'optimizer.pt').write_bytes(b'exported optimizer')
        config=cloud.make_config('local','bfloat16',1,self.root/'runtime',self.vault)
        config['run_dir']=str(self.run)
        with patch.object(cloud,'PROJECT',self.root),patch.object(cloud.subprocess,'run',side_effect=execute):
            self.assertEqual(cloud.initialize_data(config),'restored saved session')
        self.assertEqual(len(commands),2)
        self.assertIn('download',commands[0]);self.assertIn('restore',commands[1])
        self.assertEqual(self.blob.read_bytes(),b'exported optimizer')
        self.assertEqual((self.vault/'CURRENT.json').read_bytes(),self.pointer)
        self.assertEqual((self.run/'seed_1/sampling_progress.json').read_text(),'{"preserved": "new sampling evidence"}')
        self.assertEqual((self.run/'seed_1/latest_checkpoint/model.bin').read_bytes(),b'saved model')
        self.assertEqual(list(self.run.parent.glob('vault-repair-*')),[])

    def test_unknown_missing_blob_stops_without_reset_or_download(self):
        manifest=store.candidates(self.vault)[0]
        records=json.loads(manifest.read_text())['files']
        (self.vault/'blobs'/records['seed_1/latest_checkpoint/model.bin']['sha256']).unlink()
        config=cloud.make_config('local','bfloat16',1,self.root/'runtime',self.vault);config['run_dir']=str(self.run)
        with patch.object(cloud,'PROJECT',self.root),patch.object(cloud.subprocess,'run') as execute:
            with self.assertRaisesRegex(ValueError,'exact hash is unavailable'):cloud.initialize_data(config)
            execute.assert_not_called()
        self.assertEqual((self.vault/'CURRENT.json').read_bytes(),self.pointer)
        self.assertFalse(self.run.exists())

    def test_missing_blob_is_diagnosed_before_any_large_copy(self):
        self.blob.unlink()
        with patch.object(store.shutil,'copyfile') as copy:
            with self.assertRaises(store.SnapshotRestoreError) as error:store.restore(self.vault,self.run)
            copy.assert_not_called()
        self.assertEqual(error.exception.blob_issues[0]['name'],'seed_1/latest_checkpoint/optimizer.pt')
        self.assertFalse(self.run.exists())

    def test_wrong_recovery_source_cannot_replace_existing_blob(self):
        issue=dict(name='seed_1/latest_checkpoint/optimizer.pt',**self.record)
        plan=recovery.recovery_plan([issue],self.artifact_manifest)
        export=self.root/'export';path=export/plan[self.digest]['source'];path.parent.mkdir(parents=True)
        path.write_bytes(b'invalid optimizer!')
        with self.assertRaisesRegex(ValueError,'source hash mismatch'):recovery.repair_blobs(self.vault,export,plan)
        self.assertEqual(self.blob.read_bytes(),b'exported optimizer')
        self.assertEqual(list((self.vault/'blobs').glob('.repair-*')),[])

    def test_corruption_during_copy_is_detected_before_promotion(self):
        actual_copy=shutil.copyfile
        def corrupt(source,target):
            actual_copy(source,target)
            if str(target).endswith('optimizer.pt'):Path(target).write_bytes(b'x'*self.record['bytes'])
        with patch.object(store.shutil,'copyfile',side_effect=corrupt):
            with self.assertRaisesRegex(store.SnapshotRestoreError,'Corrupt snapshot blob'):store.restore(self.vault,self.run)
        self.assertFalse(self.run.exists())
        self.assertEqual(store.sha(self.blob),self.digest)

    def test_interrupted_repair_leaves_existing_blob_and_manifest_intact(self):
        issue=dict(name='seed_1/latest_checkpoint/optimizer.pt',**self.record)
        plan=recovery.recovery_plan([issue],self.artifact_manifest)
        export=self.root/'export';path=export/plan[self.digest]['source'];path.parent.mkdir(parents=True)
        path.write_bytes(b'exported optimizer')
        def interrupted(source,target):
            Path(target).write_bytes(b'partial upload');raise OSError('Drive write interrupted')
        with patch.object(recovery.shutil,'copyfile',side_effect=interrupted):
            with self.assertRaisesRegex(OSError,'Drive write interrupted'):recovery.repair_blobs(self.vault,export,plan)
        self.assertEqual(self.blob.read_bytes(),b'exported optimizer')
        self.assertEqual((self.vault/'CURRENT.json').read_bytes(),self.pointer)
        self.assertEqual(list((self.vault/'blobs').glob('.repair-*')),[])


if __name__=='__main__':unittest.main()
