"""CPU tests for notebook snapshots, interrupted generation and FP16 scaling."""
import gzip
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_portability import driver,load,write
store=load('session_store_test','scripts/session_store.py')
artifacts=load('cloud_artifacts_test','scripts/artifacts.py')


class DownloadAndNotebookTests(unittest.TestCase):
    def test_lfs_range_resume_does_not_forward_repo_credentials_to_object_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=b'checkpoint-object';pending=b'pending'
            digest=hashlib.sha256(data).hexdigest()
            manifest=dict(parts=[dict(name='part000',bytes=len(data),sha256=digest)],archive_sha256=digest,
                pending_update=dict(name='pending.gz',sha256=hashlib.sha256(pending).hexdigest()))
            write(root/'manifest.json',manifest);(root/'pending.gz').write_bytes(pending)
            (root/'part000.download').write_bytes(data[:4]);requests=[]
            def open_url(request,timeout):
                requests.append(request)
                if request.full_url.endswith('/objects/batch'):
                    return io.BytesIO(json.dumps(dict(objects=[dict(actions=dict(download=dict(href='https://object.invalid/blob')))])).encode())
                response=io.BytesIO(data[4:]);response.status=206;return response
            with patch.object(artifacts,'ARTIFACTS',root),patch.object(artifacts.urllib.request,'urlopen',side_effect=open_url),patch.dict('os.environ',{'RLVR_GITHUB_TOKEN':'dummy-token'}):
                artifacts.download()
            self.assertEqual((root/'part000').read_bytes(),data)
            self.assertIsNotNone(requests[0].get_header('Authorization'))
            self.assertIsNone(requests[1].get_header('Authorization'))
            self.assertEqual(requests[1].get_header('Range'),'bytes=4-')

    def test_notebook_cells_compile_and_have_no_stored_outputs(self):
        project=Path(__file__).resolve().parents[1]
        for path in (project/'notebooks').glob('*.ipynb'):
            notebook=json.loads(path.read_text());self.assertEqual(notebook['nbformat'],4)
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    compile(''.join(cell['source']),str(path),'exec')
                    self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'run';self.vault=self.root/'vault'
        write(self.run/'inputs.json',{'population':[1,2]})
        write(self.run/'seed_1/latest_checkpoint/commit.json',{'step':1})
        write(self.run/'seed_1/sampling_progress.json',{'state':{'cursor':5120},'groups':[1]})
        self.model=self.run/'seed_1/latest_checkpoint/model.bin';self.model.write_bytes(b'weights-v1')
        (self.run/'worker.log').write_text('volatile log')

    def test_restore_across_hosts_and_blob_reuse(self):
        manifest=store.publish(self.run,self.vault)
        before=sorted(p.name for p in (self.vault/'blobs').iterdir())
        store.publish(self.run,self.vault)
        self.assertEqual(before,sorted(p.name for p in (self.vault/'blobs').iterdir()))
        destination=self.root/'new-host/run';store.restore(self.vault,destination)
        self.assertEqual((destination/'seed_1/latest_checkpoint/model.bin').read_bytes(),b'weights-v1')
        self.assertEqual(json.loads((destination/'seed_1/sampling_progress.json').read_text())['state']['cursor'],5120)
        self.assertFalse((destination/'worker.log').exists())
        with self.assertRaises(FileExistsError):store.restore(self.vault,destination)

    def test_interrupted_publish_keeps_last_complete_snapshot(self):
        store.publish(self.run,self.vault);pointer=(self.vault/'CURRENT.json').read_bytes()
        self.model.write_bytes(b'new-incomplete-weights')
        with patch.object(store.shutil,'copyfile',side_effect=OSError('quota exceeded')):
            with self.assertRaises(OSError):store.publish(self.run,self.vault)
        self.assertEqual((self.vault/'CURRENT.json').read_bytes(),pointer)
        store.restore(self.vault,self.root/'recovery')
        self.assertEqual((self.root/'recovery/seed_1/latest_checkpoint/model.bin').read_bytes(),b'weights-v1')

    def test_new_snapshot_prunes_old_checkpoint_and_corruption_rejected(self):
        old=store.publish(self.run,self.vault)['files']['seed_1/latest_checkpoint/model.bin']['sha256']
        self.model.write_bytes(b'weights-v2')
        new=store.publish(self.run,self.vault)['files']['seed_1/latest_checkpoint/model.bin']['sha256']
        self.assertFalse((self.vault/'blobs'/old).exists())
        (self.vault/'blobs'/new).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'No complete snapshot'):store.restore(self.vault,self.root/'recovery')
        self.assertFalse((self.root/'recovery').exists())


class GenerationResumeTests(unittest.TestCase):
    def test_compressed_chunks_resume_on_a_different_host_without_repeating_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'host-a';output=run/'scratch/final_evaluation.json'
            write(run/'inputs.json',{})
            output.parent.mkdir(parents=True)
            rows=[dict(id=i,prompt_tokens=[i],answer='2') for i in range(5)]
            request=root/'request.json';write(request,dict(model_path=str(root/'model-a'),rows=rows,n=1,
                temperature=.7,seed=123,prompt_key='prompt_tokens'))
            generated=[]
            class FakeLLM:
                def __init__(self,**kwargs):pass
                def generate(self,prompts,params,**kwargs):
                    generated.extend(p['prompt_token_ids'][0] for p in prompts)
                    return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2],finish_reason='stop',stop_reason=0)]) for p in prompts]
            class FakePool:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def map(self,function,items):return list(map(function,items))
            modules={'vllm':SimpleNamespace(LLM=FakeLLM,SamplingParams=lambda **kw:SimpleNamespace(**kw)),
                'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _:SimpleNamespace(eos_token_id=0,decode=lambda *a,**k:'2')))}
            with patch.dict(sys.modules,modules),patch('multiprocessing.get_context',return_value=SimpleNamespace(Pool=lambda _:FakePool())),patch.object(driver,'base_path',return_value='base'),patch.object(driver,'grade_one',return_value=dict(reward=1,evaluation_reward=1)),patch.object(driver,'backup_session'),patch.dict(driver.CONFIG,{'generation_chunk_responses':2,'grading_workers':1}):
                with patch.object(driver,'check_session_margin',side_effect=[None,driver.SessionPause('slice done')]):
                    with self.assertRaises(driver.SessionPause):driver.generation_worker(request,output)
                self.assertEqual(generated,[0,1])
                store.publish(run,root/'vault');new_run=root/'host-b'
                store.restore(root/'vault',new_run)
                new_output=new_run/'scratch/final_evaluation.json'
                req=json.loads(request.read_text());req['model_path']=str(root/'different-cache/model-a');write(request,req)
                with patch.object(driver,'check_session_margin'):
                    driver.generation_worker(request,new_output)
                self.assertEqual(generated,[0,1,2,3,4])
                receipt=driver.read_json(new_output)
                self.assertEqual([g['id'] for g in driver.iter_groups(receipt)],list(range(5)))
                with patch.object(driver,'RUN',new_run),patch.object(driver,'SCRATCH',new_run/'scratch'),patch.object(driver,'generate',return_value=receipt):
                    driver.evaluate(rows,'model',1,1,'final',n=1)
                archive=new_run/'seed_1/final_evaluation.rollouts.jsonl.gz'
                with gzip.open(archive,'rt') as stream:self.assertEqual([json.loads(s)['id'] for s in stream],list(range(5)))
                self.assertFalse(new_output.exists())

    def test_session_pause_is_not_a_completed_result(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(sys,'argv',['reproduce_failure.py','run','--run-dir',temporary]),patch.object(driver,'driver',side_effect=driver.SessionPause('budget')),patch.object(driver,'backup_session'),patch.dict(driver.CONFIG,dict(driver.CONFIG)):
            driver.main()
            self.assertEqual(json.loads((Path(temporary)/'status.json').read_text())['status'],'PAUSED')


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required for the FP16 arithmetic checks')
class ScalingTests(unittest.TestCase):
    def test_fp16_split_backward_scales_once_and_preserves_tied_fp32_master(self):
        import torch
        from torch import nn
        class Tied(nn.Module):
            def __init__(self):
                super().__init__();self.hidden=nn.Linear(4,4,bias=False);self.head=nn.Linear(4,4,bias=False)
                self.head.weight=self.hidden.weight
        torch.manual_seed(42);model=Tied();x=torch.randn(6,4).half()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        with patch.dict(driver.CONFIG,{'precision':'float16'}):
            with driver.source_precision(model):
                hidden=model.hidden(x);loss=model.head(hidden).float().square().sum()/24
                loss.backward()
            expected=model.hidden.weight.grad.clone();optimizer.zero_grad(set_to_none=True)
            with driver.source_precision(model):
                hidden=model.hidden(x);proxy=hidden.detach().requires_grad_(True)
                for start in range(0,6,2):
                    loss=model.head(proxy[start:start+2]).float().square().sum()/24
                    scaler.scale(loss).backward()
                # proxy.grad already carries the scale: never scale this again.
                hidden.backward(proxy.grad)
            scaler.unscale_(optimizer)
            torch.testing.assert_close(model.hidden.weight.grad,expected,rtol=.01,atol=.0005)
            self.assertEqual(model.hidden.weight.dtype,torch.float32)
            scaler.step(optimizer);scaler.update()
            state=optimizer.state[model.hidden.weight]
            self.assertEqual(int(state['step']),1)
            self.assertEqual(state['exp_avg'].dtype,torch.float32)

    def test_overflow_does_not_consume_an_adam_step(self):
        import torch
        model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        scaler.scale(model(torch.ones(1,2)).sum()).backward()
        for parameter in model.parameters():parameter.grad.fill_(float('inf'))
        _,applied=driver.finish_scaled_update(model,optimizer,scaler)
        self.assertFalse(applied);self.assertFalse(optimizer.state)
        self.assertEqual(scaler.get_scale(),64.)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.ones(1,2)).square().sum()).backward()
        _,applied=driver.finish_scaled_update(model,optimizer,scaler)
        self.assertTrue(applied)
        self.assertTrue(all(int(state['step'])==1 for state in optimizer.state.values()))


if __name__=='__main__':unittest.main()
