"""Source-based, full-data RLVR reproduction on one GPU.

The original four data-parallel ranks are accumulated serially, preserving
their token-weighted microbatch losses. No LoRA, shortened responses, selected
easy training subset, or increased learning rate. See reproduction_report.md.
Workers run in separate processes so inference and Adam state do not compete
for VRAM. Only the latest resumable checkpoint is retained.
"""
import argparse
import collections
from contextlib import contextmanager
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = Path(os.environ.get('RLVR_SOURCE_DIR', PROJECT / 'resources/rlvr_success_transfer/source')).resolve()
RUN = Path(os.environ.get('RLVR_RUN_DIR', PROJECT / 'runs/reproduction')).resolve()
SCRATCH = Path(os.environ.get('RLVR_SCRATCH_DIR', RUN / 'scratch')).resolve()
TRAIN_PYTHON = os.environ.get('RLVR_TRAIN_PYTHON', sys.executable)
INFER_PYTHON = os.environ.get('RLVR_INFER_PYTHON', str(PROJECT / '.venv/infer/bin/python'))
RUN_DEADLINE = None
CONFIG = dict(model="Qwen/Qwen2.5-0.5B",
    model_revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
    dataset="hkust-nlp/SimpleRL-Zoo-Data",
    dataset_revision="50ec8f09045e785fd20c13a508820e77cccb4211",
    source_commit="1dce7fec92e7685929a5efbe73ff2091e7d474a1",
    group_size=8, max_prompt_tokens=1024, max_new_tokens=5120,
    train_batch_prompts=1024, sampling_batch_prompts=1024,
    optimizer_batch_rollouts=512, logical_ranks=4, rank_microbatch=32,
    learning_rate=5e-7, weight_decay=.01, adam_betas=[.9, .999],
    adam_eps=1e-8, grad_clip=1., clip_epsilon=.2, kl_coef=.0001,
    entropy_coef=.001, train_temperature=1., evaluation_temperature=.7,
    validation_temperature=0.,validation_prompt='original dataset prompt',
    evaluation_samples=32, validation_every=5, final_step=120,
    inference_execution='CUDA graphs for decode; no torch.compile',
    training_epochs=50, seeds=[1,42,37], threshold=.1,
    hard_timeout_hours_per_seed=96, initial_evaluation_seed=20261016,
    precision='bfloat16', physical_batch_size=4, generation_chunk_responses=8192,
    max_num_seqs=128, max_num_batched_tokens=8192, grading_workers=8,
    enforce_eager=False, pause_margin_minutes=0)
CONFIG.update(json.loads(os.environ.get('RLVR_EXECUTION_CONFIG','{}')))
PERSIST_DIR=os.environ.get('RLVR_PERSIST_DIR')
SESSION_END_EPOCH=float(os.environ.get('RLVR_SESSION_END_EPOCH','0'))


class SessionPause(Exception):
    """A bounded notebook slice ended; scientific work remains incomplete."""


def compute_dtype():
    import torch
    return getattr(torch,CONFIG['precision'])


def backup_session():
    if PERSIST_DIR:
        sys.path.insert(0,str(PROJECT/'scripts'))
        from session_store import publish
        publish(RUN,PERSIST_DIR)


def check_session_margin(margin=None):
    if SESSION_END_EPOCH and time.time() >= SESSION_END_EPOCH-(CONFIG['pause_margin_minutes']*60 if margin is None else margin):
        raise SessionPause('Notebook time budget reached; resume this same run in another session.')



def log(s):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), s, flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def base_path():
    from huggingface_hub import snapshot_download
    override=os.environ.get('RLVR_MODEL_PATH')
    if override:
        path=Path(override).resolve()
        if not (path/'config.json').exists():raise FileNotFoundError(f'Invalid RLVR_MODEL_PATH: {path}')
        return str(path)
    return snapshot_download(CONFIG['model'], revision=CONFIG['model_revision'],
        local_files_only=os.environ.get('HF_HUB_OFFLINE')=='1',
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE'])


def prepare():
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path())
    splits = {}
    for split in ['train', 'test']:
        path = hf_hub_download(CONFIG['dataset'],
            f'simplelr_qwen_level1to4/{split}.parquet', repo_type='dataset',
            revision=CONFIG['dataset_revision'])
        rows = []
        for row in Dataset.from_parquet(path):
            text = row['prompt'][0]['content']
            tokens = tokenizer.encode(text)
            # Exactly the length filter in the released RLHFDataset.
            if len(tokens) > CONFIG['max_prompt_tokens']:
                continue
            question = row['extra_info']['question']
            # Source inference/config.py, qwen template, then chat template.
            eval_text = tokenizer.apply_chat_template([dict(role='user',
                content=question + '\nPlease reason step by step, and put your final answer within \\boxed{}.')],
                tokenize=False, add_generation_prompt=True)
            rows.append(dict(id=int(row['extra_info']['index']),
                question=question, answer=row['reward_model']['ground_truth'],
                prompt=text, prompt_tokens=tokens,
                eval_prompt_tokens=tokenizer.encode(eval_text),
                level=row['level'], subject=row['subject']))
        splits[split] = rows
        log(f'prepared {split}: {len(rows)} eligible prompts')
    write_json(RUN / 'inputs.json', dict(config=CONFIG, splits=splits))
    released_log_analysis(splits['train'])


def released_log_analysis(rows):
    """Audit source artifacts without treating arbitrary example IDs as labels."""
    d = read_json(SOURCE / 'all_decomposed_data_qwen_0.5b_math_level1to4_unlearnable_filtered.json')
    candidate_ids = {int(x['id'].split('_')[0]) for x in d}
    eligible = {x['id'] for x in rows}
    stats = collections.defaultdict(list)
    with (SOURCE.parent / 'prompt_reward_stats.csv').open() as f:
        for r in csv.DictReader(f):
            stats[int(r['prompt_id'])].append(dict(step=int(r['step']),
                successes=round(float(r['mean_reward'])*int(r['n'])), n=int(r['n'])))
    trajectory = {}
    for label, lo, hi in [('early',1,20),('late',101,120),('final',127,146)]:
        a=[r for i in candidate_ids & eligible for r in stats.get(i,[]) if lo <= r['step'] <= hi]
        trajectory[label] = dict(successes=sum(r['successes'] for r in a),
            samples=sum(r['n'] for r in a), rows=len(a))
    # Candidate labels are not independently verified; report the provenance gap.
    write_json(RUN / 'released_log_audit.json', dict(
        source_sha256=hashlib.sha256((SOURCE.parent/'prompt_reward_stats.csv').read_bytes()).hexdigest(),
        candidate_ids=sorted(candidate_ids), matched_eligible=len(candidate_ids & eligible),
        ids_with_log=len(candidate_ids & stats.keys()), trajectory=trajectory,
        warning='Augmentation parent IDs are candidate cohort labels, not the missing three-seed partition exports. Learnable examples are renumbered; do not join those IDs to the log.'))


def grade_one(args):
    """Released training extractor + Math-Verify; also record inference scorer.

    Source mix reward accepts correctly extracted unboxed answers. The source
    inference scorer instead parses the complete response. Keep both, so their
    disagreement cannot masquerade as learning or failure.
    """
    sys.path.insert(0, str(SOURCE / 'verl/utils/reward_score'))
    from qwen_math_eval_toolkit.parser import extract_answer
    from math_verify import parse, verify
    text, gold = args
    def timed_out(*_):
        raise TimeoutError('answer verifier timeout')
    signal.signal(signal.SIGALRM, timed_out)
    box = lambda x: x if '\\boxed' in x else '\\boxed{' + x + '}'
    result=dict(reward=0,evaluation_reward=0,extracted_answer='')
    signal.alarm(10)
    try:
        import re
        cleaned = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', text, flags=re.DOTALL, count=1)
        for stop in ['</s>', '<|im_end|>', '<|endoftext|>']:
            cleaned = cleaned.split(stop)[0].strip()
        answer = extract_answer(cleaned, data_name='math')
        result['extracted_answer']=answer
        result['reward']=int(bool(verify(gold=parse(box(gold)),target=parse(box(answer)))))
    except Exception as e:
        result['training_grader_error']=type(e).__name__
    finally:
        signal.alarm(0)
    # An evaluation-only exception must not erase a valid training reward.
    signal.alarm(10)
    try:
        result['evaluation_reward']=int(bool(verify(gold=parse(box(gold)),target=parse(box(text)))))
    except Exception as e:
        result['evaluation_grader_error']=type(e).__name__
    finally:
        signal.alarm(0)
    return result


def generation_worker(request_path, output_path):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import multiprocessing as mp
    req = read_json(request_path)
    tok = AutoTokenizer.from_pretrained(base_path())
    t0 = time.monotonic()
    llm = LLM(model=req['model_path'], tokenizer=base_path(), dtype=CONFIG['precision'],
        tensor_parallel_size=1, max_model_len=6144, gpu_memory_utilization=.8,
        max_num_seqs=CONFIG['max_num_seqs'], max_num_batched_tokens=CONFIG['max_num_batched_tokens'], enforce_eager=CONFIG['enforce_eager'],
        compilation_config=dict(mode=0,cudagraph_mode='NONE' if CONFIG['enforce_eager'] else 'FULL_DECODE_ONLY',
            cudagraph_capture_sizes=[1,2,4,8,16,32,64,128]),
        enable_prefix_caching=True, generation_config='vllm', seed=req['seed'],
        disable_log_stats=False)
    engine_seconds = time.monotonic()-t0
    completed=0
    groups_dir=Path(output_path).with_suffix('.groups')
    # Paths change across notebook hosts. Identity is pinned model + committed
    # training state and the sampling request, rather than a machine path.
    identity=dict(req)
    identity.pop('model_path')
    commit=Path(req['model_path'])/'commit.json'
    identity['model_state']=read_json(commit) if commit.exists() else CONFIG['model_revision']
    identity['execution']={k:CONFIG[k] for k in ['precision','generation_chunk_responses','max_num_seqs','enforce_eager']}
    request_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    chunks=[];prior_elapsed=0.;prior_engine=0.
    if Path(output_path).exists():
        prior=read_json(output_path)
        if prior.get('request_sha256')==request_hash and groups_dir.exists():
            completed=prior['completed_prompts'];chunks=prior['chunks']
            prior_elapsed=prior['elapsed_seconds'];prior_engine=prior['engine_seconds']
            for chunk in chunks:
                path=groups_dir/chunk['name']
                if path.stat().st_size!=chunk['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=chunk['sha256']:
                    raise ValueError('Committed generation chunk is missing or corrupt')
    if not completed and groups_dir.exists():shutil.rmtree(groups_dir)
    groups_dir.mkdir(exist_ok=True)
    request_chunk=max(1,CONFIG['generation_chunk_responses']//req['n'])
    receipt=dict(request_sha256=request_hash,groups_dir=str(groups_dir),chunks=chunks,
        completed_prompts=completed,engine_seconds=engine_seconds,elapsed_seconds=0)
    if completed==len(req['rows']):
        write_json(output_path,dict(prior,groups_dir=str(groups_dir)))
        return
    with mp.get_context('spawn').Pool(CONFIG['grading_workers']) as pool:
        for start in range(completed,len(req['rows']),request_chunk):
            check_session_margin(60)
            rows=req['rows'][start:start+request_chunk]
            params=SamplingParams(n=req['n'],temperature=req['temperature'],
                top_p=1.,top_k=-1,max_tokens=5120,seed=req['seed']+start,
                detokenize=True,skip_special_tokens=False)
            t1=time.monotonic()
            generated=llm.generate([dict(prompt_token_ids=r[req['prompt_key']]) for r in rows], params, use_tqdm=True)
            grade_inputs=[]
            groups=[]
            for row, response in zip(rows, generated):
                group=dict(id=row['id'], prompt_tokens=row[req['prompt_key']], rollouts=[])
                for out in response.outputs:
                    ids=list(out.token_ids)
                    # vLLM excludes the terminal stop token from output IDs.
                    if out.finish_reason == 'stop':
                        stop=out.stop_reason if isinstance(out.stop_reason,int) else tok.eos_token_id
                        if not ids or ids[-1] != stop:
                            ids.append(stop)
                    text=tok.decode(ids, skip_special_tokens=False)
                    group['rollouts'].append(dict(tokens=ids,text=text,
                        truncated=out.finish_reason=='length', finish_reason=out.finish_reason))
                    grade_inputs.append((text,row['answer']))
                groups.append(group)
            grades=pool.map(grade_one, grade_inputs)
            for rollout, grade in zip((x for g in groups for x in g['rollouts']), grades):
                rollout.update(grade)
            chunk_name=f'{start:08d}.jsonl.gz'
            chunk_path=groups_dir/chunk_name
            temp=chunk_path.with_suffix('.tmp')
            with gzip.open(temp,'wt') as stream:
                for group in groups:stream.write(json.dumps(group,ensure_ascii=False)+'\n')
            temp.replace(chunk_path)
            chunks.append(dict(name=chunk_name,bytes=chunk_path.stat().st_size,
                sha256=hashlib.sha256(chunk_path.read_bytes()).hexdigest()))
            completed+=len(groups)
            receipt=dict(request_sha256=request_hash,groups_dir=str(groups_dir),chunks=chunks,
                completed_prompts=completed,engine_seconds=prior_engine+engine_seconds,elapsed_seconds=prior_elapsed+time.monotonic()-t0)
            write_json(output_path,receipt)
            backup_session()
            log(f"generation {completed}/{len(req['rows'])} prompts; chunk {time.monotonic()-t1:.1f}s")


def iter_groups(receipt):
    if 'chunks' in receipt:
        for chunk in receipt['chunks']:
            with gzip.open(Path(receipt['groups_dir'])/chunk['name'],'rt') as stream:
                for line in stream:yield json.loads(line)
    else:
        with Path(receipt['groups_path']).open() as stream:
            for line in stream:yield json.loads(line)


def clear_generation(receipt,path):
    Path(path).unlink(missing_ok=True)
    if 'groups_dir' in receipt:shutil.rmtree(receipt['groups_dir'],ignore_errors=True)
    elif 'groups_path' in receipt:Path(receipt['groups_path']).unlink(missing_ok=True)


def head_values(model, hidden, targets):
    """FP32 softmax after the selected BF16/FP16 vocabulary projection."""
    import torch
    import torch.nn.functional as F
    with torch.autocast('cuda',dtype=compute_dtype()):
        logits=model.lm_head(hidden)
    logits=logits.float()
    logp=F.log_softmax(logits,dim=-1)
    selected=logp.gather(-1,targets[:,None]).squeeze(-1)
    entropy=-(logp.exp()*logp).sum(-1)
    return selected,entropy


def hidden_for(model, group, rollout):
    import torch
    ids=torch.tensor([group['prompt_tokens']+rollout['tokens']],device='cuda')
    with torch.autocast('cuda',dtype=compute_dtype()):
        hidden=model.model(input_ids=ids[:,:-1],use_cache=False).last_hidden_state[0]
    n=len(rollout['tokens'])
    return hidden[-n:],ids[0,-n:]


@contextmanager
def source_precision(model):
    """Selected forward precision, FP32 master weights/gradients and optimizer.

    Autocast alone leaves embeddings, residuals and RMSNorm weights in FP32.
    The original FSDP actor instead casts *all parameters* during its forward.
    Keep this context through backward so checkpoint recomputation sees the
    same forward parameters. torch 2.9's tied-parameter context preserves aliases.
    """
    import torch
    from torch.nn.utils.stateless import _reparametrize_module
    parameters={name:p.to(compute_dtype()) for name,p in model.named_parameters()}
    with _reparametrize_module(model,parameters,tie_weights=True):
        yield


def record_logps(model, groups, key):
    import torch
    model.eval()
    model.gradient_checkpointing_disable()
    with torch.no_grad():
        flat=sorted([(g,r) for g in groups for r in g['rollouts']],
            key=lambda pair:len(pair[0]['prompt_tokens'])+len(pair[1]['tokens']))
        for start in range(0,len(flat),CONFIG['physical_batch_size']):
            items=flat[start:start+CONFIG['physical_batch_size']]
            with source_precision(model):
                hidden,targets=batched_hidden_for(model,items)
                values=[]
                for i in range(0,len(targets),256):
                    p,_=head_values(model,hidden[i:i+256],targets[i:i+256]);values.extend(p.cpu().tolist())
            offset=0
            for _,rollout in items:
                n=len(rollout['tokens']);rollout[key]=values[offset:offset+n];offset+=n
            if start%256==0:log(f'{key} {start+len(items)}/{len(flat)} responses')


def _policy_backward(model, group, rollout, scale, scaler=None):
    """Chunk the vocabulary projection without changing its gradient.

    Accumulate dL/dhidden in chunks, then backpropagate through the transformer
    once. Tied embedding/output parameters receive both gradient paths.
    """
    import torch
    hidden,targets=hidden_for(model,group,rollout)
    proxy=hidden.detach().requires_grad_(True)
    loss_value=0.
    for i in range(0,len(targets),128):
        p,entropy=head_values(model,proxy[i:i+128],targets[i:i+128])
        old=torch.tensor(rollout['old_logp'][i:i+128],device='cuda')
        ref=torch.tensor(rollout['ref_logp'][i:i+128],device='cuda')
        ratio=(p-old).exp()
        advantage=rollout['advantage']
        policy=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2))
        delta=ref-p
        kl=(delta.exp()-delta-1).clamp(-10,10)
        loss=(policy-CONFIG['entropy_coef']*entropy+CONFIG['kl_coef']*kl).sum()*scale
        (scaler.scale(loss) if scaler else loss).backward()
        loss_value+=loss.detach().item()
    hidden.backward(proxy.grad)
    return loss_value


def policy_backward(model,group,rollout,scale,scaler=None):
    with source_precision(model):
        return _policy_backward(model,group,rollout,scale,scaler)


def batched_hidden_for(model,items):
    import torch
    lengths=[len(g['prompt_tokens'])+len(r['tokens']) for g,r in items]
    # Padding positions are masked; zero is valid for both the real and test vocabularies.
    ids=torch.zeros((len(items),max(lengths)),device='cuda',dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,(g,r) in enumerate(items):
        ids[i,:lengths[i]]=torch.tensor(g['prompt_tokens']+r['tokens'],device='cuda')
        mask[i,:lengths[i]]=1
    with torch.autocast('cuda',dtype=compute_dtype()):
        full=model.model(input_ids=ids[:,:-1],attention_mask=mask[:,:-1],use_cache=False).last_hidden_state
    hidden=torch.cat([full[i,len(g['prompt_tokens'])-1:lengths[i]-1] for i,(g,r) in enumerate(items)])
    targets=torch.cat([ids[i,len(g['prompt_tokens']):lengths[i]] for i,(g,r) in enumerate(items)])
    return hidden,targets


def _batched_policy_backward(model, items, scale, scaler=None):
    """Padded transformer batches; exact response-only microbatch denominator."""
    import torch
    hidden,targets=batched_hidden_for(model,items)
    old=torch.tensor([p for _,r in items for p in r['old_logp']],device='cuda')
    ref=torch.tensor([p for _,r in items for p in r['ref_logp']],device='cuda')
    advantages=torch.tensor([r['advantage'] for _,r in items for _ in r['tokens']],device='cuda')
    proxy=hidden.detach().requires_grad_(True)
    value=0.
    for i in range(0,len(targets),256):
        p,entropy=head_values(model,proxy[i:i+256],targets[i:i+256])
        ratio=(p-old[i:i+256]).exp()
        a=advantages[i:i+256]
        delta=ref[i:i+256]-p
        loss=(torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2))
            -.001*entropy+.0001*(delta.exp()-delta-1).clamp(-10,10)).sum()*scale
        (scaler.scale(loss) if scaler else loss).backward();value+=loss.detach().item()
    hidden.backward(proxy.grad)
    return value


def batched_policy_backward(model,items,scale,scaler=None):
    with source_precision(model):
        return _batched_policy_backward(model,items,scale,scaler)


def finish_scaled_update(model,optimizer,scaler):
    import torch
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(norm)):
        if not scaler.is_enabled():raise FloatingPointError(f'nonfinite gradient norm {norm}')
        scaler.update(new_scale=scaler.get_scale()*.5)
        return norm,False
    scaler.step(optimizer);scaler.update()
    return norm,True


def update_worker(request_path, output_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import importlib.util
    req=read_json(request_path)
    set_seed(req['seed'])
    all_groups=read_json(req['buffer_path'])['groups']
    contributing_prompts=CONFIG['train_batch_prompts']
    groups=all_groups[:contributing_prompts]
    t0=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(req['model_path'],dtype=torch.float32,
        attn_implementation='sdpa').to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-7,betas=(.9,.999),eps=1e-8,
        weight_decay=.01,foreach=False)
    if req.get('optimizer_path'):
        optimizer.load_state_dict(torch.load(req['optimizer_path'],map_location='cpu',weights_only=True))
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and key!='step': state[key]=value.to('cuda')
    # Existing carry-over groups retain their behavior-policy log probabilities.
    fresh=[g for g in all_groups if 'old_logp' not in g['rollouts'][0]]
    record_logps(model,fresh,'old_logp')
    write_json(req['carry_output'],all_groups[contributing_prompts:])
    reference=AutoModelForCausalLM.from_pretrained(base_path(),dtype=compute_dtype(),
        attn_implementation='sdpa').to('cuda')
    reference.requires_grad_(False)
    record_logps(reference,groups,'ref_logp')
    del reference
    torch.cuda.empty_cache()
    for group in groups:
        rewards=torch.tensor([r['reward'] for r in group['rollouts']],dtype=torch.float32)
        advantages=(rewards-rewards.mean())/(rewards.std(correction=1)+1e-6)
        for r,a in zip(group['rollouts'],advantages):r['advantage']=a.item()
    # Use the exact released length-balancing routine, with four logical ranks.
    spec=importlib.util.spec_from_file_location('source_balance',SOURCE/'verl/utils/seqlen_balancing.py')
    balance=importlib.util.module_from_spec(spec);spec.loader.exec_module(balance)
    flat=[(g,r) for g in groups for r in g['rollouts']]
    parts=balance.get_seqlen_balanced_partitions([len(g['prompt_tokens'])+len(r['tokens']) for g,r in flat],4,equal_size=True)
    ranks=[[flat[i] for i in part] for part in parts]
    rank_responses=contributing_prompts*8//4
    assert all(len(rank)==rank_responses for rank in ranks), [len(x) for x in ranks]
    optimizer_steps=rank_responses//128
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    scaler=torch.amp.GradScaler('cuda',enabled=CONFIG['precision']=='float16',init_scale=128.)
    scaler_path=Path(req['model_path'])/'amp_scaler.pt'
    if scaler.is_enabled() and scaler_path.exists():scaler.load_state_dict(torch.load(scaler_path,map_location='cpu',weights_only=True))
    updates=[]
    for mini in range(optimizer_steps):
        for attempt in range(9):
            optimizer.zero_grad(set_to_none=True)
            value=0.
            for rank in ranks:
                local=rank[mini*128:(mini+1)*128]
                for start in range(0,128,32):
                    micro=local[start:start+32]
                    denom=sum(len(r['tokens']) for _,r in micro)
                    micro=sorted(micro,key=lambda pair:len(pair[0]['prompt_tokens'])+len(pair[1]['tokens']))
                    physical=CONFIG['physical_batch_size']
                    for offset in range(0,len(micro),physical):
                        value+=batched_policy_backward(model,micro[offset:offset+physical],1./(16*denom),scaler)
            # An FP16 overflow skips the optimizer step, lowers the scale and
            # replays this same minibatch, preserving 16 actual AdamW updates.
            norm,applied=finish_scaled_update(model,optimizer,scaler)
            if applied:break
            log(f'FP16 overflow at update {mini+1}; replay {attempt+1}, scale={scaler.get_scale()}')
        else:raise FloatingPointError('FP16 gradients stayed nonfinite after 9 attempts; use BF16 hardware.')
        updates.append(dict(loss=value,gradient_norm=norm.item(),loss_scale=scaler.get_scale(),overflow_retries=attempt))
        log(f'GRPO update {mini+1}/{optimizer_steps} loss={value:.6f} norm={norm.item():.5f}')
    optimizer.zero_grad(set_to_none=True)
    checkpoint=Path(req['checkpoint_path'])
    staging=checkpoint.parent/'.next_checkpoint'
    staging.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(staging,safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path()).save_pretrained(staging)
    optimizer_file=staging/'optimizer.pt'
    torch.save(optimizer.state_dict(),optimizer_file.with_suffix('.tmp'))
    optimizer_file.with_suffix('.tmp').replace(optimizer_file)
    if scaler.is_enabled():torch.save(scaler.state_dict(),staging/'amp_scaler.pt')
    write_json(staging/'commit.json',req['next_state'])
    write_json(staging/'carry.json',all_groups[contributing_prompts:])
    receipt=dict(updates=updates,elapsed_seconds=time.monotonic()-t0,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(staging/'update_receipt.json',receipt)
    import shutil
    previous=checkpoint.parent/'.previous_checkpoint'
    if previous.exists():shutil.rmtree(previous)
    if checkpoint.exists():checkpoint.rename(previous)
    staging.rename(checkpoint)
    if previous.exists():shutil.rmtree(previous)
    write_json(output_path,receipt)


def worker_error_tail(path,start=0,max_bytes=16384,max_lines=100):
    """Bound diagnostics to this worker's appended output, not older failures."""
    with Path(path).open('rb') as stream:
        stream.seek(0,2);end=stream.tell()
        stream.seek(max(start,end-max_bytes))
        text=stream.read(max_bytes).decode('utf-8',errors='replace')
    return '\n'.join(text.splitlines()[-max_lines:])


def subprocess_phase(python,args,phase,timeout):
    check_session_margin()
    if RUN_DEADLINE is not None:
        timeout=min(timeout,max(0,RUN_DEADLINE-time.monotonic()))
    env=os.environ.copy()
    env.update(HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS=os.environ.get('RLVR_CPU_THREADS','8'),VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONDONTWRITEBYTECODE='1',
        RLVR_RUN_DIR=str(RUN),RLVR_SCRATCH_DIR=str(SCRATCH),RLVR_SOURCE_DIR=str(SOURCE),
        RLVR_TRAIN_PYTHON=TRAIN_PYTHON,RLVR_INFER_PYTHON=INFER_PYTHON,
        RLVR_EXECUTION_CONFIG=json.dumps(CONFIG),RLVR_SESSION_END_EPOCH=str(SESSION_END_EPOCH))
    if PERSIST_DIR:env['RLVR_PERSIST_DIR']=str(PERSIST_DIR)
    if CONFIG['precision']=='float16':env['VLLM_ATTENTION_BACKEND']='XFORMERS'
    path=RUN/'worker.log'
    with path.open('a') as log_file:
        output_start=path.stat().st_size
        process=subprocess.Popen([python,'-u',str(Path(__file__).resolve()),*args],
            stdout=log_file,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        started=time.monotonic()
        log(f'{phase}: PID {process.pid}')
        while process.poll() is None:
            if time.monotonic()-started>timeout:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                if SESSION_END_EPOCH and time.time()>=SESSION_END_EPOCH:raise SessionPause('Session budget exhausted; uncommitted optimization will replay.')
                raise TimeoutError(f'{phase} exceeded {timeout}s')
            write_json(RUN/'status.json',dict(status='RUNNING',phase=phase,pid=process.pid,
                elapsed_seconds=time.monotonic()-started,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                config=CONFIG))
            time.sleep(30)
        if process.returncode==75:raise SessionPause('Worker saved a partial generation and paused for the next session.')
        if process.returncode:
            details=worker_error_tail(path,output_start)
            log(f'{phase}: worker error output from this invocation (last 100 lines, up to 16 KiB):')
            print(details or '(Worker produced no output.)',flush=True)
            raise RuntimeError(f'{phase} failed with exit code {process.returncode}; see {path}')


def generate(rows,model_path,n,temperature,seed,output,label,prompt_key='prompt_tokens'):
    request=SCRATCH/'generation_request.json'
    write_json(request,dict(rows=rows,model_path=str(model_path),n=n,temperature=temperature,
        seed=seed,prompt_key=prompt_key))
    subprocess_phase(INFER_PYTHON,['generate','--request',str(request),'--output',str(output)],label,24*3600)
    return read_json(output)


def evaluate(rows,model_path,seed,step,kind,n=32):
    dest=RUN/f'seed_{seed}'/f'{kind}_evaluation.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and read_json(dest).get('step')==step:return
    raw=SCRATCH/f'{kind}_evaluation.json'
    offset={'final':0,'confirmation':1000000,'validation':2000000}.get(kind,0)
    evaluation_seed=CONFIG['initial_evaluation_seed'] if seed==0 else 700000+seed*10000+step+offset
    validation=kind=='validation'
    temperature=0. if validation else .7
    out=generate(rows,model_path,n,temperature,evaluation_seed,raw,
        f'seed {seed} step {step} {kind}',prompt_key='prompt_tokens' if validation else 'eval_prompt_tokens')
    result=dict(step=step,seed=seed,generation_seed=evaluation_seed,samples_per_prompt=n,temperature=temperature,
        prompt_key='prompt_tokens' if validation else 'eval_prompt_tokens',
        elapsed_seconds=out['elapsed_seconds'],per_prompt=[])
    for g in iter_groups(out):
        result['per_prompt'].append(dict(id=g['id'],samples=len(g['rollouts']),
            successes=sum(r['reward' if validation else 'evaluation_reward'] for r in g['rollouts']),
            training_scorer_successes=sum(r['reward'] for r in g['rollouts']),
            truncated=sum(r['truncated'] for r in g['rollouts'])))
    # Retain evaluation evidence once per question rather than a new report copy.
    if kind!='validation':
        final=dest.with_suffix('.rollouts.jsonl.gz')
        temporary=final.with_suffix('.tmp')
        if 'chunks' in out:
            with temporary.open('wb') as output:
                for chunk in out['chunks']:
                    with (Path(out['groups_dir'])/chunk['name']).open('rb') as source:shutil.copyfileobj(source,output)
        else:
            with Path(out['groups_path']).open('rb') as source,gzip.open(temporary,'wb') as output:shutil.copyfileobj(source,output)
        temporary.replace(final)
    else:
        curve=RUN/f'seed_{seed}'/'validation_curve.jsonl'
        existing=[json.loads(line)['step'] for line in curve.read_text().splitlines()] if curve.exists() else []
        if step not in existing:
            with curve.open('a') as stream:stream.write(json.dumps(result)+'\n')
    result['precision']=CONFIG['precision']
    write_json(dest,result)
    clear_generation(out,raw)
    backup_session()


def summarize(seed):
    """Operational source labels, explicitly separate from impossibility claims."""
    import numpy as np
    from scipy.stats import beta
    seed_dir=RUN/f'seed_{seed}'
    initial=read_json(RUN/'seed_0'/'initial_evaluation.json')
    final=read_json(seed_dir/'final_evaluation.json')
    initial_by_id={r['id']:r for r in initial['per_prompt']}
    final_by_id={r['id']:r for r in final['per_prompt']}
    exposures=collections.defaultdict(lambda:dict(samples=0,positives=0,visits=0))
    with (seed_dir/'prompt_reward_stats.csv').open() as f:
        for row in csv.DictReader(f):
            e=exposures[int(row['prompt_id'])];n=int(row['n'])
            e['samples']+=n;e['visits']+=1;e['positives']+=round(float(row['mean_reward'])*n)
    hard={i for i,r in initial_by_id.items() if r['successes']/r['samples']<=.1}
    no_positive={i for i in hard if exposures[i]['positives']==0}
    candidate={i for i in hard-no_positive if final_by_id[i]['successes']/final_by_id[i]['samples']<=.1}
    learned={i for i in hard-no_positive if i not in candidate}
    numerical=read_json(RUN/'numerical_protocol.json') if (RUN/'numerical_protocol.json').exists() else {}
    result=dict(seed=seed,step=final['step'],training_population=len(initial_by_id),
        precision=final.get('precision',numerical.get('precision','bfloat16')),numerical_protocol=numerical,
        initially_hard=len(hard),hard_without_training_positive=len(no_positive),
        single_seed_candidate_count=len(candidate),single_seed_learned_count=len(learned),
        candidate_fraction_of_initially_hard=len(candidate)/max(1,len(hard)),
        candidate_ids=sorted(candidate),candidate_positive_exposures={i:exposures[i] for i in sorted(candidate)},
        candidates_above_threshold_under_training_scorer=sum(
            final_by_id[i]['training_scorer_successes']/final_by_id[i]['samples']>.1 for i in candidate),
        source_definition='initial and final empirical pass@1 <= .1; at least one training positive',
        warning='Single seed, finite 32-sample labels, not a three-seed intersection or a representation impossibility result.')
    validation_path=seed_dir/'validation_curve.jsonl'
    if validation_path.exists():
        curve=[]
        for line in validation_path.read_text().splitlines():
            row=json.loads(line);samples=sum(r['samples'] for r in row['per_prompt'])
            curve.append(dict(step=row['step'],success_rate=sum(r['successes'] for r in row['per_prompt'])/samples))
        result['validation_curve']=curve
        result['validation_final_minus_initial']=curve[-1]['success_rate']-curve[0]['success_rate']
        result['validation_last_three_mean_minus_previous_three_mean']=(
            float(np.mean([r['success_rate'] for r in curve[-3:]]))-float(np.mean([r['success_rate'] for r in curve[-6:-3]]))) if len(curve)>=6 else None
        result['validation_interpretation']='Descriptive trajectory only. Saturation and effective training require examining the curve; no automatic failure-reproduced declaration.'
    confirmation_path=seed_dir/'confirmation_evaluation.json'
    if confirmation_path.exists():
        confirmed=read_json(confirmation_path)
        scores=[];training_scores=[];certified=[]
        for row in confirmed['per_prompt']:
            n,k=row['samples'],row['successes'];scores.append(k/n)
            training_scores.append(row['training_scorer_successes']/n)
            upper=1. if k==n else float(beta.ppf(.975,k+1,n-k))
            if upper<.1:certified.append(row['id'])
        rng=np.random.default_rng(2851)
        means=[float(np.mean(rng.choice(scores,len(scores),replace=True))) for _ in range(5000)] if scores else []
        result['independent_confirmation']=dict(prompts=len(scores),samples_per_prompt=128,
            mean_success_rate=float(np.mean(scores)) if scores else None,
            mean_training_scorer_success_rate=float(np.mean(training_scores)) if training_scores else None,
            prompt_bootstrap_ci95=np.quantile(means,[.025,.975]).tolist() if means else None,
            pointwise_binomial_upper_below_threshold=len(certified),pointwise_ids=certified,
            caveat='Pointwise 95% binomial intervals, not simultaneous confidence statements. Bootstrap conditions on this selected cohort and one training seed.')
    write_json(seed_dir/'analysis.json',result)
    log(f'seed {seed}: {len(candidate)} operational persistent-failure candidates; {len(learned)} initially hard positive-exposed prompts learned')
    return result


def report_completed_seed(analysis):
    """Write continuation results without modifying the archived study."""
    path=RUN/'report.md'
    rows=[f"- Precision: {analysis.get('precision','bfloat16')}; FP16 runs are a numerical variant, separate from BF16 reproduction.",
        f"- Training seed: {analysis['seed']}; completed step: {analysis['step']}.",
        f"- Initially hard prompts: {analysis['initially_hard']}.",
        f"- Hard prompts with no logged training positive: {analysis['hard_without_training_positive']}.",
        f"- Positive-exposed hard prompts still below the final 10% empirical threshold: {analysis['single_seed_candidate_count']}.",
        f"- Positive-exposed hard prompts above the final threshold: {analysis['single_seed_learned_count']}.",
        f"- Candidate prompts exceeding the threshold under the training answer extractor: {analysis['candidates_above_threshold_under_training_scorer']}."]
    if analysis.get('independent_confirmation'):
        c=analysis['independent_confirmation']
        rows.extend([f"- Independent 128-response confirmation mean: {100*c['mean_success_rate']:.3f}%.",
            f"- Confirmation mean under the training extractor: {100*c['mean_training_scorer_success_rate']:.3f}%.",
            f"- Candidate prompts whose pointwise 95% binomial upper bound is below 10%: {c['pointwise_binomial_upper_below_threshold']}."])
    if 'validation_final_minus_initial' in analysis:
        rows.append(f"- Final minus initial validation accuracy: {100*analysis['validation_final_minus_initial']:+.2f} percentage points.")
    rows.extend(['',f"Canonical detailed results: [seed {analysis['seed']} analysis](seed_{analysis['seed']}/analysis.json).",'',
        'This is a single-seed result. Inspect validation convergence, extractor disagreements, truncation and reward exposure before interpreting the cohort as a selective learning failure. Three-seed robustness and ordinary repairs remain separate required experiments.'])
    path.write_text('# Local reproduction results\n\n'+'\n'.join(rows)+'\n')


def gpu_preflight():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check nvidia-smi, the CUDA-enabled Torch install, '
            'CUDA_VISIBLE_DEVICES, and container GPU passthrough. No sampling or training was started.')
    capability=torch.cuda.get_device_capability(0)
    if CONFIG['precision']=='bfloat16' and capability[0]<8:
        raise RuntimeError('Native BF16 requires Ampere or newer (e.g. L4/A100). T4 requires the explicitly labelled --precision float16 variant.')
    if capability<(7,5):raise RuntimeError('This notebook stack requires Turing/T4 or newer.')
    x=torch.ones((8,8),device='cuda',dtype=compute_dtype())
    assert bool(torch.isfinite(x@x).all())
    log(f'GPU preflight passed: {torch.cuda.get_device_name(0)}; Torch {torch.__version__}')


def persist_update(seed_dir,state,groups,rollout_seed):
    """Durable, machine-independent transaction, before any optimizer update."""
    folder=seed_dir/'pending_update';folder.mkdir(exist_ok=True)
    buffer=folder/'buffer.json';write_json(buffer,dict(groups=groups))
    next_state=dict(state,step=state['step']+1)
    write_json(folder/'request.json',dict(seed=rollout_seed,next_state=next_state,
        prepared_from_step=state['step'],buffer_sha256=hashlib.sha256(buffer.read_bytes()).hexdigest()))
    (seed_dir/'sampling_progress.json').unlink(missing_ok=True)
    backup_session()


def finish_pending_update(seed_dir,state):
    """Resume from committed weights, using saved rollouts, without duplication."""
    import shutil
    folder=seed_dir/'pending_update';metadata=read_json(folder/'request.json')
    target=metadata['next_state']['step']
    if target<=state['step']:
        # The model transaction committed before a prior driver interruption.
        receipt_path=seed_dir/'latest_checkpoint'/'update_receipt.json'
        if receipt_path.exists():append_update_receipt(seed_dir,target,read_json(receipt_path))
        write_json(seed_dir/'state.json',state)
        shutil.rmtree(folder)
        return state,read_json(seed_dir/'latest_checkpoint'/'carry.json')
    if metadata['prepared_from_step']!=state['step'] or target!=state['step']+1:
        raise ValueError('Pending update does not follow the committed checkpoint; refusing unsafe resume.')
    buffer=folder/'buffer.json'
    if hashlib.sha256(buffer.read_bytes()).hexdigest()!=metadata['buffer_sha256']:
        raise ValueError('Pending rollout buffer hash mismatch')
    checkpoint=seed_dir/'latest_checkpoint'
    request=folder/'runtime_request.json';receipt=folder/'receipt.json';carry=folder/'carry.json'
    write_json(request,dict(seed=metadata['seed'],model_path=str(checkpoint if state['step'] else base_path()),
        optimizer_path=str(checkpoint/'optimizer.pt') if state['step'] else None,
        buffer_path=str(buffer),checkpoint_path=str(checkpoint),carry_output=str(carry),
        next_state=metadata['next_state']))
    subprocess_phase(TRAIN_PYTHON,['update','--request',str(request),'--output',str(receipt)],
        f'seed {seed_dir.name} step {target} optimization',12*3600)
    committed=read_json(checkpoint/'commit.json')
    if committed!=metadata['next_state']:raise RuntimeError('Checkpoint commit does not match pending transaction')
    append_update_receipt(seed_dir,target,read_json(checkpoint/'update_receipt.json'))
    write_json(seed_dir/'state.json',committed)
    pending=read_json(checkpoint/'carry.json')
    shutil.rmtree(folder)
    backup_session()
    return committed,pending


def append_update_receipt(seed_dir,step,receipt):
    path=seed_dir/'optimization_curve.jsonl'
    existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if not any(r['step']==step for r in existing):
        with path.open('a') as f:f.write(json.dumps(dict(step=step,**receipt))+'\n')


def rollback_uncommitted_stats(seed_dir,step):
    """Unfinished sampling can be replayed deterministically after a crash."""
    path=seed_dir/'prompt_reward_stats.csv'
    if not path.exists():return
    with path.open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    retained=[r for r in rows if int(r['step'])<=step]
    if len(retained)!=len(rows):
        tmp=path.with_suffix('.tmp')
        with tmp.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(retained)
        tmp.replace(path)
        log(f'Rewound {len(rows)-len(retained)} uncommitted sampling rows before deterministic replay')


def driver(benchmark=False,seed=1):
    global RUN_DEADLINE,SESSION_END_EPOCH
    import torch
    RUN.mkdir(exist_ok=True);SCRATCH.mkdir(exist_ok=True)
    gpu_preflight()
    if not (RUN/'inputs.json').exists():prepare()
    backup_session()
    inputs=read_json(RUN/'inputs.json')['splits']
    if benchmark:
        # Runtime/memory benchmark, never interpreted as learning evidence.
        output=SCRATCH/'benchmark_rollouts.json'
        out=generate(inputs['train'][:64],base_path(),8,1.,431,output,'source recipe runtime benchmark')
        tokens=sum(len(r['tokens']) for g in iter_groups(out) for r in g['rollouts'])
        previous=read_json(RUN/'benchmark.json') if (RUN/'benchmark.json').exists() else None
        write_json(RUN/'benchmark.json',dict(prompts=64,responses=512,
            response_tokens=tokens,elapsed_seconds=out['elapsed_seconds'],
            engine_startup_seconds=out['engine_seconds'],
            output_tokens_per_second=tokens/(out['elapsed_seconds']-out['engine_seconds']),
            truncated=sum(r['truncated'] for g in iter_groups(out) for r in g['rollouts']),
            inference_execution=CONFIG['inference_execution'],previous_eager_benchmark=previous,
            interpretation='Throughput check only; not a persistent-failure test.'))
        log(f'benchmark generated {tokens} tokens in {out["elapsed_seconds"]:.1f}s')
        write_json(RUN/'status.json',dict(status='BENCHMARK_COMPLETED',benchmark=read_json(RUN/'benchmark.json')))
        return
    RUN_DEADLINE=time.monotonic()+max(0,SESSION_END_EPOCH-time.time()) if SESSION_END_EPOCH else time.monotonic()+CONFIG['hard_timeout_hours_per_seed']*3600
    seed_dir=RUN/f'seed_{seed}';seed_dir.mkdir(exist_ok=True)
    # A process may stop between moving the old checkpoint aside and promoting
    # the staged replacement. Replay from the last visible committed version.
    checkpoint=seed_dir/'latest_checkpoint'
    previous=seed_dir/'.previous_checkpoint'
    if not checkpoint.exists() and previous.exists():previous.rename(checkpoint)
    state_path=seed_dir/'state.json'
    state=read_json(state_path) if state_path.exists() else dict(step=0,epoch=0,cursor=0,generation_batches=0)
    commit=seed_dir/'latest_checkpoint'/'commit.json'
    if commit.exists():state=read_json(commit)
    model_path=seed_dir/'latest_checkpoint' if state['step'] else Path(base_path())
    started=time.monotonic()
    checkpoint_carry=seed_dir/'latest_checkpoint'/'carry.json'
    pending=read_json(checkpoint_carry) if checkpoint_carry.exists() else []
    resumed_sampling=(seed_dir/'sampling_progress.json').exists()
    if (seed_dir/'pending_update/request.json').exists():
        state,pending=finish_pending_update(seed_dir,state)
        model_path=seed_dir/'latest_checkpoint'
    elif (seed_dir/'sampling_progress.json').exists():
        progress=read_json(seed_dir/'sampling_progress.json')
        if progress['base_step']!=state['step']:raise ValueError('Sampling progress does not match the committed model.')
        state=progress['state'];pending=progress['groups']
    else:rollback_uncommitted_stats(seed_dir,state['step'])
    if state['step']==0 or state['step']%CONFIG['validation_every']==0:
        evaluate(inputs['test'],model_path,seed,state['step'],'validation',n=1)
    while state['step']<CONFIG['final_step'] and state['epoch']<CONFIG['training_epochs']:
        check_session_margin()
        cycle_started=time.monotonic()
        target=CONFIG['train_batch_prompts']
        sampling_batch=CONFIG['sampling_batch_prompts']
        while len(pending)<target and state['epoch']<CONFIG['training_epochs']:
            # Torch RandomSampler equivalent deterministic permutation per epoch.
            from torch.utils.data import RandomSampler
            sampler=RandomSampler(range(len(inputs['train'])),generator=torch.Generator().manual_seed(seed))
            for _ in range(state['epoch']+1):order=list(sampler)
            if state['cursor']+sampling_batch>len(order):
                state['epoch']+=1;state['cursor']=0;continue
            ids=order[state['cursor']:state['cursor']+sampling_batch]
            rows=[inputs['train'][i] for i in ids]
            dest=SCRATCH/'training_rollouts.json'
            rollout_seed=seed*100000+state['generation_batches']
            out=generate(rows,model_path,8,1.,rollout_seed,dest,
                f'seed {seed} step {state["step"]+1} sampling batch {state["generation_batches"]+1}')
            with (seed_dir/'prompt_reward_stats.csv').open('a') as f:
                writer=csv.writer(f)
                if f.tell()==0:writer.writerow(['step','prompt_id','mean_reward','std_reward','n','processed'])
                for group in iter_groups(out):
                    rewards=torch.tensor([r['reward'] for r in group['rollouts']],dtype=torch.float32)
                    writer.writerow([state['step']+1,group['id'],rewards.mean().item(),rewards.std().item(),8,False])
                    if 0<sum(rewards)<8:pending.append(group)
            state['cursor']+=sampling_batch;state['generation_batches']+=1
            # State advanced only in the in-progress journal; checkpoint state
            # remains the last completely committed model/optimizer transaction.
            write_json(seed_dir/'sampling_journal.json',state)
            write_json(seed_dir/'sampling_progress.json',dict(base_step=state['step'],state=state,groups=pending))
            clear_generation(out,dest)
            backup_session()
            log(f'seed {seed}: {len(pending)}/{target} contributing prompts collected')
        if len(pending)<target:break
        persist_update(seed_dir,state,pending,seed+state['step'])
        state,pending=finish_pending_update(seed_dir,state)
        model_path=seed_dir/'latest_checkpoint'
        measurement_id=os.environ.get('RLVR_TIMING_MEASUREMENT_ID')
        if measurement_id and not resumed_sampling:
            with (seed_dir/'step_timings.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(measurement_id=measurement_id,step=state['step'],
                    seconds=time.monotonic()-cycle_started,scope='sampling, optimization and checkpoint publication; excludes validation/evaluation'))+'\n')
        resumed_sampling=False
        if state['step']%CONFIG['validation_every']==0:evaluate(inputs['test'],model_path,seed,state['step'],'validation',n=1)
    # The immutable base model can be evaluated after training without leaking
    # outcomes into its fixed, full-population evaluation sample or RNG seed.
    if not (RUN/'seed_0'/'initial_evaluation.json').exists():
        evaluate(inputs['train'],base_path(),0,0,'initial')
    evaluate(inputs['train'],model_path,seed,state['step'],'final')
    analysis=summarize(seed)
    if analysis['candidate_ids']:
        candidate_ids=set(analysis['candidate_ids'])
        evaluate([r for r in inputs['train'] if r['id'] in candidate_ids],
            model_path,seed,state['step'],'confirmation',n=128)
        analysis=summarize(seed)
    report_completed_seed(analysis)
    write_json(RUN/'status.json',dict(status='SEED_COMPLETED',seed=seed,state=state,precision=CONFIG['precision']))
    backup_session()


def main():
    global RUN,SCRATCH,SOURCE,TRAIN_PYTHON,INFER_PYTHON,PERSIST_DIR,SESSION_END_EPOCH
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','benchmark','run','generate','update','summarize'])
    parser.add_argument('--request');parser.add_argument('--output');parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--scratch-dir',type=Path)
    parser.add_argument('--train-python',default=TRAIN_PYTHON)
    parser.add_argument('--infer-python',default=INFER_PYTHON)
    parser.add_argument('--max-hours',type=float,default=CONFIG['hard_timeout_hours_per_seed'])
    parser.add_argument('--steps',type=int,default=CONFIG['final_step'])
    parser.add_argument('--precision',choices=['bfloat16','float16'],default=CONFIG['precision'])
    parser.add_argument('--physical-batch-size',type=int,default=CONFIG['physical_batch_size'])
    parser.add_argument('--generation-chunk-responses',type=int,default=CONFIG['generation_chunk_responses'])
    parser.add_argument('--max-num-seqs',type=int,default=CONFIG['max_num_seqs'])
    parser.add_argument('--max-num-batched-tokens',type=int,default=CONFIG['max_num_batched_tokens'])
    parser.add_argument('--grading-workers',type=int,default=CONFIG['grading_workers'])
    parser.add_argument('--enforce-eager',action='store_true',default=CONFIG['enforce_eager'])
    parser.add_argument('--pause-margin-minutes',type=float,default=CONFIG['pause_margin_minutes'])
    parser.add_argument('--persist-dir',type=Path,default=Path(PERSIST_DIR) if PERSIST_DIR else None)
    args=parser.parse_args()
    RUN=args.run_dir.expanduser().resolve();RUN.mkdir(parents=True,exist_ok=True)
    SCRATCH=(args.scratch_dir or RUN/'scratch').expanduser().resolve();SCRATCH.mkdir(parents=True,exist_ok=True)
    TRAIN_PYTHON=str(Path(shutil.which(args.train_python) or args.train_python).expanduser().resolve())
    INFER_PYTHON=str(Path(shutil.which(args.infer_python) or args.infer_python).expanduser().resolve())
    if args.max_hours<=0 or args.steps<1:parser.error('--max-hours and --steps must be positive')
    CONFIG['hard_timeout_hours_per_seed']=args.max_hours;CONFIG['final_step']=args.steps
    for key in ['precision','physical_batch_size','generation_chunk_responses','max_num_seqs',
        'max_num_batched_tokens','grading_workers','enforce_eager','pause_margin_minutes']:
        CONFIG[key]=getattr(args,key)
    for key in ['physical_batch_size','generation_chunk_responses','max_num_seqs','max_num_batched_tokens','grading_workers']:
        if CONFIG[key]<1:parser.error(f'{key} must be positive')
    if not 0<=args.pause_margin_minutes<args.max_hours*60:parser.error('Pause margin must fit inside the session budget')
    PERSIST_DIR=str(args.persist_dir.expanduser().resolve()) if args.persist_dir else None
    CONFIG['inference_execution']='eager xFormers; no torch.compile' if args.enforce_eager and args.precision=='float16' else ('eager; no torch.compile' if args.enforce_eager else 'CUDA graphs for decode; no torch.compile')
    if args.mode in ['run','benchmark']:
        SESSION_END_EPOCH=time.time()+args.max_hours*3600
        protocol_path=RUN/'numerical_protocol.json'
        numerical={key:CONFIG[key] for key in ['precision','physical_batch_size','generation_chunk_responses',
            'max_num_seqs','max_num_batched_tokens','enforce_eager']}
        numerical['interpretation']='FP16 numerical variant; do not pool with BF16 reproduction' if args.precision=='float16' else 'BF16 reproduction adaptation'
        if protocol_path.exists() and read_json(protocol_path)!=numerical:
            parser.error('Numerical settings changed for this run; use a separate directory and explicit experiment label')
        write_json(protocol_path,numerical)
    if args.mode in ['run','benchmark']:
        write_json(RUN/'execution_config.json',dict(config=CONFIG,run_dir=str(RUN),scratch_dir=str(SCRATCH),
            training_python=TRAIN_PYTHON,inference_python=INFER_PYTHON,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='summarize':summarize(args.seed)
        elif args.mode=='generate':generation_worker(args.request,args.output)
        elif args.mode=='update':update_worker(args.request,args.output)
        else:driver(args.mode=='benchmark',args.seed)
    except SessionPause as error:
        write_json(RUN/'status.json',dict(status='PAUSED',reason=str(error),precision=CONFIG['precision']))
        backup_session()
        log(str(error))
        if args.mode in ['generate','update']:sys.exit(75)
    except Exception as e:
        if args.mode in ['run','benchmark']:
            write_json(RUN/'status.json',dict(status='FAILED',error=repr(e),mode=args.mode))
        raise


if __name__=='__main__':main()
