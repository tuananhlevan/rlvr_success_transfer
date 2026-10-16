#!/usr/bin/env python3
"""A bounded final-answer/response-context probe, not a solution-quality evaluator."""
import argparse
import gzip
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
INPUT_SHA256 = '5d40512e6b0ead45f0f0f82d3f39853e3ca546cbc008982bc76d6d196dc4eb1b'
SEED = 20261016


def last_box(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += 7
    depth = 1
    for end in range(start, len(text)):
        depth += int(text[end] == '{') - int(text[end] == '}')
        if depth == 0:
            value = text[start:end].strip()
            return (start, end, value) if re.fullmatch(r'-?\d+', value) else None
    return None


def select_items(source, tokenizer, rng):
    items = []
    for line in gzip.open(source, 'rt'):
        row = json.loads(line)
        groups = [[], []]
        for rollout in row['rollouts']:
            box = last_box(rollout['text'])
            if (box and not rollout['truncated']
                    and rollout['reward'] == rollout.get('evaluation_reward')
                    and len(tokenizer.encode(rollout['text'])) <= 650
                    and str(rollout['extracted_answer']).strip() == box[2]):
                groups[int(rollout['reward'])].append((rollout, box))
        if not all(groups):
            continue
        good, good_box = groups[1][0]
        bad, bad_box = groups[0][0]
        if good_box[2] == bad_box[2]:
            continue
        good_text = good['text'].replace('<|endoftext|>', '')
        bad_text = bad['text'].replace('<|endoftext|>', '')
        injected = good_text[:good_box[0]] + bad_box[2] + good_text[good_box[1]:]
        repaired = bad_text[:bad_box[0]] + good_box[2] + bad_text[bad_box[1]:]
        items.append(dict(id=row['id'], prompt_tokens=row['prompt_tokens'],
                          responses=[good_text, injected, repaired, bad_text],
                          answers=[good_box[2], bad_box[2]]))
    eligible = len(items)
    rng.shuffle(items)
    items = items[:160]
    if len(items) < 60:
        raise ValueError('Insufficient eligible paired prompts')
    return items, eligible


def load_cached_first(cls, offline, **kwargs):
    kwargs.update(pretrained_model_name_or_path='Qwen/Qwen2.5-0.5B', revision=REVISION)
    try:
        return cls.from_pretrained(local_files_only=True, **kwargs)
    except OSError:
        if offline:
            raise
        print('[probe] Downloading the pinned model to the Hugging Face cache', flush=True)
        return cls.from_pretrained(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'studies/rlvr_success_transfer/reproduction/latest_sampling_groups.jsonl.gz')
    parser.add_argument('--output', type=Path, default=Path('/tmp/quartet_results.json'))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != INPUT_SHA256:
        raise ValueError(f'Input changed: expected {INPUT_SHA256}, received {digest}')
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA Torch environment is required')
    total_start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tokenizer = load_cached_first(AutoTokenizer, args.offline)
    items, eligible = select_items(args.source, tokenizer, rng)
    print(f'[probe] {eligible} eligible; using {len(items)} grouped prompts', flush=True)
    model = load_cached_first(AutoModelForCausalLM, args.offline,
                              dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    features, lengths, canonical = [], [], []

    def feature(item, response):
        ids = item['prompt_tokens'] + tokenizer.encode(response, add_special_tokens=False)
        prompt_length = len(item['prompt_tokens'])
        tensor = torch.tensor([ids], device='cuda')
        out = model.model(input_ids=tensor, output_hidden_states=True, use_cache=False)
        vector = out.hidden_states[-5][0, prompt_length:].float().mean(0).cpu().numpy()
        return vector, len(ids) - prompt_length

    feature_start = time.monotonic()
    with torch.inference_mode():
        for index, item in enumerate(items):
            rows, lens = zip(*(feature(item, response) for response in item['responses']))
            features.append(rows)
            lengths.append(lens)
            if index % 20 == 0:
                print(f'[probe] Features {index}/{len(items)}; {time.monotonic()-feature_start:.1f}s', flush=True)
    feature_seconds = time.monotonic() - feature_start
    canonical_start = time.monotonic()
    with torch.inference_mode():
        for item in items:
            rows = [feature(item, 'Final answer: \\boxed{' + answer + '}')[0]
                    for answer in item['answers']]
            canonical.append([rows[0], rows[1], rows[0], rows[1]])
    canonical_seconds = time.monotonic() - canonical_start
    del model
    torch.cuda.empty_cache()
    x, length, canonical = np.array(features), np.array(lengths, dtype=np.float32), np.array(canonical)
    count = len(items)
    cut = 2 * count // 3
    if len({a['id'] for a in items}) != count:
        raise ValueError('Repeated prompt IDs would invalidate the grouped split')
    mean = x[:cut].reshape(-1, x.shape[-1]).mean(0)
    std = x[:cut].reshape(-1, x.shape[-1]).std(0).clip(.05)
    z = torch.tensor((x - mean) / std)

    def fit(kind, regime, seed):
        torch.manual_seed(seed)
        inputs = z
        if kind == 'canonical':
            cm = canonical[:cut].reshape(-1, canonical.shape[-1]).mean(0)
            cs = canonical[:cut].reshape(-1, canonical.shape[-1]).std(0).clip(.05)
            inputs = torch.tensor((canonical - cm) / cs)
        if kind == 'length':
            inputs = torch.tensor(((length - length[:cut].mean()) / max(length[:cut].std(), 1))[:, :, None])
        dimension = inputs.shape[-1]
        head = (torch.nn.Sequential(torch.nn.Linear(dimension, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
                if kind == 'mlp' else torch.nn.Linear(dimension, 1, bias=False))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.005, weight_decay=.1)
        for _ in range(350):
            scores = head(inputs[:cut]).squeeze(-1)
            differences = {'synthetic': scores[:, 0] - scores[:, 1],
                           'natural': scores[:, 0] - scores[:, 3],
                           'repaired': scores[:, 2] - scores[:, 3]}
            delta = torch.cat(list(differences.values())) if regime == 'mixed' else differences[regime]
            loss = torch.nn.functional.softplus(-delta).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return head(inputs[cut:]).squeeze(-1).numpy()

    result = dict(status='exploratory; no new method validated',
                  meaning='Final integer answer correctness only. Edits may contradict the reasoning; not overall solution quality labels.',
                  eligible_prompts=eligible, n_prompts=count, train_prompts=cut, test_prompts=count-cut,
                  feature_seconds=feature_seconds, canonical_feature_seconds=canonical_seconds,
                  model_revision=REVISION, layer='hidden_states[-5]', source_sha256=digest,
                  cell_order=['natural_correct', 'injected_wrong', 'repaired_outcome', 'natural_wrong'],
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   transformers=transformers.__version__, numpy=np.__version__,
                                   gpu=torch.cuda.get_device_name()),
                  models={}, predictions={},
                  ids=dict(train=[a['id'] for a in items[:cut]], test=[a['id'] for a in items[cut:]]))
    primary = {}
    for kind in ['length', 'linear', 'mlp', 'canonical']:
        for regime in ['synthetic', 'natural', 'repaired', 'mixed']:
            scores = np.array([fit(kind, regime, seed) for seed in [1, 2, 3]])
            differences = np.stack([scores[:, :, 0]-scores[:, :, 1],
                                    scores[:, :, 2]-scores[:, :, 3],
                                    scores[:, :, 0]-scores[:, :, 3]], -1)
            hits = (differences > 0).astype(float) + .5 * (differences == 0)
            key = f'{kind}_{regime}'
            result['models'][key] = dict(injected_pair_accuracy=hits[:, :, 0].mean().item(),
                                         repaired_pair_accuracy=hits[:, :, 1].mean().item(),
                                         natural_pair_accuracy=hits[:, :, 2].mean().item(),
                                         per_seed=hits.mean(1).tolist())
            result['predictions'][key] = scores.tolist()
            if kind == 'linear' and regime in ['synthetic', 'natural']:
                primary[regime] = hits[:, :, 2].mean(0)
            print(f'[probe] {key}: {result["models"][key]["natural_pair_accuracy"]:.3f} natural-pair accuracy', flush=True)
    delta = primary['natural'] - primary['synthetic']
    boots = np.array([delta[rng.integers(0, count-cut, count-cut)].mean() for _ in range(3000)])
    result['natural_vs_synthetic_natural_pair_difference'] = dict(
        mean=float(delta.mean()), paired_prompt_bootstrap_95=np.quantile(boots, [.025, .975]).tolist(), replicates=3000)
    result['total_seconds'] = time.monotonic() - total_start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[probe] Complete: {args.output}; {result["total_seconds"]:.1f}s total', flush=True)


if __name__ == '__main__':
    main()
