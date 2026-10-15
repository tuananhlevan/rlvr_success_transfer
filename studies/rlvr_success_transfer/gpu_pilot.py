"""Single-GPU, full-parameter pilot. Not a reproduction of the source paper.

Frozen selection data, independent baseline evaluation, three ordinary repairs.
No external model/API calls. Retains one canonical JSON with rollout evidence.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parent
CONFIG = dict(seed=20261015, model="Qwen/Qwen2.5-0.5B",
              revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
              max_new_tokens=1024, group_size=8, generation_batch=8,
              selected_prompts=8, epochs=4, learning_rate=5e-6,
              temperature=1.0, top_p=1.0, top_k=0, clip_epsilon=.2,
              grad_clip=1., dtype="float32 parameters / bfloat16 autocast",
              arms=["clipped_grpo_replay", "student_success_sft", "expert_sft"],
              timeout_seconds=10800)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prompt(row):
    return row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def grade(row, text):
    try:
        # Require an explicitly boxed generated answer; avoid matching a number
        # in the reasoning or an echoed question as a final answer.
        if "\\boxed" not in text:
            return False
        gold, prediction = parse(row["solution"]), parse(text)
        return bool(gold and prediction and verify(gold, prediction))
    except Exception:
        return False


def sample(model, tokenizer, rows, seed, label):
    set_seed(seed)
    model.eval()
    model.gradient_checkpointing_disable()
    outputs = []
    for row in rows:
        start = time.monotonic()
        encoded = tokenizer([prompt(row)] * CONFIG["group_size"], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sequences = model.generate(**encoded, do_sample=True, temperature=1., top_p=1., top_k=0,
                                       max_new_tokens=CONFIG["max_new_tokens"],
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id, use_cache=True)
        n_prompt = encoded.input_ids.shape[1]
        group = []
        for seq in sequences:
            response = seq[n_prompt:].tolist()
            if tokenizer.eos_token_id in response:
                response = response[:response.index(tokenizer.eos_token_id)+1]
            text = tokenizer.decode(response, skip_special_tokens=True)
            group.append(dict(tokens=response, text=text, reward=int(grade(row, text)),
                              truncated=response[-1] != tokenizer.eos_token_id,
                              length=len(response)))
        outputs.append(dict(id=row["id"], prompt_tokens=encoded.input_ids[0].tolist(), rollouts=group))
        log(f'{label} {len(outputs)}/{len(rows)} {row["id"]} successes={sum(x["reward"] for x in group)}/8 elapsed={time.monotonic()-start:.1f}s')
        del sequences, encoded
        gc.collect()
    return outputs


def sequence_logp(model, group, rollout):
    ids = torch.tensor([group["prompt_tokens"] + rollout["tokens"]], device="cuda")
    n = len(rollout["tokens"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids[:, :-1], use_cache=False, logits_to_keep=n).logits[0]
    targets = ids[0, -n:]
    return F.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)


def record_old_logp(model, groups):
    model.eval()
    with torch.no_grad():
        for group in groups:
            for rollout in group["rollouts"]:
                rollout["old_logp"] = sequence_logp(model, group, rollout).cpu().tolist()


def reward_gradient(model, groups):
    """LOO-centered full-sequence score gradient, including zero-variance groups.

    Independent evaluation rollouts only; no length normalization. Finite Monte
    Carlo estimator, not an exact population gradient or a barrier certificate.
    """
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.zero_grad(set_to_none=True)
    for group in groups:
        rewards = np.array([x["reward"] for x in group["rollouts"]], dtype=float)
        weights = (rewards - rewards.mean()) * len(rewards)/(len(rewards)-1)
        if not np.any(weights):
            continue
        for rollout, weight in zip(group["rollouts"], weights):
            (-sequence_logp(model, group, rollout).sum() * weight / (len(groups)*len(rewards))).backward()
    result = {name: -p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    log(f'reward gradient: {len(groups)} groups; norm={sum(float(x.double().square().sum()) for x in result.values())**.5:.6g}')
    return result


def predicted_gain(model, base, gradient):
    total = 0.
    norm2 = 0.
    for name, p in model.named_parameters():
        delta = p.detach().cpu() - base[name]
        norm2 += float(delta.double().square().sum())
        if name in gradient:
            total += float((gradient[name].double()*delta.double()).sum())
    return dict(linear_reward_gain=total, parameter_step_norm=norm2**.5)


def metrics(groups):
    return dict(prompts=len(groups), responses=sum(len(g["rollouts"]) for g in groups),
                successes=sum(r["reward"] for g in groups for r in g["rollouts"]),
                mean_success=float(np.mean([np.mean([r["reward"] for r in g["rollouts"]]) for g in groups])),
                truncation_rate=float(np.mean([r["truncated"] for g in groups for r in g["rollouts"]])),
                mean_response_tokens=float(np.mean([r["length"] for g in groups for r in g["rollouts"]])))


def paired_change(before, after):
    assert [x["id"] for x in before] == [x["id"] for x in after]
    delta = np.array([np.mean([r["reward"] for r in a["rollouts"]])-np.mean([r["reward"] for r in b["rollouts"]]) for b,a in zip(before,after)])
    rng = np.random.default_rng(CONFIG["seed"])
    boots = delta[rng.integers(len(delta), size=(10000,len(delta)))].mean(axis=1)
    return dict(mean_change=float(delta.mean()), prompt_bootstrap_95_interval=np.quantile(boots,[.025,.975]).tolist(),
                per_prompt_changes=delta.tolist(), note="Descriptive prompt bootstrap; one training seed; small panel; not a confirmatory test.")


def run():
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "CUDA is required; do not silently run the NLP pilot on CPU."
    config_hash = hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
    inp = json.loads((ROOT/"gpu_pilot_inputs.json").read_text())
    output = dict(status="running", scope="Single-seed full-parameter fixed-rollout intervention pilot; not original GRPO replication or a test of persistent unlearnability.",
                  config=CONFIG, config_sha256=config_hash, dataset=inp["dataset"], dataset_revision=inp["revision"],
                  gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  versions={x:importlib.metadata.version(x) for x in ["torch","transformers","datasets","math-verify"]}, arms={})
    destination = ROOT/"gpu_pilot_results.json"
    atomic_json(destination,output)
    set_seed(CONFIG["seed"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"],revision=CONFIG["revision"],local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model"],revision=CONFIG["revision"],local_files_only=True,
                                               dtype=torch.float32,attn_implementation="sdpa").to("cuda")
    model.config.use_cache = False
    # Freeze the base before seeing any rollout or evaluation result.
    base = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    train_rows = [x for x in inp["rows"] if x["split"]=="train"]
    test_rows = [x for x in inp["rows"] if x["split"]=="test"]
    selection = sample(model,tokenizer,train_rows,CONFIG["seed"],"selection")
    output["selection_rollouts"] = selection
    mixed = [g for g in selection if 0<sum(r["reward"] for r in g["rollouts"])<len(g["rollouts"])]
    chosen = mixed[:CONFIG["selected_prompts"]]
    output["selected_ids"] = [g["id"] for g in chosen]
    output["selection_summary"] = metrics(selection)
    atomic_json(destination,output)
    if len(chosen)<4:
        output["status"]="insufficient_mixed_groups"; atomic_json(destination,output); return
    by_id = {r["id"]:r for r in train_rows}
    selected_rows = [by_id[g["id"]] for g in chosen]
    variants = [dict(r,id=r["id"]+":prefix",problem="Solve the following problem carefully.\n\n"+r["problem"]) for r in selected_rows]
    eval_rows = selected_rows + variants + test_rows
    baseline = sample(model,tokenizer,eval_rows,CONFIG["seed"]+100,"baseline")
    count = len(chosen)
    output["baseline_rollouts"] = baseline
    output["baseline_metrics"] = {"selected":metrics(baseline[:count]),"prefix_variants":metrics(baseline[count:2*count]),"heldout_math":metrics(baseline[2*count:])}
    atomic_json(destination,output)
    record_old_logp(model,chosen)
    gradients = {"selected":reward_gradient(model,baseline[:count]),"heldout_math":reward_gradient(model,baseline[2*count:])}
    experts=[]
    for group,row in zip(chosen,selected_rows):
        tokens=tokenizer.encode(row["solution"],add_special_tokens=False)+[tokenizer.eos_token_id]
        assert len(tokens)<=CONFIG["max_new_tokens"], "Expert longer than pilot cap; revise the declared protocol before training."
        experts.append(dict(id=group["id"],prompt_tokens=group["prompt_tokens"],rollouts=[dict(tokens=tokens,reward=1)]))
    for arm in CONFIG["arms"]:
        for name,p in model.named_parameters(): p.data.copy_(base[name].to("cuda"))
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=0.,foreach=False)
        history=[]; tokens_seen=0; arm_start=time.monotonic()
        for epoch in range(CONFIG["epochs"]):
            order=list(range(count)); random.Random(CONFIG["seed"]+epoch).shuffle(order)
            for j in order:
                assert time.monotonic()-start<CONFIG["timeout_seconds"], "Predeclared hard timeout reached."
                group=experts[j] if arm=="expert_sft" else chosen[j]
                rollouts=group["rollouts"]
                if arm=="student_success_sft": rollouts=[r for r in rollouts if r["reward"]]
                optimizer.zero_grad(set_to_none=True)
                losses=[]
                rewards=np.array([r["reward"] for r in group["rollouts"]],dtype=float)
                for r in rollouts:
                    lp=sequence_logp(model,group,r)
                    if arm=="clipped_grpo_replay":
                        advantage=(r["reward"]-rewards.mean())/(rewards.std()+1e-6)
                        old=torch.tensor(r["old_logp"],device="cuda")
                        ratio=(lp-old).exp()
                        loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                    else: loss=-lp.mean()
                    (loss/len(rollouts)).backward()
                    losses.append(float(loss.detach())); tokens_seen+=len(r["tokens"])
                grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG["grad_clip"]))
                optimizer.step()
                history.append(dict(epoch=epoch,group=group["id"],loss=float(np.mean(losses)),gradient_norm_before_clip=grad_norm))
                if len(history)==1:
                    first={k:predicted_gain(model,base,h) for k,h in gradients.items()}
                log(f'{arm} update={len(history)}/{count*CONFIG["epochs"]} loss={np.mean(losses):.5f} vram_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')
        prediction={k:predicted_gain(model,base,h) for k,h in gradients.items()}
        del optimizer
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        after=sample(model,tokenizer,eval_rows,CONFIG["seed"]+100,arm+" evaluation")
        result=dict(training_history=history,response_tokens_trained=tokens_seen,wall_seconds=time.monotonic()-arm_start,
                    first_step_linear_predictions=first,final_linear_predictions=prediction,rollouts=after,
                    metrics={"selected":metrics(after[:count]),"prefix_variants":metrics(after[count:2*count]),"heldout_math":metrics(after[2*count:])},
                    changes={"selected":paired_change(baseline[:count],after[:count]),"prefix_variants":paired_change(baseline[count:2*count],after[count:2*count]),"heldout_math":paired_change(baseline[2*count:],after[2*count:])})
        output["arms"][arm]=result;atomic_json(destination,output)
    output.update(status="completed",wall_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(destination,output)
    log(f'COMPLETED {destination} elapsed={output["wall_seconds"]:.1f}s')


def make_kl_probe(model, groups):
    """One uniform response position per independent base-policy trajectory."""
    model.eval();model.gradient_checkpointing_disable()
    rng=np.random.default_rng(CONFIG["seed"]+9000)
    probes=[]
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for group in groups:
            for response in group["rollouts"]:
                length=len(response["tokens"]);position=int(rng.integers(length))
                prefix=group["prompt_tokens"]+response["tokens"][:position]
                lp=F.log_softmax(model(torch.tensor([prefix],device="cuda"),use_cache=False,logits_to_keep=1).logits[0,0].float(),dim=-1)
                probes.append(dict(id=group["id"],prefix=prefix,length=length,old_logp=lp.cpu()))
    return probes


def measure_kl_probe(model, probes):
    """Monte Carlo estimate of forward sequence KL, not a hard KL bound.

    The full-vocabulary conditional KL is exact at each sampled prefix. Length
    weighting makes one uniformly chosen position estimate the trajectory sum.
    """
    model.eval();model.gradient_checkpointing_disable()
    values={}
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for p in probes:
            lp=F.log_softmax(model(torch.tensor([p["prefix"]],device="cuda"),use_cache=False,logits_to_keep=1).logits[0,0].float(),dim=-1)
            old=p["old_logp"].to("cuda")
            value=float((old.exp()*(old-lp)).sum())*p["length"]
            values.setdefault(p["id"],[]).append(value)
    means=np.array([np.mean(x) for x in values.values()])
    return dict(estimated_forward_sequence_kl_nats=float(means.mean()),
                prompt_means=values,prompts=len(values),sampled_prefixes=len(probes),
                method="Exact full-vocabulary conditional KL at one uniform position per base response, weighted by response length; finite Monte Carlo diagnostic, not an upper bound.")


def run_onpolicy(learning_rate=None):
    """Additional baseline declared during initial sampling, before arm results.

    One first-order group-relative update per fresh group. No stale replay or
    multiple PPO epochs. This is a small single-group training recipe, not DAPO.
    """
    start=time.monotonic()
    rate=CONFIG["learning_rate"] if learning_rate is None else learning_rate
    arm_name="on_policy_grpo" if rate==CONFIG["learning_rate"] else f"on_policy_grpo_lr_{rate:g}"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=True
    destination=ROOT/"gpu_pilot_results.json"
    output=json.loads(destination.read_text())
    assert output["status"]=="completed", "Finish the original three-arm pilot first."
    assert arm_name not in output["arms"], "This result already exists; do not overwrite an experimental arm."
    protocol=json.loads((ROOT/"gpu_pilot_protocol.json").read_text())
    output["additional_baseline_protocol"]=protocol
    output["status"]="on_policy_running"
    atomic_json(destination,output)
    tokenizer=AutoTokenizer.from_pretrained(CONFIG["model"],revision=CONFIG["revision"],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(CONFIG["model"],revision=CONFIG["revision"],local_files_only=True,
                                             dtype=torch.float32,attn_implementation="sdpa").to("cuda")
    model.config.use_cache=False
    base={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    inputs=json.loads((ROOT/"gpu_pilot_inputs.json").read_text())
    by_id={r["id"]:r for r in inputs["rows"]}
    rows=[by_id[k] for k in output["selected_ids"]];count=len(rows)
    variants=[dict(r,id=r["id"]+":prefix",problem="Solve the following problem carefully.\n\n"+r["problem"]) for r in rows]
    eval_rows=rows+variants+[r for r in inputs["rows"] if r["split"]=="test"]
    baseline=output["baseline_rollouts"]
    kl_probes=make_kl_probe(model,baseline[2*count:])
    gradients={"selected":reward_gradient(model,baseline[:count]),"heldout_math":reward_gradient(model,baseline[2*count:])}
    optimizer=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=0.,foreach=False)
    history=[];generated=[];trained_tokens=0;sampled_tokens=0;first=None;first_kl=None
    for epoch in range(CONFIG["epochs"]):
        order=list(range(count));random.Random(CONFIG["seed"]+epoch).shuffle(order)
        for j in order:
            assert time.monotonic()-start<CONFIG["timeout_seconds"], "Hard timeout reached."
            model.zero_grad(set_to_none=True)
            group=sample(model,tokenizer,[rows[j]],CONFIG["seed"]+5000+len(history),"on-policy training")[0]
            generated.append(group)
            rewards=np.array([r["reward"] for r in group["rollouts"]],dtype=float)
            sampled_tokens+=sum(r["length"] for r in group["rollouts"])
            record=dict(epoch=epoch,group=group["id"],successes=int(rewards.sum()),update_applied=False)
            if rewards.std()>0:
                model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
                losses=[]
                for r in group["rollouts"]:
                    advantage=(r["reward"]-rewards.mean())/(rewards.std()+1e-6)
                    loss=-advantage*sequence_logp(model,group,r).mean()
                    (loss/len(rewards)).backward()
                    losses.append(float(loss.detach()));trained_tokens+=len(r["tokens"])
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG["grad_clip"]))
                optimizer.step()
                record.update(update_applied=True,loss=float(np.mean(losses)),gradient_norm_before_clip=norm)
                if first is None:
                    first={k:predicted_gain(model,base,h) for k,h in gradients.items()}
                    first_kl=measure_kl_probe(model,kl_probes)
            history.append(record)
            log(f'{arm_name} attempt={len(history)}/{count*CONFIG["epochs"]} updated={record["update_applied"]} peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB')
    prediction={k:predicted_gain(model,base,h) for k,h in gradients.items()}
    del optimizer
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    final_kl=measure_kl_probe(model,kl_probes)
    after=sample(model,tokenizer,eval_rows,CONFIG["seed"]+100,arm_name+" evaluation")
    result=dict(learning_rate=rate,training_history=history,training_rollouts=generated,response_tokens_trained=trained_tokens,
                first_step_policy_kl_probe=first_kl,final_policy_kl_probe=final_kl,
                additional_response_tokens_sampled=sampled_tokens,wall_seconds=time.monotonic()-start,
                first_step_linear_predictions=first,final_linear_predictions=prediction,rollouts=after,
                updates_applied=sum(r["update_applied"] for r in history),
                metrics={"selected":metrics(after[:count]),"prefix_variants":metrics(after[count:2*count]),"heldout_math":metrics(after[2*count:])},
                changes={"selected":paired_change(baseline[:count],after[:count]),"prefix_variants":paired_change(baseline[count:2*count],after[count:2*count]),"heldout_math":paired_change(baseline[2*count:],after[2*count:])})
    output["arms"][arm_name]=result
    output["status"]="completed"
    output["additional_baseline_wall_seconds"]=output.get("additional_baseline_wall_seconds",0)+time.monotonic()-start
    output["peak_gpu_allocated_bytes"]=max(output["peak_gpu_allocated_bytes"],torch.cuda.max_memory_allocated())
    atomic_json(destination,output)
    log("ON_POLICY_COMPLETED")


def analyze():
    """Summarize the retained evidence without rerunning or tuning training."""
    path=ROOT/"gpu_pilot_results.json"
    r=json.loads(path.read_text())
    assert r["status"]=="completed" and "on_policy_grpo" in r["arms"]
    inputs=json.loads((ROOT/"gpu_pilot_inputs.json").read_text())
    count=len(r["selected_ids"])
    labels={"clipped_grpo_replay":"Fixed-rollout clipped replay", "student_success_sft":"Student-correct SFT",
            "expert_sft":"Expert SFT", "on_policy_grpo":"Fresh group-relative PG, LR 5e-6",
            "on_policy_grpo_lr_5e-07":"Fresh group-relative PG, LR 5e-7"}
    expected_ids=[g["id"] for g in r["baseline_rollouts"]]
    assert len(expected_ids)==2*count+16 and len(set(expected_ids))==len(expected_ids)
    for value in r["arms"].values():
        assert [g["id"] for g in value["rollouts"]]==expected_ids
        assert all(len(g["rollouts"])==CONFIG["group_size"] for g in value["rollouts"])
    initial={g["id"]:g for g in r["selection_rollouts"]}
    low=[i for i,k in enumerate(r["selected_ids"]) if 1<=sum(x["reward"] for x in initial[k]["rollouts"])<=2]
    rest=[i for i in range(count) if i not in low]
    strata={"1-2_of_8_selection_successes":low,"3-7_of_8_selection_successes":rest}
    details={}
    for name,indices in strata.items():
        b=[r["baseline_rollouts"][i] for i in indices]
        details[name]={"ids":[r["selected_ids"][i] for i in indices],"baseline":metrics(b),"arms":{}}
        for arm,value in r["arms"].items():
            a=[value["rollouts"][i] for i in indices]
            details[name]["arms"][arm]={"metrics":metrics(a),"change":paired_change(b,a)}
    normalization=lambda x:" ".join(x.split()).lower()
    train={normalization(x["problem"]) for x in inputs["rows"] if x["split"]=="train"}
    test={normalization(x["problem"]) for x in inputs["rows"] if x["split"]=="test"}
    format_diagnostics={}
    for label,groups in [("baseline",r["baseline_rollouts"])]+[(k,v["rollouts"]) for k,v in r["arms"].items()]:
        format_diagnostics[label]={}
        for panel,part in [("selected",groups[:count]),("prefix_variants",groups[count:2*count]),("heldout_math",groups[2*count:])]:
            responses=[x for group in part for x in group["rollouts"]]
            boxed=sum("\\boxed" in x["text"] for x in responses)
            format_diagnostics[label][panel]={"boxed_answer_rate":boxed/len(responses),
                 "verified_success_among_boxed":sum(x["reward"] for x in responses)/boxed if boxed else None,
                 "truncation_rate":float(np.mean([x["truncated"] for x in responses])),
                 "mean_response_tokens":float(np.mean([x["length"] for x in responses]))}
    r["analysis"]={"observed_rate_strata":details,"format_and_length_diagnostics":format_diagnostics,
                   "normalized_train_test_overlap_in_fixed_inputs":len(train & test),
                   "raw_rollout_evidence_retained":True,
                   "input_sha256":hashlib.sha256((ROOT/"gpu_pilot_inputs.json").read_bytes()).hexdigest(),
                   "analysis_implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "frozen_evaluation_panel_integrity":"All arms retain the same 32 prompt IDs and eight responses per ID",
                   "verification_status":"ANALYZED; not an independent reproducibility rerun",
                   "limitations":["One training seed and short interventions", "Small prompt panels and nonsimultaneous descriptive intervals", "Selected prompts are not established unlearnable prompts", "Prefix variants do not establish new-skill transfer", "Different supervision and token budgets", "No GAR, SRL, natural-gradient or mid-training comparison", "No representation-necessity theorem tested"]}
    atomic_json(path,r)
    lines=["## 11. Completed RTX 5060 Ti experiment", "",
           "### Material Passport", "",
           "- Evidence: locally generated full-parameter Qwen2.5-0.5B pilot on MATH; raw responses, token IDs and rewards retained in [one result](gpu_pilot_results.json).",
           "- Verification status: **ANALYZED**. The script completed; no independent training rerun has been performed.",
           "- Model/data revisions, package versions, seeds and configuration are recorded in the result. The [additional on-policy control](gpu_pilot_protocol.json) was declared during initial-policy baseline sampling, before intervention outcomes.", "",
           "### Outcome table", "",
           "Each cell is verified success before → after, followed by the absolute change and a descriptive 95% prompt-bootstrap interval. Intervals are not simultaneous, and one training seed does not quantify training-run variability.", "",
           "| Intervention | Selected prompts (8) | Same-problem prefix variants (8) | Held-out MATH problems (16) | Training response tokens |", "| --- | --- | --- | --- | --- |"]
    for arm,value in r["arms"].items():
        cells=[]
        for panel in ["selected","prefix_variants","heldout_math"]:
            b=r["baseline_metrics"][panel]["mean_success"]*100
            a=value["metrics"][panel]["mean_success"]*100
            d=value["changes"][panel]
            lo,hi=np.array(d["prompt_bootstrap_95_interval"])*100
            cells.append(f'{b:.2f}% → {a:.2f}%; {100*d["mean_change"]:+.1f} pp [{lo:+.1f}, {hi:+.1f}]')
        lines.append("| "+" | ".join([labels.get(arm,arm),*cells,f'{value["response_tokens_trained"]:,}'])+" |")
    lines += ["", "### Low observed-success subset", "",
              "This split was declared before intervention results. It uses noisy selection counts, not a validated latent-difficulty label. Independent baseline outcomes are not used to relabel the strata.", "",
              "| Selection stratum | Prompts | Independent baseline success | Intervention | Final success | Change |", "| --- | --- | --- | --- | --- | --- |"]
    for name,d in details.items():
        for arm,value in d["arms"].items():
            lines.append(f'| {name} | {len(d["ids"])} | {100*d["baseline"]["mean_success"]:.1f}% | {labels.get(arm,arm)} | {100*value["metrics"]["mean_success"]:.1f}% | {100*value["change"]["mean_change"]:+.1f} pp |')
    lines += ["", "### Formatting and truncation check", "",
              "Our verifier requires a boxed answer. Training changes formatting and response length, so success gains cannot automatically be attributed to better reasoning. Conditional accuracy below is descriptive; conditioning on formatting does not identify a causal reasoning effect.", "",
              "| Method | Held-out boxed-answer rate | Held-out success among boxed responses | Held-out truncation | Mean held-out response tokens |", "| --- | --- | --- | --- | --- |"]
    for label,panels in format_diagnostics.items():
        d=panels["heldout_math"]
        c="undefined" if d["verified_success_among_boxed"] is None else f'{100*d["verified_success_among_boxed"]:.1f}%'
        lines.append(f'| {labels.get(label,label)} | {100*d["boxed_answer_rate"]:.1f}% | {c} | {100*d["truncation_rate"]:.1f}% | {d["mean_response_tokens"]:.0f} |')
    lines += ["", "### Gradient prediction and finite updates", "",
              "Full-sequence reward-gradient estimates were computed from independent baseline rollouts. The following compares their linear predictions along each realized final parameter change with the independently sampled outcome change. The gradients themselves are noisy Monte Carlo estimates; no confidence bound for their predictions was established. Final updates are not constrained to the neighborhood required by the Taylor bound. These figures therefore do not validate a representation barrier or a finite-step guarantee.", "",
              "| Intervention | Predicted selected gain | Observed selected change | Predicted held-out gain | Observed held-out change |", "| --- | --- | --- | --- | --- |"]
    for arm,v in r["arms"].items():
        p=v["final_linear_predictions"]
        lines.append(f'| {labels.get(arm,arm)} | {100*p["selected"]["linear_reward_gain"]:+.1f} pp | {100*v["changes"]["selected"]["mean_change"]:+.1f} pp | {100*p["heldout_math"]["linear_reward_gain"]:+.1f} pp | {100*v["changes"]["heldout_math"]["mean_change"]:+.1f} pp |')
    lines += ["", "### Execution and interpretation boundaries", "",
              f'- Peak allocated PyTorch GPU memory: **{r["peak_gpu_allocated_bytes"]/2**30:.2f} GiB**. This excludes some driver/display/reserved memory.',
              f'- Original pilot wall time: {r["wall_seconds"]/60:.1f} minutes; additional fresh-rollout controls: {r["additional_baseline_wall_seconds"]/60:.1f} minutes.',
              '- Fresh-rollout reward-driven updates / sampled groups: '+ '; '.join(f'{labels.get(k,k)}: {v["updates_applied"]}/{len(v["training_history"])}' for k,v in r["arms"].items() if "updates_applied" in v)+'. Zero-variance groups were retained in logs and skipped for optimization.',
              f'- Normalized text overlap between sampled training and test inputs: {len(train & test)}. Pretraining contamination of MATH is not ruled out.',
              '- The fixed-data replay arm and fresh-rollout arm are small group-relative training implementations, with no KL term. Neither is a reproduction of the original distributed DAPO configuration.',
              '- The [published scorer](https://raw.githubusercontent.com/yulinchen99/unlearnability-rlvr/main/verl/utils/reward_score/hf_math_verify.py) has a default mode that can accept unboxed extracted answers; our boxed-answer requirement is a deliberate stricter pilot choice. Answer-level verification does not certify intermediate reasoning. This is another reason the pilot is not a source replication.',
              '- Truncation, generated lengths, trained tokens and per-prompt outcomes are retained. Differences in information, trajectory length and optimization exposure prevent claiming a controlled algorithm ranking.',
              '- Existing strong repair baselines remain outstanding. This experiment can expose a short-horizon pattern and verify feasibility; it cannot establish persistent unlearnability or the necessity of a new method.', "",
              "### Statistical self-review: 11/11 checks", "",
              "| Check | Disposition |", "| --- | --- |",
              "| Simpson reversal | Observed-rate strata reported alongside the pooled panel; a broad homogeneous effect is not assumed. |",
              "| Ecological inference | Aggregate success is not attributed to every prompt or every model. |",
              "| Berkson selection | Mixed-reward selection is disclosed; the affected population is not a random hard-prompt sample. |",
              "| Collider conditioning | No causal representation explanation is inferred by conditioning on successful rollouts. |",
              "| Base rates | Selection, baseline rates and all-failure groups are retained. |",
              "| Regression to the mean | Improvement is measured against an independent baseline, not the selection rollouts. |",
              "| Survivorship | Frozen evaluation panel retained; optimizer skips are logged. |",
              "| Look-elsewhere | Every arm/panel is reported; intervals are descriptive and unadjusted. |",
              "| Forking paths | Configuration and added control are disclosed; this remains exploratory. |",
              "| Causation | Interventions change training recipes; unequal information/cost and one seed preclude a representation-mechanism claim. |",
              "| Reverse causality | Updates precede endpoint evaluation; gradient association is not used as proof of a learning barrier. |", ""]
    kl_rows=[]
    for arm,value in r["arms"].items():
        if "final_policy_kl_probe" in value:
            first=value.get("first_step_policy_kl_probe")
            last=value["final_policy_kl_probe"]
            first_value="not available" if first is None else f'{first["estimated_forward_sequence_kl_nats"]:.6f}'
            kl_rows.append(f'| {labels.get(arm,arm)} | {first_value} | {last["estimated_forward_sequence_kl_nats"]:.6f} |')
    if kl_rows:
        lines += ["### Policy-change diagnostic for the source-rate control", "",
                  "The source-rate control additionally measures full-vocabulary conditional KL at one uniformly sampled response position per independent base-policy trajectory. Weighting by that trajectory's response length estimates the chain-rule sequence KL sum:", "",
                  r"$$",r"\widehat D=\frac{1}{n}\sum_{i=1}^{n}L_i\,D_{\mathrm{KL}}\!\left(\pi_{\theta_0}(\cdot\mid h_{i,U_i})\,\|\,\pi_{\theta_1}(\cdot\mid h_{i,U_i})\right),\qquad U_i\sim\mathrm{Unif}\{1,\ldots,L_i\}.",r"$$", "",
                  "This is a classical chain-rule Monte Carlo diagnostic, not a new result or an upper bound. It uses 128 baseline trajectories across 16 held-out prompts. The additional probe was added after the first fresh-rollout run started, so it is available for the source-rate control only. Mixed-precision computation and finite gradient samples also affect first-order comparisons. No Taylor remainder constant was established.", "",
                  "| Control | Estimated sequence KL after first applied update (nats) | Estimated sequence KL at endpoint (nats) |","| --- | --- | --- |",*kl_rows,""]
    lines += ["### Research decision after the pilot", "",
              "**Do not commit to a new methods or theory paper on this evidence.** Ordinary student-correct SFT raised selected-prompt success from 43.75% to 81.25%. In the four low observed-success cases it raised the independent baseline from 21.875% to 71.875%; expert SFT reached 81.25%. The held-out student-SFT gain was only 3.9 points, with a descriptive interval spanning zero. Its conditional success among boxed responses did not improve. These results support neither a demonstrated representation barrier nor a broadly effective new repair.", "",
              "The sampled mixed-reward cohort is not the original paper's persistent, threshold-defined cohort. Even our low observed-rate subset has an independent baseline of 21.875%, above the original hard threshold. Thus this experiment does not refute the published phenomenon or establish that SFT repairs the original cases. It establishes that this pilot has not reproduced the necessary residual failure.", "",
              "The uncalibrated gradient forecasts also give wrong signs or gains outside the physically possible probability range in several arms. This does not falsify the conditional Taylor theorem: gradient-estimation error, update size and mixed precision were uncontrolled for that purpose. It shows that the proposed point diagnostic is not a validated practical certificate. The underlying identities remain classical foundations rather than a new theoretical contribution.", "",
              "The next scientific requirement, if this question is retained, is a fixed-cohort reproduction of the published failure with the original response cap, independent evaluation and multiple training seeds, followed by the strongest current repairs. Until a consequential residual survives, developing another weighting, geometric score or optimizer is not justified. The present candidate should remain unendorsed.", ""]
    report=ROOT/"research_report.md"
    old=report.read_text().split("## 11. Completed RTX 5060 Ti experiment")[0].rstrip()
    report.write_text(old+"\n\n"+"\n".join(lines)+"\n")
    log("ANALYSIS_SAVED")


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--on-policy",action="store_true")
    args.add_argument("--analyze",action="store_true")
    args.add_argument("--learning-rate",type=float)
    options=args.parse_args()
    if options.analyze: analyze()
    elif options.on_policy: run_onpolicy(options.learning_rate)
    else: run()
