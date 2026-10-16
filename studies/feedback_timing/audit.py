#!/usr/bin/env python3
"""Bounded public-data timing audit; no human recruitment or LM training.

Requires NumPy and SciPy. Raw public data stays in the ignored .cache directory.
The target is stated clear/slight preference, NOT latent utility or correctness.
"""
import hashlib
import json
import platform
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REVISION = "12910233a0238a997ebe425656e9dfed7b0ff031"
SHA256 = "027072c52e39300c55ebc34265dda5ef2cfb1682edbc22b1a7d978c94776aaca"
URL = f"https://huggingface.co/datasets/allenai/multipref/resolve/{REVISION}/MultiPref-Train.jsonl"
SEEDS = [20261016, 20261017, 20261018, 20261019, 20261020]


def fetch():
    path = ROOT / ".cache/feedback_timing/MultiPref-Train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print("Download pinned 64.8 MB public MultiPref data", flush=True)
        temp = path.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=60) as response, temp.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temp.replace(path)
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256, "Dataset hash mismatch"
    return [json.loads(line) for line in raw.splitlines()]


def extract(items):
    rows, counts, fields = [], Counter(), set()
    durations = []
    for item in items:
        # Mirrors response-pair lengths: prompt appears in both conversations.
        length = 2 * len(item["text"]) + len(item["completion_a"]) + len(item["completion_b"])
        la, lb = len(item["completion_a"]), len(item["completion_b"])
        for cohort, key in enumerate(["normal_worker_annotations", "expert_worker_annotations"]):
            for ann in item[key]:
                fields.update(ann)
                pref = ann["overall_pref"]
                counts[pref] += 1
                duration = ann.get("time_spent")
                if isinstance(duration, (int, float)) and duration > 0:
                    durations.append(duration)
                if pref == "Tie" or not isinstance(duration, (int, float)) or duration <= 0:
                    continue
                assert "clearly" in pref or "slightly" in pref
                y = int("clearly" in pref)
                rows.append({
                    "prompt": item["prompt_id"], "user": ann["evaluator"],
                    "y": y, "lt": float(np.log(duration)), "length": length,
                    "meta": [float(np.log1p(length)), float(np.log1p(min(la, lb))),
                             float(np.log1p(abs(la-lb))), float(cohort)],
                })
    return rows, {
        "comparisons": len(items), "annotations": sum(counts.values()),
        "preference_counts": dict(counts), "eligible_non_ties": len(rows),
        "unique_prompts": len({r["prompt"] for r in rows}),
        "unique_annotators": len({r["user"] for r in rows}),
        "clear_fraction": float(np.mean([r["y"] for r in rows])),
        "annotation_fields": sorted(fields),
        "duration_quantiles_native_units": dict(zip(["min", "p25", "median", "p75", "p95", "max"],
                                                   map(float, np.quantile(durations, [0, .25, .5, .75, .95, 1])))),
        "isolated_choice_timestamp_available": False,
        "timing_measure": "time_spent, total annotation duration; timestamp is one annotation-level field",
        "target": "self-reported clear versus slight overall preference; excludes ties",
    }


def auc(y, score):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return None
    return float((rankdata(score)[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))


def losses(y, prob):
    p = np.clip(prob, 1e-12, 1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def fit_logistic(x, y):
    # Fixed ridge penalty, no test-set hyperparameter selection.
    reg = np.full(x.shape[1], .001); reg[0] = 0
    def objective(w):
        z = x @ w
        value = np.mean(np.logaddexp(0, z) - y*z) + .5*np.sum(reg*w*w)
        gradient = x.T @ (expit(z)-y) / len(y) + reg*w
        return value, gradient
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 250, "gtol": 1e-7, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"Logistic fit did not converge: {result.message}")
    return result.x


def standardized(train, test):
    mean = train.mean(axis=0); sd = train.std(axis=0)
    sd[sd < 1e-8] = 1
    return (train-mean)/sd, (test-mean)/sd


def cluster_ci(values, groups, seed, repetitions=1000):
    # Resample held-out prompts, keeping their annotations together.
    unique, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values); sizes = np.bincount(inv)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sample = rng.integers(len(unique), size=len(unique))
        estimates.append(float(sums[sample].sum()/sizes[sample].sum()))
    return list(map(float, np.quantile(estimates, [.025, .975])))


def evaluate(rows, mode, seed):
    rng = np.random.default_rng(seed)
    split_key = "prompt" if mode == "held_out_prompts" else "user"
    unique = sorted({r[split_key] for r in rows})
    rng.shuffle(unique)
    heldout = set(unique[:max(1, len(unique)//5)])
    tr = np.array([i for i,r in enumerate(rows) if r[split_key] not in heldout])
    te = np.array([i for i,r in enumerate(rows) if r[split_key] in heldout])
    y = np.array([r["y"] for r in rows], dtype=float)
    lt = np.array([r["lt"] for r in rows])
    meta = np.array([r["meta"] for r in rows])
    # Fixed low-order nonlinearities allow ordinary length adjustment.
    meta = np.column_stack([meta, meta[:, :3]**2])
    mt, me = standardized(meta[tr], meta[te])
    intercept_t, intercept_e = np.ones((len(tr),1)), np.ones((len(te),1))
    xt, xe = np.column_stack([intercept_t, mt]), np.column_stack([intercept_e, me])
    # Piecewise-linear timing model with training-only knots and scaling.
    knots = np.quantile(lt[tr], [.1, .25, .5, .75, .9])
    rt = np.column_stack([lt, *[np.maximum(lt-k, 0) for k in knots]])
    rt_t, rt_e = standardized(rt[tr], rt[te])
    models = {"length_cohort": (xt, xe),
              "length_cohort_plus_flexible_time": (np.column_stack([xt,rt_t]), np.column_stack([xe,rt_e]))}
    # Conventional per-annotator / 8 length-bucket calibration, train only.
    edges = np.quantile([rows[i]["length"] for i in tr], np.linspace(0,1,9))[1:-1]
    bins = np.searchsorted(edges, [r["length"] for r in rows], side="right")
    by_user, by_stratum = defaultdict(list), defaultdict(list)
    clear_by_user = defaultdict(list)
    for i in tr:
        u = rows[i]["user"]
        by_user[u].append(lt[i]); by_stratum[(u,int(bins[i]))].append(lt[i])
        clear_by_user[u].append(y[i])
    global_mean = float(lt[tr].mean()); global_p = float(y[tr].mean())
    # Leave each training row out of its own calibration statistics. Test rows
    # use training statistics only; unseen users receive global fallbacks.
    training = set(tr)
    centered, prior = [], []
    for i, r in enumerate(rows):
        own = int(i in training)
        pool = by_stratum.get((r["user"], int(bins[i])), [])
        if len(pool) <= own:
            pool = by_user.get(r["user"], [])
        center = ((sum(pool)-own*lt[i])/(len(pool)-own)
                  if len(pool) > own else
                  (lt[tr].sum()-own*lt[i])/(len(tr)-own))
        centered.append(lt[i]-center)
        labels = clear_by_user.get(r["user"], [])
        prior.append((sum(labels)-own*y[i]+10*global_p)/(len(labels)-own+10))
    centered, prior = np.array(centered), np.array(prior)
    cal = np.column_stack([prior, centered, centered**2])
    ct, ce = standardized(cal[tr],cal[te])
    models["calibrated_metadata"] = (np.column_stack([xt, ct[:,:1]]), np.column_stack([xe, ce[:,:1]]))
    models["calibrated_metadata_plus_time"] = (np.column_stack([xt,ct]), np.column_stack([xe,ce]))
    result = {"mode":mode,"seed":seed,"train_n":len(tr),"test_n":len(te),
              "raw_shorter_means_clear_auc":auc(y[te],-lt[te]),
              "normalized_shorter_means_clear_auc":auc(y[te],-centered[te]),"models":{}}
    saved = {}
    for name,(train_x,test_x) in models.items():
        w=fit_logistic(train_x,y[tr]); prob=expit(test_x@w)
        l=losses(y[te],prob);saved[name]=l
        result["models"][name]={"auroc":auc(y[te],prob),"log_loss_nats":float(l.mean()),
                                "brier":float(np.mean((y[te]-prob)**2))}
    result["timing_gains"]={}
    for base,extended in [("length_cohort","length_cohort_plus_flexible_time"),
                          ("calibrated_metadata","calibrated_metadata_plus_time")]:
        delta=saved[base]-saved[extended]
        result["timing_gains"][extended]={"log_loss_reduction_nats":float(delta.mean()),
            "prompt_bootstrap_95ci":cluster_ci(delta,np.array([rows[i]["prompt"] for i in te]),seed)}
    return result


def main():
    started=time.monotonic()
    print("Start bounded CPU timing audit; target is stated strength, not policy utility",flush=True)
    items=fetch(); rows,inventory=extract(items)
    print(f"Data: {inventory['annotations']} annotations, {len(rows)} eligible non-ties",flush=True)
    runs=[]
    for mode in ["held_out_prompts","held_out_annotators"]:
        for seed in SEEDS:
            run=evaluate(rows,mode,seed);runs.append(run)
            gains={k:round(v['log_loss_reduction_nats'],6) for k,v in run['timing_gains'].items()}
            print(f"{mode} seed={seed} log-loss reductions={gains}",flush=True)
    summary={}
    for mode in ["held_out_prompts","held_out_annotators"]:
        sub=[r for r in runs if r['mode']==mode]
        summary[mode]={
            "mean_raw_timing_auc":float(np.mean([r['raw_shorter_means_clear_auc'] for r in sub])),
            "mean_normalized_timing_auc":float(np.mean([r['normalized_shorter_means_clear_auc'] for r in sub])),
            "mean_model_metrics":{name:{metric:float(np.mean([r['models'][name][metric] for r in sub]))
                                          for metric in ['auroc','log_loss_nats','brier']}
                                  for name in sub[0]['models']},
            "mean_timing_log_loss_reductions":{name:float(np.mean([r['timing_gains'][name]['log_loss_reduction_nats'] for r in sub]))
                                                for name in sub[0]['timing_gains']}}
    output={"material_passport":{"mode":"run","type":"public-data observational audit",
             "status":"COMPLETED","created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},
            "source":{"url":URL,"revision":REVISION,"sha256":SHA256},
            "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "protocol":{"seeds":SEEDS,"holdout_fraction":.2,"ridge_penalty":.001,
                        "bootstrap_repetitions":1000,"split_uncertainty":"Per-split CIs, not independent meta-analysis across overlapping splits",
                        "bootstrap_scope":"Prompt clusters only; conditional on fitted model, not full two-way annotator/prompt uncertainty",
                        "calibration":"Training rows leave themselves out; test rows use training-only statistics",
                        "no_lm_training":True,"no_latent_utility_labels":True,
                        "held_out_annotators_caveat":"Prompts may overlap; annotator transfer test, not joint cold-start test"},
            "inventory":inventory,"summary":summary,"runs":runs,"elapsed_seconds":time.monotonic()-started}
    path=HERE/'results.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Saved {path}; elapsed {output['elapsed_seconds']:.1f}s",flush=True)


if __name__=='__main__':
    main()
