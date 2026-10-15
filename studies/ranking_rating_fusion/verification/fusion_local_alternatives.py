"""Prespecified, paired sufficient-statistic check of the T2 local alternatives.

No model training. Run with the probml Python environment (NumPy required).
Writes one canonical cell-results CSV; raw draws can be regenerated from the configuration.
"""
from pathlib import Path
from fractions import Fraction as F
import csv
import json
import hashlib
import math
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'fusion_local_alternatives_config.json').read_text())

FTR = np.array(CONFIG["features"])
RR = np.array(CONFIG["reward_direction"])
EDGES = [(i, j) for i in range(3) for j in range(i + 1, 3)]
DD = np.array([FTR[i] - FTR[j] for i, j in EDGES])
GG = np.array([RR[i] - RR[j] for i, j in EDGES])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def fitted(counts, successes, gap_sum, lam):
    """Vectorized monotone bisection, including constrained boundary solutions."""
    low = np.full(len(counts), CONFIG["parameter_interval"][0])
    high = np.full(len(counts), CONFIG["parameter_interval"][1])
    def score(theta):
        zz = theta[:, None] * DD[None, :]
        return np.sum(DD * (counts * sigmoid(zz) - successes), axis=1) + lam * (
            theta * np.sum(counts * DD**2, axis=1) - np.sum(gap_sum * DD, axis=1))
    at_lo, at_hi = score(low), score(high)
    left_boundary, right_boundary = at_lo >= 0, at_hi <= 0
    for _ in range(55):
        mid = (low + high) / 2
        positive = score(mid) > 0
        high = np.where(positive, mid, high)
        low = np.where(positive, low, mid)
    ans = (low + high) / 2
    ans[left_boundary] = CONFIG["parameter_interval"][0]
    ans[right_boundary] = CONFIG["parameter_interval"][1]
    return ans


def value(theta, epsilon):
    logits = np.asarray(theta)[..., None] * FTR
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    pp = np.exp(logits)
    pp = pp / np.sum(pp, axis=-1, keepdims=True)
    return np.sum(pp * (epsilon * RR), axis=-1) - np.sum(pp * np.log(3 * pp), axis=-1)


def coefficients(epsilon, lam, noise_variance=0.0):
    g = epsilon * GG
    theta = float(fitted(np.ones((1, 3)), sigmoid(g)[None, :], g[None, :], lam)[0])
    s, p = sigmoid(theta * DD), sigmoid(g)
    h, third = s * (1-s), s * (1-s) * (1-2*s)
    m = DD * ((s-p) + lam*(theta*DD-g))
    kobs = DD**2 * (h+lam)
    H, T = float(kobs.mean()), float((DD**3 * third).mean())
    C = float((m*m + DD**2*(p*(1-p)+lam**2*noise_variance)).mean())
    D = float((m*kobs).mean())
    bias = D/H**2 - T*C/(2*H**3)
    pp = np.exp(theta * FTR)
    pp /= pp.sum()
    fc, rc = FTR-(pp*FTR).sum(), epsilon*RR-(pp*epsilon*RR).sum()
    variance_f = float((pp*fc*fc).sum())
    jp = float((pp*fc*rc).sum() - theta*variance_f)
    jpp = float((pp*fc*fc*rc).sum() - variance_f - theta*(pp*fc**3).sum())
    V = -jp*bias - jpp*C/(2*H**2)
    return {"epsilon": epsilon, "lambda": lam, "theta": theta, "J": float(value(theta, epsilon)),
            "H": H, "T": T, "C": C, "D": D, "bias": bias, "Jprime": jp, "Jsecond": jpp, "V": V}


def exact_constants():
    f, r = [F(0), F(7, 10), F(1)], [F(0), F(1), F(2, 5)]
    edges = [(f[i]-f[j], r[i]-r[j]) for i, j in EDGES]
    K = sum(d*d for d, g in edges)/3
    k = sum(d*g for d, g in edges)/(3*K)
    b3 = (k**3*sum(d**4 for d, g in edges)/3-sum(d*g**3 for d, g in edges)/3)/(12*K)
    mf, mr = sum(f)/3, sum(r)/3
    ff_r = sum((x-mf)**2*(y-mr) for x, y in zip(f, r))/3
    fff = sum((x-mf)**3 for x in f)/3
    c2 = k*ff_r-k*k*fff
    lam = F(1, 10)
    P = c2*b3*4*lam/(1+4*lam)
    V0, Vl = F(2, 3), F(1, 6)*F(1, 4)/(F(1, 4)+lam)**2
    threshold = (V0-Vl)/P
    return {"K": str(K), "k": str(k), "b3": str(b3), "c2": str(c2),
            "P_lambda": str(P), "P_lambda_float": float(P), "V0": str(V0),
            "V_lambda": str(Vl), "t_star": float(threshold), "t_star_exact": str(threshold)}


def write_csv(path, rows):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.time()
    constants = exact_constants()
    rng = np.random.default_rng(CONFIG["seed"])
    rows = []
    lam, replicates = CONFIG["lambda"], CONFIG["paired_draws"]
    for eps in CONFIG["epsilons"]:
        cb, cf = coefficients(eps, 0.0), coefficients(eps, lam)
        actual_gap = cb["J"]-cf["J"]
        actual_vdiff = cf["V"]-cb["V"]
        for factor in CONFIG["sample_factors"]:
            n = round(factor*constants["t_star"]/eps**5)
            counts = rng.multinomial(n, [1/3]*3, size=replicates)
            successes = rng.binomial(counts, sigmoid(eps*GG))
            gaps = counts*(eps*GG)
            tb = fitted(counts, successes, gaps, 0.0)
            tf = fitted(counts, successes, gaps, lam)
            # Regret_fused-regret_binary = J_binary-J_fused, so no oracle root needed.
            difference = value(tb, eps)-value(tf, eps)
            mean, se = float(difference.mean()), float(difference.std(ddof=1)/math.sqrt(replicates))
            row = {"epsilon": eps, "factor": factor, "n": n, "paired_draws": replicates,
                   "mean_regret_difference": mean, "monte_carlo_se": se,
                   "mc_ci_low": mean-1.96*se, "mc_ci_high": mean+1.96*se,
                   "fixed_epsilon_expansion": actual_gap+actual_vdiff/n,
                   "local_alternative_prediction": eps**5*constants["P_lambda_float"]*(1-1/factor),
                   "scaled_regret_difference": mean/eps**5,
                   "population_regret_difference": actual_gap,
                   "V_binary": cb["V"], "V_fused": cf["V"], "P_limit": constants["P_lambda_float"],
                   "boundary_binary": int(np.count_nonzero(np.abs(tb)>=2-1e-12)),
                   "boundary_fused": int(np.count_nonzero(np.abs(tf)>=2-1e-12))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_csv(HERE/'fusion_local_alternatives_results.csv', rows)
    manifest = {"elapsed_seconds": time.time()-started, "numpy": np.__version__,
                "python": platform.python_version(), "count_of_prespecified_cells": len(rows),
                "all_cells_reported": True, "raw_trials_regenerable_from_config": True,
                "cell_statistics": "fusion_local_alternatives_results.csv", "exact_constants": constants, "training": False,
                "config_sha256": hashlib.sha256((HERE/"fusion_local_alternatives_config.json").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path=HERE/'fusion_sprint1/run_manifest.json'
    complete=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    complete['local_alternatives']=manifest
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(complete,indent=2)+'\n')


if __name__ == "__main__":
    main()
