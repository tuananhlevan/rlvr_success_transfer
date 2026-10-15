"""Reproduce the exact lemma certificate and paired queue stress tests.

CPU only. All jobs have finite service and are processed once. Results are
synthetic mechanism checks, not language-model or distributed-system results.
"""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import heapq
import json
import math
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = {
    "workers": [32, 128, 512], "seeds": [0, 1, 2],
    "horizon_multiplier": 10, "negative_probability": 1 / 16,
    "slow_service": 1_000_000_000, "initial_x": 0.1,
    "conditions": ["informative", "independent", "equal"],
    "initial_service": 1, "parallel_processes": 3,
    "objective": "f(x)=x^2/2; g(x,e)=x+e; L=sigma=1",
    "initial_rule": "Designated worker 0 first; other initial gradients zero",
    "tie_rule": "Monotone job identifier",
}


def certificate():
    beta = Fraction(1, 1008)
    c = 1 - beta
    lower = c ** 256 * sum((1 - c ** j) ** 2 for j in range(1, 449)) / 8432
    upper = beta / (1 - Fraction(1, 10000))
    assert lower > upper
    return {
        "M": 64, "T": 512, "beta": "1/1008", "eta": "beta/100",
        "service_negative": 1_000_000,
        "cantelli_event_probability_lower": "512/527",
        "expectation_lower_bound": float(lower),
        "lemma_implied_upper_bound": float(upper),
        "lower_upper_ratio": float(lower / upper),
        "strict_inequality_verified_with_exact_rationals": True,
        "scope": "Appendix A.3 lemma; not a theorem counterexample",
    }


def queue_identity():
    """Closed-form FIFO negative-release attenuation, separate from simulation."""
    p = Fraction(1, 16)
    values = []
    for M in (32, 64, 128, 512, 4096):
        K = M - 1
        beta = Fraction(1, 16 * K)
        c = 1 - beta
        attenuation = (p * c / (1 - (1 - p) * c)) ** K
        values.append({"workers": M, "negative_mean_attenuation": float(attenuation),
                       "stationary_mean_momentum": math.sqrt(float(p * (1 - p))) * (1 - float(attenuation))})
    return {"values": values, "limiting_mean_momentum": math.sqrt(15) / 16 * (1 - math.exp(-1)),
            "scope": "Noise-only FIFO scheduler; no full objective convergence theorem"}


def information_lower_bound():
    """Exact problem parameters for the original two-population derivation."""
    rows = []
    for M in (32, 128, 512, 4096, 65536):
        T = M * M
        p = Fraction(M, 4 * (T + M))
        delta_squared = p / (1 - p)
        gap = delta_squared / 2
        event_probability = Fraction(9 * M, 9 * M + 4)
        lower = event_probability * delta_squared
        rate_shape = float(gap * M / T) + math.sqrt(float(gap) / T)
        rows.append({"workers": M, "T": T, "rare_probability": str(p),
                     "population_mean_squared": float(delta_squared),
                     "initial_gap": float(gap), "event_probability_lower": str(event_probability),
                     "gradient_error_lower": float(lower),
                     "worker_independent_noise_rate_shape": rate_shape,
                     "lower_over_rate_shape": float(lower) / rate_shape})
    return {"rows": rows,
            "scope": "Completion-only oracle; work-conserving immediate refill; no barrier or new oracle",
            "status": "Original derived lower bound; not externally peer reviewed; novelty not established"}


def plot(aggregates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for cond, label in (("informative", "Noise-dependent delays"), ("independent", "Independent delays")):
        selected = [r for r in aggregates if r["condition"] == cond]
        axis = axes[0 if cond == "informative" else 1]
        xs = [r["workers"] for r in selected]
        for method, name in (("ordered", "Delay attenuation"), ("ordinary", "Ordinary momentum")):
            axis.errorbar(xs, [r[method + "_mean_gradient_squared_mean"] for r in selected],
                          yerr=[r[method + "_mean_gradient_squared_sd"] for r in selected],
                          marker="o", capsize=3, label=name)
        axis.set(xscale="log", yscale="log", xlabel="Workers M (T = 10 M²)",
                 ylabel="Mean squared population gradient", title=label)
        axis.set_xticks(xs, [str(v) for v in xs])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Synthetic paired queue experiment; mean ± SD across 3 seeds", fontsize=11)
    fig.savefig(HERE / "results.png", dpi=180)
    plt.close(fig)


def trial(args):
    M, seed, condition = args
    T = CONFIG["horizon_multiplier"] * M * M
    x0 = CONFIG["initial_x"]
    delta = x0 * x0 / 2
    beta = min(1 / (16 * (M - 1)), math.sqrt(5 * delta / T))
    eta = beta / math.sqrt(8)
    c = 1 - beta
    p = CONFIG["negative_probability"]
    rng = np.random.default_rng(seed)
    negatives = rng.random(T + M) < p
    independent_slow = np.random.default_rng(seed + 100_000).random(T + M) < p
    a = math.sqrt(p / (1 - p))
    b = math.sqrt((1 - p) / p)
    noise = np.where(negatives, -b, a)
    queue = [(1, j, 1, x0, x0, float(noise[j])) for j in range(M)]
    heapq.heapify(queue)
    xo = xu = x0
    mo = mu = 0.0
    sum_o = sum_u = sum_noise = sum_weighted = sum_lag = 0.0
    launched_noise = float(noise[:M].sum())
    used_negative = 0
    sum_factor = [0.0, 0.0]
    factor_n = [0, 0]
    max_lag = 0
    for t in range(1, T + 1):
        ready, jid, launch, old_o, old_u, e = heapq.heappop(queue)
        # Evaluate the source's uniform-iterate quantity before updating.
        sum_o += xo * xo
        sum_u += xu * xu
        lag = t - launch
        factor = c ** lag
        go, gu = old_o + e, old_u + e
        if launch == 1 and t > 1:
            go = gu = 0.0
        else:
            sum_weighted += factor * e
            used_negative += int(e < 0)
            group = int(e < 0)
            sum_factor[group] += factor
            factor_n[group] += 1
        mo = c * mo + beta * factor * go
        mu = c * mu + beta * gu
        xo -= eta * mo
        xu -= eta * mu
        sum_noise += e
        sum_lag += lag
        max_lag = max(max_lag, lag)
        new_jid = M + t - 1
        new_e = float(noise[new_jid])
        launched_noise += new_e
        if condition == "informative":
            slow = new_e < 0
        elif condition == "independent":
            slow = bool(independent_slow[new_jid])
        else:
            slow = False
        service = CONFIG["slow_service"] if slow else 1
        heapq.heappush(queue, (ready + service, new_jid, t + 1, xo, xu, new_e))
    pending_noise = sum(job[-1] for job in queue)
    boundary_error = abs(launched_noise - sum_noise - pending_noise)
    # This identity concerns raw samples, including initial zeroed gradients.
    assert boundary_error < 1e-6 * (1 + abs(launched_noise))
    explicit = 640 * delta * (M - 1) / T + 16 * math.sqrt(5 * delta / T)
    return {
        "condition": condition, "workers": M, "seed": seed, "T": T,
        "beta": beta, "eta": eta,
        "ordered_mean_gradient_squared": sum_o / T,
        "ordinary_mean_gradient_squared": sum_u / T,
        "ordered_final_x": xo, "ordinary_final_x": xu,
        "explicit_v1_bound": explicit,
        "raw_arrival_noise_mean": sum_noise / T,
        "attenuated_noise_sum_per_update": sum_weighted / T,
        "positive_mean_attenuation": sum_factor[0] / max(1, factor_n[0]),
        "negative_mean_attenuation": sum_factor[1] / max(1, factor_n[1]),
        "used_negative_fraction": used_negative / max(1, T - M + 1),
        "mean_update_lag": sum_lag / T, "max_update_lag": max_lag,
        "virtual_wall_time": ready, "pending_jobs": len(queue),
        "lossless_noise_boundary_error": boundary_error,
    }


def main():
    start = time.time()
    tasks = [(M, seed, cond) for cond in CONFIG["conditions"]
             for M in CONFIG["workers"] for seed in CONFIG["seeds"]]
    rows = []
    with ProcessPoolExecutor(max_workers=CONFIG["parallel_processes"]) as pool:
        for row in pool.map(trial, tasks):
            rows.append(row)
            print(f"{row['condition']} M={row['workers']} seed={row['seed']} "
                  f"ordered={row['ordered_mean_gradient_squared']:.6g} "
                  f"ordinary={row['ordinary_mean_gradient_squared']:.6g}", flush=True)
    with (HERE / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for cond in CONFIG["conditions"]:
        for M in CONFIG["workers"]:
            selected = [r for r in rows if r["condition"] == cond and r["workers"] == M]
            metrics = {"condition": cond, "workers": M, "n_seeds": len(selected)}
            for field in ("ordered_mean_gradient_squared", "ordinary_mean_gradient_squared",
                          "raw_arrival_noise_mean", "attenuated_noise_sum_per_update",
                          "positive_mean_attenuation", "negative_mean_attenuation"):
                values = np.array([r[field] for r in selected])
                metrics[field + "_mean"] = float(values.mean())
                metrics[field + "_sd"] = float(values.std(ddof=1))
            metrics["explicit_v1_bound"] = selected[0]["explicit_v1_bound"]
            aggregates.append(metrics)
    manifest = {
        "config": CONFIG, "certificate": certificate(), "queue_identity": queue_identity(),
        "information_lower_bound": information_lower_bound(), "aggregates": aggregates,
        "elapsed_seconds": time.time() - start, "python": platform.python_version(),
        "numpy": np.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256((HERE / "results.csv").read_bytes()).hexdigest(),
        "external_data_or_models": False, "gradient_updates": "Paired methods share exact job events",
        "paper_versions": "Stress schedule uses v1 parameters; v3 has changed theorem assumptions and schedules",
        "limits": "3 seeds; synthetic scalar objective; virtual clock; not a rigorous full-theorem refutation",
    }
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot(aggregates)
    print(json.dumps(manifest["certificate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
