"""One canonical runner: exact populations, executable fixture, finite-budget audits.

Run with the probml Python environment. Uses generated arithmetic code only;
never executes downloaded candidate programs. Writes aggregate results only.
"""
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
_MPL_CACHE = tempfile.TemporaryDirectory(prefix="verifier-matplotlib-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE.name)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "experiment_config.json").read_text())
STATES = np.array([[1, -2], [1, 2], [-1, -1], [-1, 1]], dtype=float)
T, Z = STATES.T
X = np.column_stack([np.ones(4), Z])
XR = np.column_stack([X, T])
BASE = np.array([0.5, C["base_slope"]])
ROWS = []
EXECUTION_BANK = {}


def reward(amplitude):
    return (Z - amplitude * T + amplitude + 2) / (2 * amplitude + 4)


def fit(x, y, weights=None):
    if weights is None:
        weights = np.ones(len(y))
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights * y)
    return np.linalg.solve(gram + C["ridge"] * np.eye(x.shape[1]), rhs)


def selection_value(scores, gold):
    # Two equally likely task types; random tie-breaking is integrated exactly.
    values = []
    for ids in [[0, 1], [2, 3]]:
        ss = scores[ids]
        tied = np.flatnonzero(np.isclose(ss, ss.max(), atol=1e-12, rtol=0))
        values.append(float(np.mean(gold[np.asarray(ids)[tied]])))
    return float(np.mean(values))


def mixture(rho):
    return np.array([1 - rho, 1 + rho, 1 - rho, 1 + rho]) / 4


def record(experiment, amplitude, method, rho, budget, tests, metrics):
    ROWS.append(dict(experiment=experiment, nuisance_amplitude=amplitude,
                     method=method, winner_fraction=rho, label_budget=budget,
                     tests_per_audit=tests, **metrics))


def make_program(task_id, amplitude, t, z):
    """Each unseen task specifies a different affine function on finite inputs.

    Candidate code corrupts a prefix of its input domain. The proxy feature z
    is a task-relative semantic-quality signal; omitted task difficulty t
    creates deliberate misspecification. The richer control receives t.
    """
    denom = 2 * amplitude + 4
    correct = int(z - amplitude * t + amplitude + 2)
    a = (task_id % 11) + 1
    b = (task_id // 11) - 4
    code = (f"def solve(x):\n    y = {a} * x + ({b})\n"
            f"    if x >= {correct}:\n        y += 1\n    return y\n")
    namespace = {"__builtins__": {}}
    exec(compile(code, f"generated_task_{task_id}", "exec"), namespace)
    observed = np.array([namespace["solve"](i) == a * i + b
                         for i in range(denom)], dtype=float)
    return code, observed


def executable_fixture():
    h = hashlib.sha256()
    unique_sources = set()
    calls = programs = 0
    maximum_error = 0.0
    for amplitude in C["nuisance_amplitudes"]:
        gold = reward(amplitude)
        bank = {}
        for task_id in range(C["unique_program_tasks"]):
            t = 1 if task_id % 2 == 0 else -1
            for state_id in ([0, 1] if t == 1 else [2, 3]):
                z = int(Z[state_id])
                code, outcomes = make_program(task_id, amplitude, t, z)
                unique_sources.add(code)
                if state_id in bank:
                    assert np.array_equal(bank[state_id], outcomes)
                else:
                    bank[state_id] = outcomes
                h.update(code.encode())
                h.update(outcomes.tobytes())
                maximum_error = max(maximum_error, abs(outcomes.mean() - gold[state_id]))
                calls += len(outcomes)
                programs += 1
        EXECUTION_BANK[amplitude] = np.stack([bank[i] for i in range(4)])
    assert maximum_error < 1e-14
    return {"programs_executed": programs, "unit_test_calls": calls,
            "distinct_program_source_strings": len(unique_sources),
            "fixture_sha256": h.hexdigest(), "maximum_label_identity_error": maximum_error,
            "split": "Fixture verifies labels on task IDs 0..39 and unseen IDs 40..79; finite fits sample the equivalent four-state law and evaluate its exact population. This is not a benchmark transfer estimate.",
            "feature_provenance": "Generator-defined quality proxy z, not a pretrained code embedding. Rich control also observes task type t.",
            "cache_contract": "Finite-budget trials sample with replacement from execution-outcome arrays, cached by the four equivalent program types. Paid audit counts are modeled, not newly executed per fit."}


def populations():
    for amplitude in C["nuisance_amplitudes"]:
        gold = reward(amplitude)
        for rho in C["winner_fractions"]:
            q = mixture(rho)
            for name in ["winner_mix_ols", "ipw", "task_conditioned", "monotone_guard", "paired_gap"]:
                if name == "ipw" and rho == 1:
                    continue  # No positive propensity on losing answers.
                if name == "paired_gap":
                    dz = Z[[1, 3]] - Z[[0, 2]]
                    dg = gold[[1, 3]] - gold[[0, 2]]
                    theta = np.array([gold.mean(), dz @ dg / (dz @ dz)])
                    scores = X @ theta
                elif name == "task_conditioned":
                    # Pure winners leave the 3-parameter head underidentified.
                    if rho == 1:
                        continue
                    theta = fit(XR, gold, q)
                    scores = XR @ theta
                else:
                    weights = np.ones(4) / 4 if name == "ipw" else q
                    theta = fit(X, gold, weights)
                    if name == "monotone_guard":
                        theta[1] = max(theta[1], 1e-7)
                    scores = X @ theta
                b_theory = (1 - 0.5 * amplitude * rho / (2.5 - 2.25 * rho ** 2)) / (2 * amplitude + 4)
                record("population", amplitude, name, rho, "infinite", "full", {
                    "slope": float(theta[1]),
                    "theoretical_slope": b_theory if name == "winner_mix_ols" else "",
                    "selection_reward": selection_value(scores, gold),
                    "frozen_selection_reward": selection_value(X @ BASE, gold),
                    "acquisition_mse": float(q @ ((scores - gold) ** 2)),
                    "global_mse": float(np.mean((scores - gold) ** 2)),
                    "replications": 1})
                if name == "winner_mix_ols":
                    assert abs(theta[1] - b_theory) < 2e-7
                    if amplitude == 10 and rho == 1:
                        assert selection_value(np.clip(scores, 0, 1), gold) == .4375


def finite_audits():
    rng = np.random.default_rng(C["seed"])
    methods = ["uniform", "winner_only", "mixture_50", "mixture_90",
               "mixture_50_ipw", "mixture_90_ipw", "task_conditioned_50",
               "paired_gap", "mixture_50_monotone"]
    rho_map = {"uniform": 0, "winner_only": 1, "mixture_50": .5,
               "mixture_90": .9, "mixture_50_ipw": .5,
               "mixture_90_ipw": .9, "task_conditioned_50": .5,
               "paired_gap": "paired", "mixture_50_monotone": .5}
    for amplitude in C["nuisance_amplitudes"]:
        bank = EXECUTION_BANK[amplitude]
        gold = bank.mean(axis=1)
        frozen = selection_value(X @ BASE, gold)
        for tests in C["tests_per_audit"]:
            for budget in C["budgets"]:
                metrics = {name: [] for name in methods}
                # Common uniforms give matched stochastic inputs to all arms.
                for _ in range(C["replications"]):
                    task_bits = rng.integers(0, 2, budget)
                    choice_u = rng.random(budget)
                    audit_u = rng.random((budget, tests if isinstance(tests, int) else 1))
                    test_ids = np.floor(audit_u * bank.shape[1]).astype(int)
                    for name in methods:
                        rho = rho_map[name]
                        if name == "paired_gap":
                            # B/2 tasks x two candidates = B labels exactly.
                            task_ids = task_bits[:budget // 2]
                            lo = 2 * task_ids
                            hi = lo + 1
                            if tests == "full":
                                yl, yh = gold[lo], gold[hi]
                            else:
                                yl = bank[lo[:, None], test_ids[:budget // 2]].mean(1)
                                yh = bank[hi[:, None], test_ids[budget // 2:]].mean(1)
                            dz = Z[hi] - Z[lo]
                            slope = float(dz @ (yh - yl) / (dz @ dz + C["ridge"]))
                            # Absolute centering uses the same audited labels.
                            theta = np.array([float(np.mean((yl + yh) / 2)), slope])
                            scores = X @ theta
                            qmse = np.mean((scores - gold) ** 2)
                        else:
                            # Winner probability = (1+rho)/2 on every task.
                            ids = 2 * task_bits + (choice_u < (1 + rho) / 2).astype(int)
                            yy = gold[ids] if tests == "full" else bank[ids[:, None], test_ids].mean(1)
                            xx = XR[ids] if name == "task_conditioned_50" else X[ids]
                            w = .25 / mixture(rho)[ids] if name.endswith("ipw") else None
                            theta = fit(xx, yy, w)
                            if name == "mixture_50_monotone":
                                theta[1] = max(theta[1], 1e-7)
                            scores = XR @ theta if name == "task_conditioned_50" else X @ theta
                            qmse = float(mixture(rho) @ ((scores - gold) ** 2))
                        value = selection_value(scores, gold)
                        metrics[name].append([value, value - frozen, qmse,
                                              float(np.mean((scores - gold) ** 2)), float(theta[1])])
                for name in methods:
                    a = np.asarray(metrics[name])
                    record("finite_budget", amplitude, name, rho_map[name], budget, tests, {
                        "selection_reward": a[:, 0].mean(),
                        "selection_reward_se": a[:, 0].std(ddof=1) / np.sqrt(len(a)),
                        "delta_frozen": a[:, 1].mean(),
                        "delta_frozen_se": a[:, 1].std(ddof=1) / np.sqrt(len(a)),
                        "harm_rate": np.mean(a[:, 1] < -1e-9),
                        "acquisition_mse": a[:, 2].mean(),
                        "global_mse": a[:, 3].mean(),
                        "slope": a[:, 4].mean(), "replications": len(a)})


def smooth_selection_check():
    # Independently verify the derivative of softmax-selected gold on a slate.
    rng = np.random.default_rng(C["seed"] + 17)
    maximum_error = 0.0
    for k in [2, 4, 16]:
        s, g, direction = rng.normal(size=(3, k))
        temp = .7
        def value(scores):
            p = np.exp((scores - scores.max()) / temp)
            p /= p.sum()
            return p @ g, p
        val, p = value(s)
        analytic = np.sum(p * (g - val) * direction) / temp
        h = 1e-5
        numeric = (value(s + h * direction)[0] - value(s - h * direction)[0]) / (2 * h)
        err = abs(numeric - analytic)
        maximum_error = max(maximum_error, err)
        record("derivative_check", 0, "softmax_selection", "", k, "", {
            "analytic_derivative": analytic, "numeric_derivative": numeric,
            "absolute_error": err, "replications": 1})
    assert maximum_error < 1e-8
    return maximum_error


def adaptive_populations():
    """Close the audit/refit feedback loop; solve each population fit exactly.

    This is a population sensitivity analysis, not additional finite-budget
    experiments or language-model RL. A window fit uses current labels only;
    a pooled fit gives equal mass to all accumulated acquisition rounds.
    """
    for amplitude in C["nuisance_amplitudes"]:
        gold = reward(amplitude)
        frozen = selection_value(X @ BASE, gold)
        for rho in [.5, 1.0]:
            for name in ["window_ols", "pooled_ols", "window_ipw", "task_conditioned"]:
                if rho == 1 and name in ["window_ipw", "task_conditioned"]:
                    continue
                scores = X @ BASE
                cumulative_q = np.zeros(4)
                values = []
                for step in range(1, C["adaptive_rounds"] + 1):
                    direction = 1 if scores[1] > scores[0] else -1
                    q = mixture(direction * rho)
                    cumulative_q += q
                    if name == "task_conditioned":
                        theta = fit(XR, gold, q)
                        scores = XR @ theta
                    else:
                        w = cumulative_q / step if name == "pooled_ols" else q
                        if name == "window_ipw":
                            w = np.ones(4) / 4
                        theta = fit(X, gold, w)
                        scores = X @ theta
                    value = selection_value(scores, gold)
                    values.append(value)
                    record("adaptive_population", amplitude, name, rho, "infinite_per_round", "full", {
                        "round": step, "slope": float(theta[1]),
                        "selection_reward": value, "delta_frozen": value - frozen,
                        "mean_reward_through_round": float(np.mean(values)),
                        "acquisition_mse": float(q @ ((scores - gold) ** 2)),
                        "global_mse": float(np.mean((scores - gold) ** 2)), "replications": 1})


def figure():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rows = [r for r in ROWS if r["experiment"] == "population"
            and r["nuisance_amplitude"] == 10 and r["method"] == "winner_mix_ols"]
    axes[0].plot([r["winner_fraction"] for r in rows], [r["slope"] for r in rows], "o-")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set(xlabel="Fraction of winner audits", ylabel="Fitted quality slope",
                title="Selected-label fitting reverses ranking")
    axes[1].plot([r["acquisition_mse"] for r in rows], [r["selection_reward"] for r in rows], "o-")
    axes[1].set(xlabel="Selected-label MSE", ylabel="Exact population selection reward",
                title="Prediction MSE and selection can disagree")
    for method in ["uniform", "mixture_50", "mixture_50_ipw", "task_conditioned_50", "paired_gap"]:
        rr = [r for r in ROWS if r["experiment"] == "finite_budget"
              and r["nuisance_amplitude"] == 10 and r["tests_per_audit"] == 8
              and r["method"] == method]
        axes[2].errorbar([r["label_budget"] for r in rr], [r["selection_reward"] for r in rr],
                         yerr=[1.96 * r["selection_reward_se"] for r in rr], marker="o", label=method)
    axes[2].set(xscale="log", xlabel="Total candidate labels (pairs cost two)",
                ylabel="Exact population selection reward", title="Strong baselines repair the fixture")
    axes[2].legend(fontsize=7)
    fig.suptitle("Controlled program-verifier experiment; not an LLM benchmark", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "results.png", dpi=170)
    plt.close(fig)


def main():
    start = time.time()
    fixture = executable_fixture()
    populations()
    finite_audits()
    adaptive_populations()
    derivative_error = smooth_selection_check()
    fields = list(dict.fromkeys(k for row in ROWS for k in row))
    with (ROOT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ROWS)
    figure()
    manifest = {
        "study": "verifier_audit_transfer", "status": "controlled mechanism pilot only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((ROOT / "experiment_config.json").read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256((ROOT / "results.csv").read_bytes()).hexdigest(),
        "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "elapsed_seconds": time.time() - start,
        "ridge_regularization": C["ridge"],
        "population_computation": "Exact four-state expectations followed by numerical normal-equation solves; slope formula tolerance 2e-7.",
        "aggregate_rows": len(ROWS), "finite_configuration_count": sum(r["experiment"] == "finite_budget" for r in ROWS),
        "finite_fits": sum(r["replications"] for r in ROWS if r["experiment"] == "finite_budget"),
        "adaptive_population_rounds": sum(r["experiment"] == "adaptive_population" for r in ROWS),
        "derivative_max_abs_error": derivative_error,
        "executable_fixture": fixture,
        "access_limitations": ["Public Aletheia bulk download failed at DNS lookup", "In-app browser unavailable", "No accessible pretrained model weights or CUDA", "PCC cached data repository not verified"],
        "not_run": ["Aletheia replay", "MBPP or PCC replication", "pretrained language-model or VLM training", "human evaluation"]
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["aggregate_rows", "finite_configuration_count", "finite_fits", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
