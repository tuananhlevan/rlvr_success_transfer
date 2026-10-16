#!/usr/bin/env python3
"""Theory sprint: exact matrix checks and finite-model simulation, no LLM run."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq, linprog
from scipy.special import expit, softmax, xlogy


def check(d, eta=None):
    eta = 1 / d if eta is None else eta
    scales = np.full((d, d), eta)
    np.fill_diagonal(scales, 1)
    features = np.stack([
        np.concatenate([np.diag(row), -np.diag(row)]) for row in scales
    ])
    local = np.einsum("iaj,iak->ijk", features, features) / (2 * d)

    def leverage(mu):
        covariance = np.einsum("i,ijk->jk", mu, local)
        inverse = np.linalg.inv(covariance)
        return np.einsum("iaj,jk,iak->ia", features, inverse, features).max()

    # Covariance is diagonal and the union contains each unit axis.
    # Maximize the minimum diagonal entry using a generic linear-program solver.
    diagonal = np.diagonal(local, axis1=1, axis2=2)
    fit = linprog(
        np.r_[np.zeros(d), -1.0],
        A_ub=np.c_[-diagonal.T, np.ones(d)], b_ub=np.zeros(d),
        A_eq=np.array([np.r_[np.ones(d), 0.0]]), b_eq=np.ones(1),
        bounds=[(0, 1)] * d + [(0, None)], method="highs",
    )
    formula = d * d / (1 + (d - 1) * eta * eta)
    measured = leverage(np.full(d, 1 / d))
    joint_covariance = np.eye(d) / d
    joint = np.einsum(
        "iaj,jk,iak->ia", features, np.linalg.inv(joint_covariance), features
    ).max()
    assert abs(measured - formula) < 1e-8
    assert abs(joint - d) < 1e-8
    assert fit.success, fit.message
    optimized = leverage(fit.x[:-1])
    assert abs(optimized - formula) / formula < 1e-6
    return {"dimension": d, "eta": eta, "conditional_design_leverage": float(measured),
            "optimized_prompt_leverage": float(optimized), "joint_design_leverage": float(joint),
            "ratio": float(measured / joint), "optimizer_success": bool(fit.success)}


def feature_matrix(d, eta, prompt=0):
    scales = np.full(d, eta)
    scales[prompt] = 1
    return np.r_[np.diag(scales), -np.diag(scales)]


def retained_coin_check(d, eta):
    """Enumerate actual ordered response pairs, rather than assume a coin model."""
    f = feature_matrix(d, eta)
    axes = np.tile(np.arange(d), 2)
    signs = np.repeat([1, -1], d)
    main = axes == 0
    keep = np.logical_xor(main[:, None], main[None, :])
    main_sign = np.where(main[:, None], signs[:, None], signs[None, :])
    orientation = main_sign * np.where(main[:, None], 1, -1)
    coin_means = []
    for target in range(d):
        p_a_wins = expit(f[:, target, None] - f[None, :, target])
        p_coin = np.where(orientation > 0, p_a_wins, 1 - p_a_wins)
        coin_means.append(float(p_coin[keep].mean()))
    rho = 2 * (d - 1) / d**2
    assert abs(keep.mean() - rho) < 1e-12
    assert abs(coin_means[0] - expit(1)) < 1e-12
    assert np.max(np.abs(np.array(coin_means[1:]) - 0.5)) < 1e-12
    return {"dimension": d, "eta": eta, "acceptance": float(keep.mean()),
            "target_coin_mean": coin_means[0],
            "other_coin_max_error_from_half": float(max(abs(p - .5) for p in coin_means[1:]))}


def temperature_checks(d):
    """One prompt suffices by permutation symmetry for these flat shared logits."""
    f = feature_matrix(d, 1 / d)
    theta = np.full(d, np.log(d) / np.sqrt(d))
    out = []
    for scale in [1.0, 1 / np.sqrt(d), 1 / (2 * np.sqrt(d)), 1 / (4 * np.sqrt(d))]:
        plus = softmax(f @ theta / scale)
        minus = softmax(-f @ theta / scale)
        for kind, q in [("single", plus), ("sign_symmetric_mixture", (plus + minus) / 2)]:
            mean = q @ f
            trace = float(q @ np.square(f).sum(axis=1) - mean @ mean)
            kl_sum = np.zeros((2 * d, 2 * d))
            for j in range(d):
                z = (f[:, j, None] - f[None, :, j]) / 2
                kl_sum += np.logaddexp(z, -z) - np.log(2)
            information = float(q @ kl_sum @ q)
            out.append({"dimension": d, "temperature": float(scale), "sampler": kind,
                        "effective_parameter_norm_over_R": float(1 / scale),
                        "main_axis_probability": float(q[0] + q[d]),
                        "pair_feature_energy": 2 * trace,
                        "null_label_kl_summed_over_targets": information})
    return out


def kl_frontier(d):
    eta = 1 / d
    reference_main = 1 / d

    def binary_kl(p):
        return float(xlogy(p, p / reference_main)
                     + xlogy(1 - p, (1 - p) / (1 - reference_main)))

    rows = []
    for budget in sorted(set([0.0, np.log(d)/d, 0.1, 1.0, np.log(d)/2, np.log(d)])):
        p = (1.0 if budget >= np.log(d) else reference_main if budget == 0 else
             brentq(lambda value: binary_kl(value) - budget, reference_main, 1.0))
        q = np.full(2*d, (1-p)/(2*(d-1)))
        q[[0, d]] = p / 2
        kl = float(np.sum(xlogy(q, q * (2*d))))
        covariance = np.zeros((d, d))
        for prompt in range(d):
            q_prompt = np.full(2*d, (1-p)/(2*(d-1)))
            q_prompt[[prompt, prompt+d]] = p/2
            f = feature_matrix(d, eta, prompt)
            mean = q_prompt @ f
            covariance += ((f.T * q_prompt) @ f - np.outer(mean, mean)) / d
        measured = float(np.diag(np.linalg.inv(covariance)).max())
        formula = d / (eta*eta + (1-eta*eta)*p)
        assert abs(kl - budget) < 1e-8
        assert abs(measured - formula) / formula < 1e-10
        rows.append({"dimension": d, "kl_budget": float(budget),
                     "attained_kl": kl, "main_axis_mass": float(p),
                     "optimal_leverage": measured})
    return rows


def tournament(d, gap, delta, rng, cached=False):
    """Known-gap knockout tournament; self-contained confidence schedule in report."""
    assert d & (d - 1) == 0
    target = int(rng.integers(d))
    actual_gap = float(expit(2 if target == 0 else 2/d) - .5) if cached else gap
    active = rng.permutation(d)
    pulls = 0
    round_index = 1
    while len(active) > 1:
        # Error along the true arm's path in this round <= delta / 2**r.
        m = math.ceil(2 / gap**2 * math.log(2 ** (round_index + 1) / delta))
        paired_n = len(active) // 2 * 2
        players = active[:paired_n]
        means = 0.5 + actual_gap * (players == target)
        samples = rng.binomial(m, means)
        pulls += paired_n * m
        scores = samples.reshape(-1, 2)
        choose_right = scores[:, 1] > scores[:, 0]
        ties = scores[:, 1] == scores[:, 0]
        choose_right[ties] = rng.integers(2, size=int(ties.sum())).astype(bool)
        active = players.reshape(-1, 2)[np.arange(paired_n // 2), choose_right.astype(int)]
        # Default dimensions are powers of two; no simulated dummy arms are used.
        round_index += 1
    return bool(active[0] == target), pulls


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(mid - half), float(mid + half)]


def simulations(d, trials, seed):
    assert d & (d - 1) == 0
    rng = np.random.default_rng(seed + d)
    delta = 0.05
    local_success, local_effective = zip(*[
        tournament(d, float(expit(1) - .5), delta, rng) for _ in range(trials)
    ])
    direct_success, direct_effective = zip(*[
        tournament(d, float(expit(2) - .5), delta, rng) for _ in range(trials)
    ])
    local_effective = np.array(local_effective)
    direct_effective = np.array(direct_effective)
    rho = 2 * (d - 1) / d**2
    # Negative-binomial sampling exactly counts the rejected raw pairs.
    # No wall-clock generation savings are inferred from this accelerated simulation.
    local_raw = local_effective + rng.negative_binomial(local_effective, rho)
    R = np.log(d)
    p_main = np.cosh(R) / (np.cosh(R) + d - 1)
    conditional_acceptance = p_main**2 / 2
    conditional_raw = direct_effective + rng.negative_binomial(direct_effective, conditional_acceptance)
    cached_success, cached_labels = zip(*[
        tournament(d, float(expit(2/d) - .5), delta, rng, cached=True) for _ in range(trials)
    ])
    # Collect all 2d signed responses at one prompt. Sum of geometric waiting
    # times is the exact coupon-collector distribution; pair draws may overshoot by one response.
    coupon_p = np.arange(2*d, 0, -1) / (2*d)
    cached_raw = (rng.geometric(coupon_p, size=(trials, 2*d)).sum(axis=1) + 1) // 2

    def row(name, success, raw, labels, effective, paired=None):
        return {"dimension": d, "method": name, "trials": trials,
                "successes": int(sum(success)), "success_fraction": float(np.mean(success)),
                "success_wilson95": wilson(sum(success), trials),
                "mean_generated_pairs": float(np.mean(raw)),
                "mean_labels": float(np.mean(labels)),
                "mean_retained_comparisons": float(np.mean(effective)),
                "paired_information_with": paired}

    return [
        row("fixed_conditional_label_every_pair", local_success, local_raw, local_raw, local_effective,
            "fixed_conditional_screen_before_label"),
        row("fixed_conditional_screen_before_label", local_success, local_raw, local_effective, local_effective,
            "fixed_conditional_label_every_pair"),
        row("coordinated_direct_pairs", direct_success, direct_effective, direct_effective, direct_effective,
            "prompt_conditioned_bounded_policy_screen"),
        row("prompt_conditioned_bounded_policy_screen", direct_success, conditional_raw, direct_effective,
            direct_effective, "coordinated_direct_pairs"),
        row("cached_pair_recombination_repeated_labels", cached_success, cached_raw,
            np.array(cached_labels), np.array(cached_labels)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()
    assert args.trials > 0
    started = time.monotonic()
    rows = [check(d) for d in [2, 4, 8, 16, 32]]
    enumerations = [retained_coin_check(16, eta) for eta in [1/16, .5, 1.0]]
    sim = []
    for d in [16, 32, 64, 128]:
        sim.extend(simulations(d, args.trials, 20261016))
        print(f"Completed d={d}: five accounting/protocol controls, {args.trials} trials each", flush=True)
    result = {"date": "2026-10-16", "status": "scoped theory sprint complete; no full methods project justified",
              "meaning": "Exact geometry, enumeration and exact-distribution finite-model simulation. Generated pairs and queried labels are distinct resources. No LLM or human-data experiment.",
              "seed": 20261016, "tournament_error_budget": .05,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "rows": rows, "retained_coin_enumeration": enumerations,
              "overlap_sweep": [check(32, eta) for eta in [1.0, .5, 1/np.sqrt(32), 1/32]],
              "simulations": sim, "temperature_controls": [r for d in [16, 64] for r in temperature_checks(d)],
              "kl_budget_frontier": [r for d in [16, 64] for r in kl_frontier(d)],
              "runtime_seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"d={row['dimension']:2}: fixed conditional={row['conditional_design_leverage']:.3f}; "
              f"joint={row['joint_design_leverage']:.3f}; ratio={row['ratio']:.3f}")


if __name__ == "__main__":
    main()
