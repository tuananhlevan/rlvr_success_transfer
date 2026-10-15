"""Reproduce the finite-model checks for policy-dependent feedback complexity.

No language models or real annotators are used. Every result is a numerical
check of the stated finite decision problem, not evidence of LLM prevalence.
Outputs are one canonical JSON and one SVG; no intermediate files are retained.
"""
from pathlib import Path
import json
import math

import numpy as np
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import expit, logsumexp, xlogy
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TAU = 0.01
FEATURE = np.array([-1., 0., 1.])
CS = np.array([.25, .375, .5, .625, .75])
UTILITIES = np.column_stack([np.zeros(len(CS)), np.ones(len(CS)), CS])
THETAS = np.linspace(-3., 3., 12001)
LOG_POLICY = THETAS[:, None] * FEATURE
LOG_POLICY -= logsumexp(LOG_POLICY, axis=1)[:, None]
POLICIES = np.exp(LOG_POLICY)
KLS = np.sum(POLICIES * (LOG_POLICY + np.log(3)), axis=1)


def continuous_optimum(c, tau=TAU):
    u = np.array([0., 1., c])
    def evaluate(theta):
        lp = theta * FEATURE
        lp -= logsumexp(lp)
        p = np.exp(lp)
        val = p @ u - tau * (p @ (lp + np.log(3)))
        mean_f = p @ FEATURE
        derivative = p @ (u * FEATURE) - (p @ u) * mean_f
        derivative -= tau * theta * (p @ FEATURE**2 - mean_f**2)
        return float(val), float(derivative)
    scan = np.linspace(-3, 3, 3001)
    ds = np.array([evaluate(t)[1] for t in scan])
    candidates = [-3., 3.]
    for i in range(len(scan)-1):
        if ds[i] == 0:
            candidates.append(float(scan[i]))
        elif ds[i] * ds[i+1] < 0:
            candidates.append(float(brentq(lambda t: evaluate(t)[1], scan[i], scan[i+1])))
    theta = max(candidates, key=lambda t: evaluate(t)[0])
    return {"theta": theta, "value": evaluate(theta)[0],
            "stationary_candidates_checked": len(candidates)-2}


def game(policy_values):
    """Exact finite zero-sum game, up to numerical LP tolerances.

    Input rows: utility hypotheses. Columns: permitted deployment policies.
    Learner can randomize which policy it recommends. The oracle comparator
    for each hypothesis is restricted to those same columns.
    """
    h = policy_values.max(axis=1)
    regrets = h[:, None] - policy_values
    m, k = regrets.shape
    # Dual maximizes z s.t. z <= sum_u lambda_u regret(u, pi), for each pi.
    answer = linprog(
        np.r_[np.zeros(m), -1.],
        A_ub=np.column_stack([-regrets.T, np.ones(k)]),
        b_ub=np.zeros(k),
        A_eq=np.array([np.r_[np.ones(m), 0.]]), b_eq=[1.],
        bounds=[(0., 1.)]*m + [(0., None)], method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
    if not answer.success:
        raise RuntimeError(answer.message)
    return {"minimax_expected_regret": float(answer.x[-1]),
            "least_favorable_prior": answer.x[:-1].tolist(),
            "dual_violation": float(max(0., np.max(answer.ineqlin.residual * -1)))}


def value_envelope(tau):
    """Upper envelope of the grid policies' affine value functions of c."""
    slopes = POLICIES[:, 2]
    intercepts = POLICIES[:, 1] - tau*KLS
    active, starts = [], []
    for j in range(len(slopes)):
        cross = -math.inf
        while active:
            i = active[-1]
            cross = (intercepts[i]-intercepts[j])/(slopes[j]-slopes[i])
            if cross > starts[-1]:
                break
            active.pop(); starts.pop()
        if not active:
            cross = -math.inf
        active.append(j); starts.append(cross)
    active = np.array(active)
    starts = np.array(starts)
    def evaluate(c):
        c = np.asarray(c)
        index = active[np.searchsorted(starts, c, side="right")-1]
        return intercepts[index] + slopes[index]*c
    return evaluate


def partition(t):
    edges = [(0, 1), (0, 2), (1, 2)]
    groups = {}
    for j, u in enumerate(UTILITIES):
        gaps = [abs(u[a]-u[b]) for a, b in edges]
        signature = tuple(d if d < t else "*" for d in gaps)
        groups.setdefault(signature, []).append(j)
    return list(groups.values())


def interval_ambiguity(a, b, envelope):
    """Minimax regret for continuous c in [a,b], on the policy grid.

    Convexity of the value function reduces the worst utility to the interval's
    endpoints. Minimax duality then gives the largest endpoint Jensen gap.
    """
    if b-a < 1e-12:
        return 0.
    ha, hb = float(envelope(a)), float(envelope(b))
    def negative_gap(weight):
        mean_c = weight*a+(1-weight)*b
        return -(weight*ha+(1-weight)*hb-float(envelope(mean_c)))
    answer = minimize_scalar(negative_gap, bounds=(0., 1.), method="bounded",
                             options={"xatol": 1e-12})
    return max(0., -float(answer.fun))


def bayes_risk(n, kappa, envelope):
    """Exact binomial summation, with negligible tails omitted.

    Oracle knows the two possible rewards c=.25/.75 before seeing labels.
    The supplied envelope is the optimal deployment value as a function of c.
    Query has reward gap c; a count is an upset. This covers both examples.
    """
    qs = expit(-kappa*np.array([.25, .75]))
    included = set()
    for q in qs:
        lo = max(0, int(binom.ppf(1e-13, n, q)))
        hi = min(n, int(binom.ppf(1-1e-13, n, q)))
        included.update(range(lo, hi+1))
    counts = np.array(sorted(included))
    logs = np.stack([binom.logpmf(counts, n, q) for q in qs])
    log_mix = logsumexp(logs, axis=0)-np.log(2)
    posterior_first = expit(logs[0]-logs[1])
    cbar = .25*posterior_first + .75*(1-posterior_first)
    # Sum posterior conditional Bayes regrets, avoiding subtraction of totals.
    conditional_loss = posterior_first*envelope(.25)
    conditional_loss += (1-posterior_first)*envelope(.75)-envelope(cbar)
    mixture = np.exp(log_mix)
    risk = float(mixture @ conditional_loss)
    return {"labels": n, "expected_regret": max(0., risk),
            "omitted_probability": float(max(0., 1-mixture.sum()))}


def main():
    exact_d = np.sqrt(13)/6 - 1/np.sqrt(3)
    analytic_checks = []
    for c in CS:
        actual = continuous_optimum(float(c), tau=0)
        formula = (2*c-1 + 2*np.sqrt(c*c-c+1))/3
        analytic_checks.append({"c": float(c), "numerical_value": actual["value"],
                                "formula_value": float(formula),
                                "absolute_error": float(abs(actual["value"]-formula))})
    assert max(r["absolute_error"] for r in analytic_checks) < 1e-11
    unreg_game = game(UTILITIES @ POLICIES.T)
    assert abs(unreg_game["minimax_expected_regret"]-exact_d) < 1e-7

    values = UTILITIES @ POLICIES.T - TAU*KLS
    regularized_game = game(values)
    envelope = value_envelope(TAU)
    for c in CS:
        direct = float(np.max(POLICIES @ [0, 1, c] - TAU*KLS))
        assert abs(float(envelope(c))-direct) < 1e-12
    two_point_bayes_gap = float((envelope(.25)+envelope(.75))/2-envelope(.5))

    stages = []
    for t in [0., .250001, .375001, .500001]:
        blocks = partition(t)
        block_results = []
        for block in blocks:
            r = game(values[block])
            block_results.append({"c_values": CS[block].tolist(), **r})
        stages.append({"threshold": t, "blocks": block_results,
                       "maximum_block_regret": max(x["minimax_expected_regret"] for x in block_results)})
    assert stages[-1]["maximum_block_regret"] < 1e-8

    continuous_curve = []
    for t in np.linspace(.25, .5, 101):
        continuous_curve.append({"threshold": float(t),
                                 "ambiguity": interval_ambiguity(t, 1-t, envelope)})
    continuous_exponents = []
    for target in [.02, .01, .005, .002, .0005]:
        if target >= continuous_curve[0]["ambiguity"]:
            exponent = 0.
        else:
            exponent = float(brentq(lambda t: interval_ambiguity(t, 1-t, envelope)-target,
                                   .25, .5, xtol=1e-10))
        continuous_exponents.append({"target_expected_regret": target,
                                     "predicted_optimal_label_exponent": exponent})
    # Independent LP check of Jensen-gap calculation on all five utilities.
    assert abs(interval_ambiguity(.25, .75, envelope)-regularized_game["minimax_expected_regret"]) < 1e-7

    bounds = []
    epsilon = .005
    for kappa in [8, 16, 24, 32, 40, 48, 64, 80, 96, 128]:
        q0, q1 = expit(-kappa*np.array([.25, .75]))
        tv_per_label = q0-q1
        h2 = (np.sqrt(q0)-np.sqrt(q1))**2
        h2 += (np.sqrt(1-q0)-np.sqrt(1-q1))**2
        negative_log_affinity = -np.log1p(-h2/2)
        lower_n = np.log(two_point_bayes_gap/epsilon)/(-np.log1p(-tv_per_label))
        # Any incorrect hypothesis recommendation loses at most this amount.
        optimal_indices = [int(np.argmax(values[i])) for i in [0, 4]]
        cross_regrets = [float(values[i].max()-values[i, optimal_indices[1-j]])
                         for j, i in enumerate([0, 4])]
        wrong_policy_loss = max(cross_regrets)
        upper_n = np.log(wrong_policy_loss/epsilon)/negative_log_affinity
        lb_1000 = two_point_bayes_gap*np.exp(1000*np.log1p(-tv_per_label))
        bounds.append({"kappa": kappa, "per_label_tv": float(tv_per_label),
                       "squared_hellinger": float(h2),
                       "necessary_label_lower_bound": float(lower_n),
                       "sufficient_label_upper_bound": float(upper_n),
                       "regret_lower_bound_at_1000_labels": float(lb_1000)})

    oracle_checks = []
    for kappa in [16, 32, 48]:
        q0, q1 = expit(-kappa*np.array([.25, .75]))
        d = q0-q1
        for n in [1, 10, 100, 1000, 10000, 100000]:
            check = bayes_risk(n, kappa, envelope)
            check.update(kappa=kappa,
                         all_algorithm_lower_bound=two_point_bayes_gap*math.exp(n*math.log1p(-d)))
            assert check["expected_regret"]+1e-10 >= check["all_algorithm_lower_bound"]
            oracle_checks.append(check)

    rich_logits = np.array([0., 16., 0.])
    rich_logp = rich_logits-logsumexp(rich_logits)
    rich_p = np.exp(rich_logp)
    rich_kl = float(rich_p @ (rich_logp+np.log(3)))
    rich_regrets = []
    for c in CS:
        u = np.array([0, 1, c])
        full_optimum = TAU*(logsumexp(u/TAU)-np.log(3))
        deployed = rich_p @ u - TAU*rich_kl
        rich_regrets.append(float(full_optimum-deployed))
    pure_approx_bound = float(2*np.exp(-16)+2*TAU*np.exp(-.25/TAU))
    assert max(rich_regrets) <= pure_approx_bound+1e-12

    # Contextual witness: exactly solvable continuous one-parameter class.
    # x1 rewards (0,c), x2 rewards (.5,0), equiprobable prompts. Small policy
    # shares a logit; rich policy has independent logits with the same bounds.
    def contextual_envelope(c):
        return .25+TAU*(np.logaddexp(0., (np.asarray(c)-.5)/(2*TAU))-np.log(2))
    contextual_gap = float(TAU*(np.logaddexp(1/(16*TAU), -1/(16*TAU))-np.log(2)))
    assert abs(contextual_gap-float((contextual_envelope(.25)+contextual_envelope(.75))/2-contextual_envelope(.5))) < 1e-14
    contextual_bounds = []
    for row in bounds:
        kappa = row["kappa"]
        d = row["per_label_tv"]
        minus_log_affinity = -np.log1p(-row["squared_hellinger"]/2)
        lower_n = np.log(contextual_gap/epsilon)/(-np.log1p(-d))
        upper_n = np.log((.125*np.tanh(1/(16*TAU)))/epsilon)/minus_log_affinity
        rich_two_labels = float(np.tanh(8)*(.125*expit(-kappa/4)+.25*expit(-kappa/2)))
        contextual_bounds.append({"kappa": kappa,
                                  "necessary_label_lower_bound": float(lower_n),
                                  "sufficient_known_hypothesis_label_upper_bound": float(upper_n),
                                  "small_class_risk_lower_bound_at_1000_labels": float(contextual_gap*np.exp(1000*np.log1p(-d))),
                                  "rich_class_worst_case_risk_using_two_labels": rich_two_labels})
    contextual_oracles = []
    for kappa in [16, 32, 48]:
        d = float(expit(-kappa/4)-expit(-3*kappa/4))
        for n in [1, 10, 100, 1000, 10000, 100000]:
            check = bayes_risk(n, kappa, contextual_envelope)
            check.update(kappa=kappa,
                         all_algorithm_lower_bound=contextual_gap*math.exp(n*math.log1p(-d)))
            assert check["expected_regret"]+1e-10 >= check["all_algorithm_lower_bound"]
            contextual_oracles.append(check)

    # Audit: a different ONE-parameter representation already realizes the
    # rich optimum. This changes the class and does not refute the lower bound,
    # but rules out parameter count alone as its practical explanation.
    # p(1|x1)=sigmoid(theta), p(1|x2)=sigmoid(-theta).
    c_audit = np.linspace(.25, .75, 1001)
    prompt_aware_unconstrained_logits = (c_audit/2+.25)/TAU
    prompt_aware_logits = np.clip(prompt_aware_unconstrained_logits, -16., 16.)
    assert np.all(prompt_aware_logits == 16.)
    prompt_aware_p = np.column_stack([expit(prompt_aware_logits), expit(-prompt_aware_logits)])
    rich_reference_p = np.array([expit(16.), expit(-16.)])
    repair_policy_error = float(np.max(abs(prompt_aware_p-rich_reference_p)))
    assert repair_policy_error == 0.
    kappa_audit = 32.
    log_win = -np.logaddexp(0., -kappa_audit*c_audit)
    log_lose = -np.logaddexp(0., kappa_audit*c_audit)
    soft_probability_gap_error = float(np.max(abs((log_win-log_lose)/kappa_audit-c_audit)))
    assert soft_probability_gap_error < 1e-14
    original_tv = float(expit(.75*kappa_audit)-expit(.25*kappa_audit))
    flip_probability = .1
    contracted_tv = (1-2*flip_probability)*original_tv
    temperature_adjusted_tv = float(expit(.75)-expit(.25))

    result = {
        "status": "finite-model theoretical verification; no LLM or real-data experiment",
        "deployment_tau": TAU,
        "choice_inverse_temperatures": [r["kappa"] for r in bounds],
        "theta_interval": [-3, 3], "policy_grid_points": len(THETAS),
        "utilities": UTILITIES.tolist(),
        "continuous_grid_value_error_bound": .53*(THETAS[1]-THETAS[0])/2,
        "unregularized_closed_form_ordinal_regret": float(exact_d),
        "unregularized_formula_checks": analytic_checks,
        "unregularized_grid_game": unreg_game,
        "regularized_grid_game": regularized_game,
        "continuous_optima": [{"c": float(c), **continuous_optimum(float(c))} for c in CS],
        "two_hypothesis_equal_prior_bayes_gap": two_point_bayes_gap,
        "analytic_lower_bound_D_minus_tau_log3": float(exact_d-TAU*np.log(3)),
        "information_partition_stages": stages,
        "continuous_utility_ambiguity_curve": continuous_curve,
        "continuous_utility_predicted_exponents": continuous_exponents,
        "target_expected_regret": epsilon, "two_hypothesis_label_bounds": bounds,
        "known_hypothesis_bayes_oracle_checks": oracle_checks,
        "rich_full_simplex_regrets": rich_regrets,
        "rich_analytic_regret_bound_given_correct_top_action": pure_approx_bound,
        "rich_two_label_tournament_regret_bound_at_kappa32": float(pure_approx_bound+(1+TAU*np.log(3))*(np.exp(-32)+np.exp(-8))),
        "contextual_example": {
            "prompt_probability": [.5, .5], "utility_x1": [0, "c"], "utility_x2": [.5, 0],
            "small_shared_logit_interval": [-16, 16],
            "rich_independent_logit_intervals": [[-16, 16], [-16, 16]],
            "small_optimal_logits_at_endpoint_rewards": [-12.5, 12.5],
            "rich_optimal_logits_for_every_c": [16, -16],
            "equal_prior_exact_bayes_gap": contextual_gap,
            "label_bounds": contextual_bounds,
            "known_hypothesis_bayes_oracle_checks": contextual_oracles},
        "practical_escape_audit": {
            "same_parameter_count_prompt_aware_policy": "p(1|x1)=sigmoid(theta); p(1|x2)=sigmoid(-theta)",
            "free_parameters": 1, "theta_bounds": [-16, 16],
            "unconstrained_optimal_theta_range": [float(prompt_aware_unconstrained_logits.min()), float(prompt_aware_unconstrained_logits.max())],
            "constrained_optimal_theta_for_all_c": 16,
            "max_policy_difference_from_rich_optimum_on_1001_c_values": repair_policy_error,
            "changes_deployment_class": True,
            "not_a_LoRA_experiment": True,
            "calibrated_log_probability_gap_identity_max_error": soft_probability_gap_error,
            "original_tv_at_kappa32": original_tv,
            "posthoc_flip_probability": flip_probability,
            "posthoc_flipped_label_tv": contracted_tv,
            "true_oracle_temperature_T_equal_kappa_tv": temperature_adjusted_tv,
            "oracle_note": "Log probability access or true temperature control changes the feedback experiment; label flips cannot add information."},
        "bayes_oracle_note": "Knows both hypotheses; uses fixed most informative pair; not claimed optimal adaptive design.",
        "partition_game_note": "LP exact for the documented finite grid, up to solver tolerance; continuous claims have separate arguments.",
    }
    (HERE/"results.json").write_text(json.dumps(result, indent=2)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    kappas = [r["kappa"] for r in bounds]
    axes[0].semilogy(kappas, [r["necessary_label_lower_bound"] for r in bounds], "o-", label="Necessary (any learner)")
    axes[0].semilogy(kappas, [r["sufficient_label_upper_bound"] for r in bounds], "s--", label="Sufficient (known hypotheses)")
    axes[0].axhline(2, color="tab:green", linestyle=":", label="Richer class: 2 labels*")
    axes[0].set(xlabel="Choice inverse temperature κ", ylabel="Labels for expected regret ≤ 0.005", title="Two reward hypotheses: matching exponent")
    axes[0].legend(fontsize=7)
    for kappa in [16, 32, 48]:
        rs = [r for r in oracle_checks if r["kappa"] == kappa]
        axes[1].semilogx([r["labels"] for r in rs], [r["expected_regret"] for r in rs], "o-", label=f"κ={kappa}")
    axes[1].axhline(epsilon, color="black", linestyle=":", linewidth=1)
    axes[1].set(xlabel="Binary labels", ylabel="Bayes expected deployment regret", title="Even a strongly informed oracle needs labels")
    axes[1].legend(fontsize=8)
    axes[2].plot([r["threshold"] for r in continuous_curve], [r["ambiguity"] for r in continuous_curve])
    axes[2].axhline(epsilon, color="black", linestyle=":", linewidth=1)
    exponent_005 = next(r["predicted_optimal_label_exponent"] for r in continuous_exponents
                        if r["target_expected_regret"] == epsilon)
    axes[2].axvline(exponent_005, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Revealed comparison-gap threshold t", ylabel="Remaining minimax regret Aτ(t)",
                title=f"Continuous reward family: Γε={exponent_005:.3f}")
    fig.text(.01, .01, "Finite-model verification. *Richer-class bound holds at κ≥32; no LLM experiment.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(HERE/"feedback_complexity.svg")
    plt.close(fig)
    print(json.dumps({"ordinal_regret": float(exact_d),
                      "regularized_grid_minimax": regularized_game["minimax_expected_regret"],
                      "two_point_bayes_gap": two_point_bayes_gap,
                      "partition_ambiguities": [s["maximum_block_regret"] for s in stages],
                      "continuous_exponents": continuous_exponents,
                      "kappa32_bound": bounds[3],
                      "rich_regret_bound": pure_approx_bound,
                      "contextual_kappa32_result": contextual_bounds[3],
                      "same_parameter_count_prompt_feature_repair_error": repair_policy_error,
                      "all_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
