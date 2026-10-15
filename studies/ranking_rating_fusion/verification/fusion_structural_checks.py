"""Bounded checks for fusion_structural_theorems.md; standard library only.

No training, external calls, or random-instance search. Numerical checks do not
replace proofs. Fixed pair weights remain fixed throughout each fusion path.
"""
import json
import math
from fractions import Fraction as F
from pathlib import Path


def rational_sign_checks():
    """Retain the exact sign certificates from the earlier prototype check."""
    def exp_bounds(x):
        assert x>=0
        term=F(1);lower=term
        for k in range(1,81):
            term*=x/k;lower+=term
        upper=lower+term*x/81/(1-x/82)
        return lower,upper
    def sigmoid_bounds(x):
        low,high=exp_bounds(abs(x))
        return (low/(1+low),high/(1+high)) if x>=0 else (1/(1+high),1/(1+low))
    def binary_derivative_bounds(theta):
        terms=[]
        for d,g in [(F(7,10),F(10)),(F(1),F(4)),(F(3,10),F(-6))]:
            a,b=sigmoid_bounds(theta*d);c,e=sigmoid_bounds(g)
            terms.append((d*(a-e)/3,d*(b-c)/3))
        return sum(x[0] for x in terms),sum(x[1] for x in terms)
    b2=binary_derivative_bounds(F(2));br=binary_derivative_bounds(F(460,79))
    a,b,c=[exp_bounds(x) for x in [F(7,5),F(2),F(17,5)]]
    j2=(F(301,50)*a[0]+2*b[0]-F(99,50)*c[1],
        F(301,50)*a[1]+2*b[1]-F(99,50)*c[0])
    assert b2[1]<0 and j2[1]<0 and br[0]>0
    return {'method':'exact Fraction Taylor bounds with geometric exponential remainder',
            'binary_derivative_at_2':[float(x) for x in b2],
            'unnormalized_value_derivative_at_2':[float(x) for x in j2],
            'binary_derivative_at_rating_optimum':[float(x) for x in br],
            'exact_sign_assertions_passed':True}


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    v = math.exp(x)
    return v / (1 + v)


def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def bisect(g, lo=-100.0, hi=100.0):
    assert g(lo) < 0 < g(hi)
    for _ in range(150):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def probabilities(theta, features, reference):
    logits = [math.log(p) + dot(theta, f) for p, f in zip(reference, features)]
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    return [w / sum(weights) for w in weights]


def scalar_case(features, rewards, reference, tau, qb, qr):
    edges = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    ds = [features[i] - features[j] for i, j in edges]
    gaps = [rewards[i] - rewards[j] for i, j in edges]

    def binary_gradient(t):
        return sum(w * tau * d * (sigmoid(tau * t * d) - sigmoid(g)) for w, d, g in zip(qb, ds, gaps))

    A = tau * tau * sum(w * d * d for w, d in zip(qr, ds))
    rating = sum(w * d * g for w, d, g in zip(qr, ds, gaps)) / (tau * sum(w * d * d for w, d in zip(qr, ds)))
    binary = bisect(binary_gradient)

    def value(t):
        p = probabilities([t], [[x] for x in features], reference)
        return dot(p, rewards) - tau * sum(pi * math.log(pi / rho) for pi, rho in zip(p, reference))

    def covariance_ratio(t):
        logs = [math.log(reference[i] * reference[j] * d * d) + t * (features[i] + features[j]) for (i, j), d in zip(edges, ds)]
        maximum = max(logs)
        w = [math.exp(x - maximum) for x in logs]
        return sum(ww * g / d for ww, g, d in zip(w, gaps, ds)) / sum(w)

    optimum = bisect(lambda t: tau * t - covariance_ratio(t))
    direction = 1 if rating > binary else -1 if rating < binary else 0
    distance = abs(rating - binary)
    towards = direction * (optimum - binary)
    endpoint_difference = value(rating) - value(binary)
    if direction == 0 or abs(distance) < 1e-12:
        classification = "neutral"
    elif towards <= 0:
        classification = "all_positive_weights_harm"
    elif towards >= distance:
        classification = "monotone_positive"
    elif endpoint_difference >= 0:
        classification = "overshoot_but_all_finite_weights_help"
    else:
        classification = "overshoot_then_cross_below_binary"
    rows = []
    for lam in (0.0, 1e-5, 0.01, 0.1, 1.0, 10.0, 1e4):
        theta = binary if lam == 0 else bisect(lambda t: binary_gradient(t) + lam * A * (t - rating))
        difference = value(theta) - value(binary)
        if lam > 0 and classification == "all_positive_weights_harm":
            assert difference < 0
        if lam > 0 and classification in ("monotone_positive", "overshoot_but_all_finite_weights_help"):
            assert difference > 0
        if lam > 0 and distance > 1e-12:
            recovered = -binary_gradient(theta) / (A * (theta - rating))
            assert abs(recovered - lam) < max(1e-8, 1e-7 * lam)
        rows.append({"lambda": lam, "theta": theta, "J_minus_binary": difference})
    special = {}
    if 0 < towards < distance:
        special["peak_lambda"] = -binary_gradient(optimum) / (A * (optimum - rating))
        if endpoint_difference < 0:
            # h ranges from the binary endpoint toward the rating endpoint.
            crossing_h = bisect(lambda h: value(binary) - value(binary + direction * h), towards, distance)
            crossing = binary + direction * crossing_h
            crossing_lambda = -binary_gradient(crossing) / (A * (crossing - rating))
            assert crossing_lambda > special["peak_lambda"] > 0
            special.update({"baseline_crossing_theta": crossing, "baseline_crossing_lambda": crossing_lambda})
    return {
        "features": features, "rewards": rewards, "reference": reference, "tau": tau,
        "binary_edge_weights": qb, "rating_edge_weights": qr,
        "theta_binary": binary, "theta_rating": rating, "theta_value_optimum": optimum,
        "J_binary": value(binary), "J_rating": value(rating), "endpoint_value_difference": endpoint_difference,
        "classification": classification, "special_weights": special, "finite_weight_rows": rows,
    }


def eigenvalues2(matrix):
    a, b, c = matrix[0][0], matrix[0][1], matrix[1][1]
    radius = math.sqrt((a - c) ** 2 + 4 * b * b)
    return ((a + c - radius) / 2, (a + c + radius) / 2)


def inverse2(matrix):
    a, b, c = matrix[0][0], matrix[0][1], matrix[1][1]
    determinant = a * c - b * b
    assert determinant > 0
    return [[c / determinant, -b / determinant], [-b / determinant, a / determinant]]


def mv(matrix, vector):
    return [dot(row, vector) for row in matrix]


def matrix_sum(a, b, factor=1.0):
    return [[a[i][j] + factor * b[i][j] for j in range(2)] for i in range(2)]


def multidimensional_checks():
    features = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    rewards = (0.0, 0.3, 0.4, 0.9)
    reference = (0.25,) * 4
    tau = 1.0
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    ds = [[features[i][k] - features[j][k] for k in range(2)] for i, j in edges]
    gaps = [rewards[i] - rewards[j] for i, j in edges]
    qb = [1 / 6] * 6

    def binary_stats(theta):
        zs = [tau * dot(theta, d) for d in ds]
        loss = sum(w * (softplus(z) - sigmoid(g) * z) for w, z, g in zip(qb, zs, gaps))
        gradient = [sum(w * tau * d[k] * (sigmoid(z) - sigmoid(g)) for w, d, z, g in zip(qb, ds, zs, gaps)) for k in range(2)]
        hessian = [[sum(w * tau * tau * d[k] * d[l] * sigmoid(z) * (1 - sigmoid(z)) for w, d, z in zip(qb, ds, zs)) for l in range(2)] for k in range(2)]
        return loss, gradient, hessian

    def optimize(lam, qr, start=(0.0, 0.0)):
        theta = list(start)
        A = [[sum(w * tau * tau * d[k] * d[l] for w, d in zip(qr, ds)) for l in range(2)] for k in range(2)]
        for _ in range(100):
            loss, gradient, hessian = binary_stats(theta)
            residuals = [tau * dot(theta, d) - g for d, g in zip(ds, gaps)]
            objective = loss + lam / 2 * sum(w * z * z for w, z in zip(qr, residuals))
            rg = [sum(w * tau * d[k] * z for w, d, z in zip(qr, ds, residuals)) for k in range(2)]
            gradient = [gradient[k] + lam * rg[k] for k in range(2)]
            if norm(gradient) < 1e-13:
                return theta
            step = mv(inverse2(matrix_sum(hessian, A, lam)), gradient)
            rate = 1.0
            while rate > 1e-12:
                candidate = [theta[k] - rate * step[k] for k in range(2)]
                candidate_loss = binary_stats(candidate)[0] + lam / 2 * sum(w * (tau * dot(candidate, d) - g) ** 2 for w, d, g in zip(qr, ds, gaps))
                if candidate_loss <= objective - 1e-4 * rate * dot(gradient, step):
                    theta = candidate
                    break
                rate /= 2
            else:
                assert norm(gradient) < 1e-9
                return theta
        raise AssertionError("Newton solver did not converge")

    binary = optimize(0.0, qb)
    p0 = probabilities(binary, features, reference)
    # Include self-pair masses p_i^2 implicitly; their rating loss is zero.
    qr = [2 * p0[i] * p0[j] for i, j in edges]
    A = [[sum(w * tau * tau * d[k] * d[l] for w, d in zip(qr, ds)) for l in range(2)] for k in range(2)]
    H0 = binary_stats(binary)[2]
    g = [sum(w * tau * d[k] * (tau * dot(binary, d) - gap) for w, d, gap in zip(qr, ds, gaps)) for k in range(2)]
    minimum_A = eigenvalues2(A)[0]
    radius = norm(g) / minimum_A
    largest_margin = max(tau * (abs(dot(binary, d)) + radius * norm(d)) for d in ds)
    gram = [[sum(w * d[k] * d[l] for w, d in zip(qb, ds)) for l in range(2)] for k in range(2)]
    minimum_B = tau * tau * sigmoid(largest_margin) * (1 - sigmoid(largest_margin)) * eigenvalues2(gram)[0]
    M = tau ** 3 / 4 * sum(w * norm(d) ** 3 for w, d in zip(qb, ds))
    diameter = math.sqrt(2)
    LJ = (tau + max(rewards) - min(rewards) + tau * (norm(binary) + radius) * diameter) * diameter ** 2 / 4

    def value_stats(theta):
        p = probabilities(theta, features, reference)
        means = [sum(p[i] * features[i][k] for i in range(4)) for k in range(2)]
        residual = [rewards[i] - tau * dot(theta, features[i]) for i in range(4)]
        residual_mean = dot(p, residual)
        gradient = [sum(p[i] * (features[i][k] - means[k]) * (residual[i] - residual_mean) for i in range(4)) for k in range(2)]
        hessian = [[sum(p[i] * (features[i][k] - means[k]) * (features[i][l] - means[l]) * (residual[i] - residual_mean - tau) for i in range(4)) for l in range(2)] for k in range(2)]
        value = dot(p, rewards) - tau * sum(pi * math.log(pi / rho) for pi, rho in zip(p, reference))
        return value, gradient, hessian, p, residual

    J0, v0, _, _, _ = value_stats(binary)
    rows = []
    for lam in (0.0001, 0.001, 0.01, 0.1, 1.0):
        theta = optimize(lam, qr, binary)
        h = [theta[k] - binary[k] for k in range(2)]
        hl = [-lam * x for x in mv(inverse2(matrix_sum(H0, A, lam)), g)]
        bound = norm(v0) * M * lam ** 2 * norm(g) ** 2 / (2 * (minimum_B + lam * minimum_A) ** 3) + LJ * lam ** 2 * norm(g) ** 2 / (2 * (minimum_B + lam * minimum_A) ** 2)
        J, v, HJ, p, residual = value_stats(theta)
        leading = dot(v0, hl)
        error = abs(J - J0 - leading)
        assert error <= bound + 1e-12
        assert norm(h) <= lam * norm(g) / (minimum_B + lam * minimum_A) + 1e-12
        moment = [sum(w * d[k] * (residual[i] - residual[j]) for w, d, (i, j) in zip(qr, ds, edges)) for k in range(2)]
        mismatch = [moment[k] - 2 * v[k] for k in range(2)]
        H = matrix_sum(binary_stats(theta)[2], A, lam)
        inverse = inverse2(H)
        response = [tau * x for x in mv(inverse, moment)]
        exact_path_derivative = dot(v, response)
        vn = math.sqrt(dot(v, mv(inverse, v)))
        dn = math.sqrt(max(0, dot(mismatch, mv(inverse, mismatch))))
        lower = tau * vn * (2 * vn - dn)
        assert exact_path_derivative >= lower - 1e-12
        chi_square = 0.0
        centered_pair_variance = 0.0
        for i in range(4):
            for j in range(4):
                target = p[i] * p[j]
                source = p0[i] * p0[j]
                chi_square += (source - target) ** 2 / target
                pair_feature = [features[i][k] - features[j][k] for k in range(2)]
                centered = [pair_feature[k] * (residual[i] - residual[j]) - 2 * v[k] for k in range(2)]
                centered_pair_variance += target * dot(centered, mv(inverse, centered))
        chi_bound = math.sqrt(max(0, chi_square * centered_pair_variance))
        assert dn <= chi_bound + 1e-12
        # Finite differences verify the exact residual-centered value Hessian.
        for axis in range(2):
            plus = theta[:]; minus = theta[:]
            plus[axis] += 1e-5; minus[axis] -= 1e-5
            vp = value_stats(plus)[1]; vm = value_stats(minus)[1]
            for coordinate in range(2):
                assert abs((vp[coordinate] - vm[coordinate]) / 2e-5 - HJ[coordinate][axis]) < 1e-9
        rows.append({
            "lambda": lam, "theta": theta, "J_minus_binary": J - J0,
            "ridge_linearized_value_change": leading, "finite_lambda_remainder_bound": bound,
            "observed_absolute_remainder": error, "positive_transfer_certified_by_bound": leading > bound,
            "exact_path_value_derivative": exact_path_derivative, "moment_metric_lower_bound": lower,
            "pair_design_chi_square_q_over_current_policy": chi_square,
            "mismatch_metric_norm": dn, "centered_chi_square_mismatch_upper_bound": chi_bound,
        })
    assert any(row["positive_transfer_certified_by_bound"] for row in rows)
    return {"features": features, "rewards": rewards, "reference": reference, "tau": tau,
            "binary_theta": binary, "rating_design": "frozen independent current-policy pairs; self pairs implicit",
            "certified_global_path_ball_radius": radius, "binary_Hessian_lower_bound": minimum_B,
            "rating_Hessian_lower_bound": minimum_A, "binary_Hessian_Lipschitz_bound": M,
            "value_Hessian_norm_bound": LJ, "rows": rows}


def main():
    f = (0.0, 0.7, 1.0); r = (0.0, 10.0, 4.0); rho = (1 / 3,) * 3
    qb = [1 / 3] * 3
    base = scalar_case(f, r, rho, 1.0, qb, qb)
    B = base["theta_binary"]
    p = probabilities([B], [[x] for x in f], rho)
    edges = [(0, 1), (0, 2), (1, 2)]
    on = [2 * p[i] * p[j] for i, j in edges]
    # Mix with the conditional distinct-edge law solely to create prespecified
    # endpoint geometries; this is not a search over empirical instances.
    def weights_for_endpoint(target):
        ds = [f[i] - f[j] for i, j in edges]
        gaps = [r[i] - r[j] for i, j in edges]
        def endpoint(alpha):
            q = [(1 - alpha) * a + alpha * b for a, b in zip(qb, on)]
            return sum(w * d * g for w, d, g in zip(q, ds, gaps)) / sum(w * d * d for w, d in zip(q, ds))
        alpha = bisect(lambda a: target - endpoint(a), 0.0, 1.0)
        return [(1 - alpha) * a + alpha * b for a, b in zip(qb, on)]
    cases = {
        "original_negative": base,
        "monotone_positive": scalar_case(f, r, rho, 1.0, qb, weights_for_endpoint(1.8)),
        "safe_overshoot": scalar_case(f, r, rho, 1.0, qb, weights_for_endpoint(1.3)),
        "unsafe_frozen_onpolicy_overshoot": scalar_case(f, r, rho, 1.0, qb, on),
        "nonuniform_reference_changed_temperature": scalar_case(f, r, (0.2, 0.3, 0.5), 1.7, qb, qb),
    }
    assert cases["monotone_positive"]["classification"] == "monotone_positive"
    assert cases["safe_overshoot"]["classification"] == "overshoot_but_all_finite_weights_help"
    assert cases["unsafe_frozen_onpolicy_overshoot"]["classification"] == "overshoot_then_cross_below_binary"
    multivariate = multidimensional_checks()
    # Convex quadratic illustration only; not claimed to be an instance of the
    # coupled logistic/Gibbs model. Convexity alone cannot yield endpoint safety.
    illustrative = []
    for lam in (0.00001, 0.1, 100.0):
        theta = (lam / (1 + lam), 100 * lam / (1 + 100 * lam))
        difference = 1 - (theta[0] - 1) ** 2 - theta[1] ** 2
        illustrative.append({"lambda": lam, "theta": theta, "J_minus_binary": difference})
    assert illustrative[0]["J_minus_binary"] > 0 > illustrative[1]["J_minus_binary"]
    assert illustrative[2]["J_minus_binary"] > 0
    results = {
        "scope": "bounded deterministic theorem checks; not neural evidence, statistical prevalence, or novelty clearance",
        "rational_sign_certificates": rational_sign_checks(),
        "scalar_cases": cases, "multidimensional_checks": multivariate,
        "generic_quadratic_endpoint_counterexample": {
            "scope": "convex-loss/concave-value illustration, not a coupled preference-model construction",
            "binary_optimum": [0, 0], "rating_optimum": [1, 1], "both_endpoint_values": -1,
            "rows": illustrative,
        }, "assertions_passed": True,
    }
    destination = Path(__file__).with_name("fusion_structural_checks_results.json")
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"results_file": str(destination), "scalar_cases": {k: v["classification"] for k, v in cases.items()},
                      "multidimensional_weights_checked": len(multivariate["rows"]), "assertions_passed": True}, indent=2))


if __name__ == "__main__":
    main()
