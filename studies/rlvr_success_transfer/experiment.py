"""Exact finite-policy checks for the research brief; no LLM results are implied.

Run with the probml environment. Writes one canonical JSON and no build files.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    rng = np.random.default_rng(20261015)
    # Coordinate change theta = M phi. Covectors and Fisher transform together.
    a = np.array([1., .2])
    b = np.array([.2, 1.])
    fisher = np.array([[2., .3], [.3, 1.]])
    transform = np.array([[1., -4.], [0., 1.]])
    inv = np.linalg.inv(fisher)
    changed_fisher = transform.T @ fisher @ transform
    changed_a, changed_b = transform.T @ a, transform.T @ b
    natural = lambda x, y, f: float(x @ np.linalg.solve(f, y))
    coordinate = {
        "raw_cosine_before": cosine(a, b),
        "raw_cosine_after": cosine(changed_a, changed_b),
        "natural_inner_product_before": natural(a, b, fisher),
        "natural_inner_product_after": natural(changed_a, changed_b, changed_fisher),
    }
    assert np.isclose(coordinate["natural_inner_product_before"],
                      coordinate["natural_inner_product_after"])

    # High-dimensional in-sample Fisher fit: independent random outcome labels.
    # This is a statistical null, not a model of the source paper's mechanism.
    n, dimension = 64, 96
    features = rng.normal(size=(n, dimension))
    features -= features.mean(axis=0)
    labels = np.zeros(n)
    labels[rng.choice(n, 4, replace=False)] = 1
    centered_labels = labels - labels.mean()
    gradient = features.T @ centered_labels / n
    empirical_fisher = features.T @ features / n
    chi = float(gradient @ np.linalg.pinv(empirical_fisher) @ gradient /
                np.mean(centered_labels ** 2))
    direction = np.linalg.pinv(empirical_fisher) @ gradient
    # The population outcome is independent of N(0,I) scores, so its true
    # gradient and true linear reward gain are exactly zero.
    test_features = rng.normal(size=(20000, dimension))
    test_labels = rng.binomial(1, labels.mean(), size=20000)
    heldout_linear_gain = float(np.mean((test_labels-labels.mean()) *
                                        (test_features @ direction)))
    interpolation = {
        "n": n, "dimension": dimension, "positive_count": int(labels.sum()),
        "empirical_chi": chi, "population_chi": 0.,
        "heldout_linear_gain_estimate": heldout_linear_gain,
        "population_linear_gain": 0.,
        "centered_design_rank": int(np.linalg.matrix_rank(features)),
    }
    assert np.isclose(chi, 1., atol=1e-9)

    # The tangent max-min / minimum-norm duality is classical, not our novelty.
    gradients = np.array([[1., 0.], [.4, 1.], [.5, -.2]])
    radius = .05
    result = minimize(lambda weights: (weights @ gradients) @ inv @
                      (weights @ gradients), np.ones(3)/3,
                      bounds=[(0., 1.)]*3,
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1},
                      method="SLSQP", options={"ftol": 1e-13, "maxiter": 1000})
    assert result.success
    mixed = result.x @ gradients
    norm = np.sqrt(mixed @ inv @ mixed)
    step = np.sqrt(2*radius) * inv @ mixed / norm
    predicted = np.sqrt(2*radius) * norm
    achieved = float(np.min(gradients @ step))
    duality = {
        "kl_quadratic_budget": radius,
        "dual_prediction": float(predicted),
        "minimum_linear_gain": achieved,
        "used_quadratic_kl": float(.5 * step @ fisher @ step),
        "absolute_duality_gap": float(abs(predicted-achieved)),
    }
    assert abs(predicted-achieved) < 1e-6

    # Orthogonal gradients do not imply non-learning: independent Bernoulli
    # logits all improve under exact gradient ascent despite zero cross-cosine.
    theta = np.full(8, -3.)
    initial = float((1/(1+np.exp(-theta))).mean())
    for _ in range(1000):
        probability = 1/(1+np.exp(-theta))
        theta += .5 * probability * (1-probability)
    orthogonal = {"initial_success": initial,
                  "final_success": float((1/(1+np.exp(-theta))).mean()),
                  "cross_prompt_gradient_cosine": 0.}
    assert orthogonal["final_success"] > .99

    output = {"scope": "analytical preflight only; no pretrained model, no NLP experiment",
              "seed": 20261015, "coordinate_change": coordinate,
              "empirical_fisher_interpolation": interpolation,
              "classical_tangent_duality": duality,
              "orthogonal_but_learnable": orthogonal}
    destination = Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
