#!/usr/bin/env python3
"""Finite-matrix checks for the conditional-design limit, not an LLM experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def check(d):
    eta = 1 / d
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    rows = [check(d) for d in [2, 4, 8, 16, 32]]
    result = {"date": "2026-10-16", "status": "theory lead; practical value unestablished",
              "meaning": "Deterministic covariance calculations and numerical prompt-design optimization. No LLM training, human labels, or deployment outcomes.",
              "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"d={row['dimension']:2}: fixed conditional={row['conditional_design_leverage']:.3f}; "
              f"joint={row['joint_design_leverage']:.3f}; ratio={row['ratio']:.3f}")


if __name__ == "__main__":
    main()
