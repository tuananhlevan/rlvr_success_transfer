# Ranking–rating fusion study

## Documents

- [Research report](research_report.md): current findings, completed theory and experiments, limitations and next requirements.
- [Research proposal](proposal.md): research question, planned experiments, ablations and completion criteria.
- [Structural theory](evidence/fusion_structural_theorems.md).
- [Statistical theory](evidence/fusion_statistical_theory.md).
- [Novelty matrix](evidence/fusion_theorem_novelty_matrix.md).

The controlled phenomenon and scoped crossover are supported. A standard reconstruction baseline succeeds; a new optimizer and practical language-model impact remain unestablished.

## Canonical results

- [Population results](verification/fusion_sprint1/population.csv).
- [Finite-sample results](verification/fusion_sprint1/finite_sample.csv).
- [Asymptotic coefficients](verification/fusion_sprint1/asymptotic_coefficients.csv).
- [Matched-feedback baseline](verification/fusion_sprint1/reconstruction_baseline.csv).
- [Local-alternative results](verification/fusion_local_alternatives_results.csv).
- [Structural checks](verification/fusion_structural_checks_results.json).
- [High-precision moment checks](verification/fusion_moment_law_checks_results.json).
- [Run provenance](verification/fusion_sprint1/run_manifest.json).

Figures are stored once: [population fit/value](verification/fusion_sprint1/population_fit_value.png) and [sample-budget transfer](verification/fusion_sprint1/sample_budget_transfer.png).

## Reproduce

Run from the workspace root:

```bash
python3 studies/ranking_rating_fusion/verification/fusion_structural_checks.py
python3 studies/ranking_rating_fusion/verification/check_fusion_moment_law.py
/home/levantuananh/anaconda3/envs/probml/bin/python studies/ranking_rating_fusion/verification/run_fusion_experiments.py
/home/levantuananh/anaconda3/envs/probml/bin/python studies/ranking_rating_fusion/verification/run_fusion_baseline_audit.py
/home/levantuananh/anaconda3/envs/probml/bin/python studies/ranking_rating_fusion/verification/fusion_local_alternatives.py
python3 scripts/validate_reports.py
```

The runners overwrite their canonical outputs. Validation uses a temporary LaTeX directory and retains only [one validation result](../../reports/validation_results.json). Configurations are the three `*_config.json` files under `verification/`.

## Literature

[Expanded literature analysis](../../reports/dpo_expanded/01_expanded_literature_analysis.md), [OpenReview audit](../../reports/dpo_expanded/03_openreview_discussion_audit.md), and [source PDFs](../../resources/README.md).
