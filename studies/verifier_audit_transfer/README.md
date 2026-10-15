# Verifier auditing and transfer

**Status: rejected as an active paper direction, 15 October 2026.** Standard baselines repair the constructed failure. The results are retained as a negative finding; they do not justify a new method.

The proposed direction investigates why auditing selected code candidates can fail to improve a verifier shared across problems. The current experiment tests a possible mechanism and shows that standard baselines repair it. It does not reproduce the reported failure on a language model.

- [Research report](research_report.md): evidence, paper intuitions, rejected alternatives, proofs, results and the historical conditional proposal.
- [Experiment runner](run_experiments.py) and [configuration](experiment_config.json).
- [Aggregate results](results.csv), [figure](results.png) and [execution manifest](run_manifest.json).

Reproduce from the workspace root:

```bash
/home/levantuananh/anaconda3/envs/probml/bin/python studies/verifier_audit_transfer/run_experiments.py
python3 scripts/validate_reports.py
```

The runner overwrites the canonical results. No HTML, worker drafts, raw-trial files or build logs are retained. This study is separate from the [previous fusion study](../ranking_rating_fusion/README.md).
