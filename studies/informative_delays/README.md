# Research screen: informative completion delays

**Status, 15 October 2026:** deprioritized as an active paper direction after practical-solution review. Existing theory addresses the general obstruction and buffering provides an established repair. The scoped lower bound is retained; its precise novelty and a consequential unresolved practical cost are not established. No new optimizer is recommended. The older-version proof correction is already addressed by the authors' October revision.

- [Canonical research report](research_report.md): literature intuitions, version audit, original information lower bound, executed experiments, provisional research plan and rejection criteria.
- [Reproducible runner](run_experiments.py), [paired trial results](results.csv), [figure](results.png), and [configuration, analytical checks and execution manifest](run_manifest.json).

Reproduce from the workspace root:

```bash
/home/levantuananh/anaconda3/envs/probml/bin/python studies/informative_delays/run_experiments.py
python3 scripts/validate_reports.py
```

The runner overwrites the canonical results. Experiments use a scalar synthetic objective and a virtual event queue; no LLM training or measured GPU throughput is claimed. No HTML, duplicate report, raw trajectory dump or build products are retained.
