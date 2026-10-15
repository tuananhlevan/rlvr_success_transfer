# Policy capacity and preference feedback

**Status: theorem retained; full-project recommendation withheld after the novelty and practical-repair audit.** The broad premise is addressed in prior work, and the contextual example is repaired by a prompt-aware feature with the same parameter count. The continuous-class theorem remains a possible structural contribution; its publication novelty and practical necessity are not established. See Section 11 of the report.

- [Research proposal, proofs, results, and next experiments](research_report.md)
- [Canonical reproducible experiment](experiment.py)
- [Canonical numerical results](results.json)
- [Figure](feedback_complexity.svg)

The experiments are exact or numerical finite-model checks. They contain no real annotator data or LLM training results. This study is separate from the rejected ranking/rating fusion, verifier-audit, delay, and router proposals.

```bash
MPLCONFIGDIR=/tmp/policy-capacity-mpl /home/levantuananh/anaconda3/envs/probml/bin/python studies/policy_capacity_feedback/experiment.py
python3 scripts/validate_reports.py
```
