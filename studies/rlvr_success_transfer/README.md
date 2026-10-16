# Learning from successful RLVR rollouts

**Status:** five full-parameter GPU pilot arms completed. Ordinary SFT repairs the sampled cases. The published persistent-failure cohort has not been reproduced; a new research-method recommendation remains withheld.

**Compute decision, 16 October:** the [research investment review](research_review.md) recommends preserving the current checkpoint and requiring a bounded feasibility test before further multi-session training. A full baseline reproduction alone would not establish the intended new mechanism, repair or theorem.

The question is: **when hard prompts already produce correct solutions, what prevents their learning from transferring, and what intervention actually removes that bottleneck?**

- [Actual-failure reproduction: protocol, fidelity audit, and current status](reproduction_report.md). Full source population and 5,120-token responses; the reproduction must pass independent learning gates before any method recommendation.
- [Research brief, prior-art audit, theory and experiment gates](research_report.md).
- [Canonical analytical checks](experiment.py) and [results](results.json). These are mathematical preflight checks, not NLP experiments.
- [GPU access diagnostics](gpu_preflight.json): device access is now working on the RTX 5060 Ti 16 GB; model and dataset downloads succeeded directly.
- [Full-parameter GPU pilot](gpu_pilot.py), [fixed input sample](gpu_pilot_inputs.json), and [canonical results](gpu_pilot_results.json). This pilot tests short interventions, not persistent unlearnability or the full source recipe.
- [Pinned pilot dependencies](gpu_requirements.txt). The executed environment inherits the existing CUDA 12.9 PyTorch build.
- [Frozen pilot protocol and added fresh-rollout control](gpu_pilot_protocol.json).

The brief incorporates the September 2026 reanalysis challenging the original “unlearnable” classification. It excludes a generic gradient reward, another difficulty scheduler, and a renamed natural-gradient method as proposed contributions.

## Portable execution

Use the [root setup and checkpoint continuation instructions](../../README.md). They replace the original machine-specific Conda and `/tmp` environment commands. The full-data run failed at step 2 CUDA initialization; step 1 and the pending step 2 buffer are packaged for continuation.

The pilot dependency receipt records the environment that produced the archived pilot. It is historical; the portable training/inference requirements are under `requirements/`. Running `gpu_pilot.py` again can overwrite the canonical pilot result, so preserve the archive before a new pilot. Analytical checks can be rerun with `.venv/train/bin/python studies/rlvr_success_transfer/experiment.py`; they do not establish an NLP learning result.
