# RLVR success transfer: research and reproduction

Research notes, paper resources, completed pilot results, and a portable continuation of a full-data RLVR reproduction of [The Unlearnability Phenomenon in RLVR for Language Models](https://github.com/yulinchen99/unlearnability-rlvr).

## Current result

**Persistent failure has not yet been reproduced locally.** Ordinary SFT repaired the sampled pilot cases. The larger source-based experiment completed **one outer GRPO step (16 AdamW updates)** before CUDA became unavailable at the start of step 2 optimization. No step 2 optimizer update completed.

The export includes the step 1 model, full optimizer state, carry groups, exact input population, and saved step 2 rollout buffer. Resuming applies that pending update before generating new samples. The original run's starting greedy validation was **125/500 (25%)**; there is no final learning result yet. See the [reproduction report](studies/rlvr_success_transfer/reproduction_report.md) and [archived failure status](studies/rlvr_success_transfer/reproduction/status.json).

The [16 October research investment review](studies/rlvr_success_transfer/research_review.md) recommends a bounded feasibility gate before additional multi-session training. The current evidence does not justify a multi-day methods/theory campaign. This recommendation does not stop an active run or change its saved protocol.

## Latest direction screen

The [exploration theory sprint](studies/conditional_design_limits/README.md) records exact design limits, matching sparse-family resource rates, and a correction distinguishing generated pairs from queried labels. Ordinary screening repairs its label bottleneck; practical LLM value and full-paper novelty remain unestablished. The [synthetic-error context study](studies/error_context_transfer/README.md) is retired after the direct prior-art check.

## Quick start: Colab or Kaggle

Select a GPU runtime, then run:

```python
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/tuananhlevan/rlvr_success_transfer.git /tmp/rlvr_success_transfer
%cd /tmp/rlvr_success_transfer
%run setup.py
!python run.py
```

Setup detects the platform and GPU, installs dependencies, prepares/restores data and checkpoints, and configures persistence. Run starts or resumes a six-hour session slice. On Colab, approve setup's Drive mount prompt. On Kaggle, enable Internet and Files-only persistence. No supplied notebook is required. See the [cloud guide](CLOUD_RUN.md) for optional settings and resuming prior Kaggle outputs.

Setup streams progress into the cell and saves `.cloud/setup.log`, with an elapsed-time heartbeat every 20 seconds. The first setup downloads several gigabytes of dependencies and, for default BF16, a 5.5 GB checkpoint before saving it to Drive. For an old silent setup, interrupt the cell, run `!git pull --ff-only`, then `%run setup.py --restart` to retry with retained caches. See [setup progress and retry](CLOUD_RUN.md#setup-progress-and-retry).

Run prints session deadlines and a progress update every minute. Its training ETA calibrates from complete cycles on the current invocation; validation/final evaluations and runtime downtime are explicitly excluded. See [finish estimates](CLOUD_RUN.md#run-progress-and-finish-estimates).

T4 automatically uses a labelled FP16 variant and starts a fresh seed. L4/A100 uses BF16 and resumes the exported seed-1 checkpoint by default. The 120-step scientific target is retained. Cloud GPU execution remains unverified here.

## Dedicated Linux machine

The desktop commands below remain available for a dedicated machine.

- Linux x86_64, Python **3.11** with `venv`, Git and Git LFS.
- One NVIDIA GPU supporting BF16. The original run used an RTX 5060 Ti **16 GB**. Training and inference execute in separate processes to release VRAM between phases.
- An NVIDIA driver compatible with CUDA 12.8 wheels. Check `nvidia-smi`; a working desktop GPU alone does not establish Torch CUDA access.
- A C++ compiler and the selected Python 3.11 interpreter's development headers. For first-use kernel compilation, install **CUDA Toolkit 12.8** and expose `nvcc` through `CUDA_HOME`/`PATH`. The Torch wheels supply runtime libraries; FlashInfer can compile kernels at runtime. Matching prebuilt FlashInfer caches are an alternative, described in its [version 0.5.2 installation guide](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.2/docs/installation.rst).
- At least **32 GB host RAM** and approximately **50 GB free disk**, allowing for both environments, wheel caches, downloaded base weights, LFS objects, and the restored checkpoint.
- Internet access to PyPI, the PyTorch wheel index, Hugging Face, and GitHub. The pinned model/dataset are public. You can set `HF_HOME` to choose a cache location.

The environments are separate because vLLM 0.11.2 requires Torch **2.9.0**, whereas training uses Torch **2.9.1**. Setup installs the matching inference Torch family; it does not inherit the old machine's Conda environment. CUDA 12.8 differs from the original machine's CUDA 12.9 runtime. Floating-point behavior can change; this is a continuation of the documented single-GPU adaptation, not a bitwise reproduction.

## Continue on another machine

Install Git LFS using its [official installation instructions](https://git-lfs.com/). Downloading the migration checkpoint uses approximately **5.5 GB** of LFS bandwidth. The six parts are below 2 GB each. Install Git LFS **before** downloading them, then:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:tuananhlevan/rlvr_success_transfer.git
cd rlvr_success_transfer
git lfs install --local
git lfs pull

# Set PYTHON_BIN if python3.11 is not on PATH.
PYTHON_BIN=python3.11 bash scripts/setup.sh
.venv/train/bin/python scripts/doctor.py --gpu

# Refuses to overwrite a nonempty run directory; verifies all artifact hashes.
python3 scripts/artifacts.py restore --run-dir runs/reproduction

mkdir -p runs/reproduction
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh run --seed 1 \
  --run-dir runs/reproduction --max-hours 240 2>&1 | tee -a runs/reproduction/live.log
```

Run the long job inside `tmux` or another persistent session. **240 hours is a wall-clock budget**, not a guaranteed completion time. At the original measured speed, 120 training steps plus independent evaluations were provisionally estimated at **8–10 days per seed**. Three seeds multiply that cost. Repeated restarts reset the launch's time budget; they retain committed progress. The code defaults to 96 hours if `--max-hours` is omitted.

After the first restoration, resume by running the final command again. **Do not restore again.** A committed update is recovered without retraining; an interrupted uncommitted update is rerun from its starting checkpoint with its saved rollout buffer. If interruption occurs before a buffer is fully persisted, that unfinished sampling step is replayed and its uncommitted CSV rows are removed. GPU generation need not be bitwise deterministic across machines.

The initial reference model is downloaded at its pinned revision when needed. `inputs.json` restored from the snapshot fixes the 8,138 eligible training prompts and 500 validation prompts. Optimizer loading uses Torch's restricted `weights_only=True` loader; only use checkpoint artifacts from a repository/commit you trust.

### Logs and outputs

```bash
tail -f runs/reproduction/live.log runs/reproduction/worker.log
cat runs/reproduction/status.json
cat runs/reproduction/seed_1/state.json
```

- `worker.log`: detailed generation/optimization progress and errors.
- `status.json`: current/last phase or failure status. Worker PID receipts describe the **current machine** only.
- `seed_1/latest_checkpoint/commit.json`: authoritative committed training state.
- `seed_1/pending_update/`: durable rollouts awaiting a committed update.
- `seed_1/prompt_reward_stats.csv`, `optimization_curve.jsonl`, `validation_curve.jsonl`: measured curves.
- `execution_config.json`: resolved paths, configuration and executed script hash.
- After training and independent evaluations: `seed_1/analysis.json` and `report.md`.

The files under `studies/.../reproduction/` are the **archived original run**, including obsolete paths/PIDs in its receipts. New runs write to `runs/`, which Git ignores. The portable driver never writes continuation results into the archived report.

## Fresh runs and checks

```bash
# Source integrity and CPU-only resume/transfer tests; no model download or GPU.
python3 scripts/doctor.py --source-only
python3 -m unittest discover -s tests -v

# Prepare public model/data in a new run directory; no training.
bash scripts/run.sh prepare --run-dir runs/fresh_seed1

# Optional inference throughput benchmark, ~minutes on the original GPU.
# Its 512 long responses make it larger than a minimal CUDA smoke check.
bash scripts/run.sh benchmark --run-dir runs/benchmark

# Train from the base model, with the source's target of 120 outer steps.
bash scripts/run.sh run --seed 1 --run-dir runs/fresh_seed1 --max-hours 240
```

Seeds 42 and 37 require separate fresh directories and full runs. Run them sequentially on one 16 GB GPU. `--steps` can shorten an engineering check, but a shortened run cannot establish the paper's persistent-failure claim. The doctor checks a real BF16 matmul, native vLLM CUDA operation and FlashInfer kernel when `--gpu` is supplied. First-use kernel compilation can take several minutes. Only a successful benchmark/run validates the complete generation stack on a new host.

For a model already downloaded elsewhere, set `RLVR_MODEL_PATH` to its directory. Preserve the pinned model revision. Advanced overrides are `RLVR_SOURCE_DIR`, `RLVR_SCRATCH_DIR`, `RLVR_TRAIN_PYTHON`, and `RLVR_INFER_PYTHON`; `scripts/run.sh` selects the two project environments. `--scratch-dir` can move temporary generation files to another disk. Avoid sharing a run directory between concurrent processes.

## Repository map

| Path | Contents |
|---|---|
| [Current study](studies/rlvr_success_transfer/README.md) | Proposal, novelty checks, analytical work, pilot results and reproduction protocol |
| [Research report](studies/rlvr_success_transfer/research_report.md) | Research question, related work, theory and decision gates |
| [Reproduction driver](studies/rlvr_success_transfer/reproduce_failure.py) | Full-data training, validation, independent evaluation and resumable updates |
| [Artifacts manifest](artifacts/manifest.json) | SHA-256 checksums, committed state and pending update metadata |
| [Literature reports](reports/dpo_expanded/README.md) | Literature analysis, intuition, proposals and subsequent screening |
| [Resources](resources/README.md) | Paper PDFs and source provenance |
| `studies/` | Earlier directions, their evidence and rejection/qualification context |
| `requirements/`, `scripts/`, `tests/` | Portable setup, launch, diagnostics, artifact migration and CPU tests |

Historical proposals are research records, not endorsed conclusions. Read the latest study status before interpreting older claims. The upstream source is pinned by commit and SHA-256. The reproduction report documents sampling-batch inconsistencies, precision, reward extraction, evaluation and one-GPU adaptations.

## Verification and provenance

See [migration verification](MIGRATION.md) for what was actually tested. Top-level dependencies and resolved transitive lock files are pinned in `requirements/` for Linux x86_64 / Python 3.11. Setup installs those locks and runs `pip check`. For a new completed environment, retain `pip freeze` alongside that run if exact transitive versions are needed.

Upstream wheel installation references: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) and [vLLM 0.11.2 package metadata](https://pypi.org/project/vllm/0.11.2/). Attribution and third-party material are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
