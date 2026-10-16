# Run in Colab or Kaggle

Select a GPU runtime first. On Kaggle, enable Internet and Files-only persistence. Then run these commands in the platform's code cells:

```python
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/tuananhlevan/rlvr_success_transfer.git /tmp/rlvr_success_transfer
%cd /tmp/rlvr_success_transfer
%run setup.py
!python run.py
```

`setup.py` detects Colab/Kaggle and the GPU, installs the isolated Python environments, runs GPU checks, downloads the pinned model/data or restores the checkpoint, and saves the run configuration. It mounts Google Drive on Colab; approve the normal Drive authorization prompt. `%run` executes setup in the notebook kernel so that authorization can open. `run.py` reads the saved configuration and starts or resumes training.

You do not need to upload or edit a supplied notebook, install dependencies separately, or pass the driver's internal arguments. The `.ipynb` files are optional examples. The root setup/run files are the standard entry points.

## Automatic defaults

| GPU | Precision | First run |
|---|---|---|
| Native BF16, such as L4/A100 | BF16 forward, FP32 master parameters and AdamW | Resume the exported seed-1 checkpoint |
| T4 | FP16 forward, FP32 master parameters/AdamW and loss scaling | Fresh seed-1 FP16 experiment |

T4 results are labelled as an FP16 numerical variant and kept separate from the BF16 reproduction. The full population, eight responses per group, 5,120-token cap, learning rate, logical batch weighting and 120-step scientific target are retained. Physical transformer batches and inference concurrency are reduced for cloud runtimes.

Setup uses disposable storage for environments, model caches and the active run. It saves verified session snapshots to Drive on Colab or `/kaggle/working` on Kaggle. Running setup again reuses an existing local run or restores its saved snapshot. After a new runtime starts, repeat clone, setup and run.

The original reproduction remains incomplete. Passing setup or finishing one session slice does not establish persistent failure or a new research method.

## Setup progress and retry

Setup streams installer output into the cell, prints an elapsed-time heartbeat every 20 seconds, and records `.cloud/setup.log` plus `.cloud/setup_status.json`. The log appends across retries. A heartbeat means the wrapper is alive; use the installer output to see whether downloads or checks are advancing.

The first installation downloads several gigabytes of Torch/CUDA/vLLM dependencies. BF16 also installs an approximately 1 GB prebuilt kernel cache, downloads the 5.5 GB checkpoint, verifies/extracts approximately 6 GB, and copies the initial snapshot to Drive. Download speed, local storage and Drive throughput determine setup time. The two environments are required by the pinned train/inference Torch versions.

If an older checkout only prints `Setting up colab`, interrupt the setup cell, then run:

```python
!git pull --ff-only
%run setup.py --restart
```

`--restart` stops leftover installer/download/prepare processes belonging to this checkout and retains completed downloads and caches. It refuses to kill an active checkpoint restore; let that process finish before retrying. A matching completed environment installation is reused, and GPU checks run again. A partially installed environment is checked against the pinned requirements using the existing package cache. Do not reset the runtime or delete caches just to retry.

### Missing Drive snapshot objects

If a saved manifest references missing or corrupt blobs, setup checks whether their exact hashes are present in the trusted export manifest. For matching objects it downloads/reuses the export, verifies and extracts it, repairs only those blobs, then retries restoration of the original saved manifest. It preserves the saved model, rollout buffers, manifests and later sampling evidence; it does not restart the experiment from the export. Existing dependency installations are reused.

For example, blob `266fbb3cfcea1212a9a23e40af9244a7630e5bebac6396c7b4589388ca7ea91c` is the exported step-1 `optimizer.pt` (3,952,505,547 bytes). Its absence from one reported Drive vault is recoverable from the repository export. This does not identify why Drive lost or failed to retain that object. Recovery may need the 5.5 GB artifact download, temporary local extraction space and another verified optimizer upload.

Recovery stops if any diagnosed blob has no exact export match, or if the source is a read-only attached input. Saved state is retained for diagnosis. No model or optimizer state is substituted by filename or training-step number. Restore verifies the actual local copied bytes before accepting the session, and reports missing files before copying the large checkpoint.

Once setup reports completion, run `!python run.py`.

## Run progress and finish estimates

`run.py` prints the planned new-phase cutoff and session budget deadline at launch. Every minute it prints elapsed time, the current phase, committed steps out of 120, and the remaining session budget. These messages appear in the cell and `RUN/live.log`.

The training ETA starts as **calibrating**. After a complete sampling, optimization and checkpoint-publication cycle in this invocation, it estimates remaining training compute from the latest five complete cycles. It excludes resumed partial cycles and timings from previous invocations, which may have run on another GPU. With only one cycle, the estimate is provisional. It counts the current incomplete cycle conservatively and prints a projected finish time assuming uninterrupted execution, plus a notice when more sessions are needed.

**This is a training ETA, not a full experiment finish estimate.** Periodic validation, final/confirmation evaluations, future setup and runtime downtime are excluded and identified in the output. Evaluation workload and generation lengths can change, so a full experiment completion date remains unestimated. Reaching 120 training steps does not mean all evaluations are finished. The session's pause margin and deadline describe the bounded invocation; checkpoint publication and an early platform disconnect can affect the actual finish time.

If a worker fails, the driver prints that worker's last 100 log lines (up to 16 KiB) into the cell and `RUN/live.log`. The full worker output remains in `RUN/worker.log`. For older versions that only report an exit code, read that file to identify the underlying exception; the wrapper traceback does not establish its cause. Keep the checkpoint and pending rollout buffer while diagnosing a resume failure.

### Worker cannot import Torch after preflight passed

An earlier driver dereferenced the virtual environment's `bin/python` symlink while selecting worker interpreters. That launched workers with the underlying base Python, which can lack Torch even though the driver and GPU doctor use the correct environment. The driver now preserves executable symlinks for both training and inference, and prints the selected Python path with each worker PID.

For this error, keep the current runtime and run directory, pull the latest code, then rerun `!python run.py`. Setup and checkpoint downloads need not be repeated when they already completed. A CPU regression test launches actual workers in two isolated virtual environments through the same directory/executable symlink layout; it checks each worker imports its own environment's module.

## Optional overrides

```python
# Start a fresh experiment on a BF16 GPU instead of downloading the exported checkpoint.
%run setup.py --fresh

# Explicitly continue the BF16 step-1 export on T4; records a precision transition.
%run setup.py --from-checkpoint

# Resume a previous Kaggle notebook's saved output, attached as input.
%run setup.py --resume-from /kaggle/input/PREVIOUS_OUTPUT/vault_float16_seed1

# Change this session's time budget without changing the scientific target.
!python run.py --hours 8
```

`setup.py --vault PATH` selects another persistent location. `--artifact-dir PATH` uses an attached directory containing `manifest.json`, all six checkpoint parts and `pending_update_buffer.json.gz`. Otherwise setup downloads and verifies existing LFS objects over HTTPS when checkpoint migration is selected. Subsequent sessions restore the saved vault without downloading the export again. Repository credentials, when required, can be supplied through the `GITHUB_TOKEN` environment variable.

## Sessions and persistence

The default training budget is **six hours per invocation**, with a **120-minute margin before starting another phase**. Setup and uploads also consume platform time. No notebook runtime duration is guaranteed. Colab documents variable resource/session limits; Kaggle documents a 12-hour GPU session and 20 GB saved-output allocation. Kaggle's September 2026 notice retires P100 and retains T4 x2. These constraints are why a multi-day training job is split across sessions. [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [Kaggle hardware notice](https://www.kaggle.com/product-announcements/735239).

### What is saved

- The latest committed model, full FP32 AdamW state, scaler state when applicable, sampler state and carry groups.
- Partially collected contributing groups and their exact sampling cursor.
- Compressed, immutable generation chunks plus their hashes. A resumed sampling/evaluation request verifies and reuses completed chunks, including after a host/path change.
- Exact input population, numerical protocol, reward/validation curves and completed independent response evidence.

Mutable live logs and process/PID receipts remain local. Read `RUN/worker.log` and `RUN/live.log` during execution. The vault contains stable scientific state, not a portable live process. Its `CURRENT.json` identifies the latest verified manifest.

The vault is a content-addressed store: unchanged model/optimizer files are reused when a small sampling chunk is added. Publishing writes new blobs before changing the manifest pointer. Interrupted copying leaves the previous complete snapshot available. After a successful publish, obsolete files/manifests are pruned. **Only the latest snapshot is retained.** Preserve a separate milestone if you need one.

Reserve roughly **13 GB free** for old-plus-incoming checkpoint files during replacement, plus evaluation evidence; 20 GB provides more room. Typical training-only saved state is about 6 GB before accumulated response evidence. The original 5.5 GB compressed LFS download stays in disposable storage, not in the Kaggle output vault. Bootstrap also requires at least 25 GiB disposable local space. The full optimizer needs substantial host RAM; a high-memory Colab runtime is preferable to the smallest free host. Neither peak cloud RAM nor T4 throughput has been measured here.

### Resume

- **Colab:** start another GPU runtime, mount the same Drive, and rerun setup and run with the same seed/precision/vault. The latest snapshot is verified and restored into local storage.
- **Kaggle, same editing session:** reuse Files-only persistence and rerun setup and run.
- **Kaggle, clean version/session:** save the previous outputs, attach that notebook output as input, and pass its `vault_float16_seed1` directory to `setup.py --resume-from`. Restoration reads the attached snapshot; the new session publishes to its own `/kaggle/working` vault. A new **Save & Run All** starts clean, so explicitly attach the prior outputs. [Kaggle persistence notice](https://www.kaggle.com/discussions/product-feedback/355440), [notebook version documentation](https://www.kaggle.com/docs/notebooks).

`PAUSED` is a successful session slice with unfinished scientific work. **A shorter wall-clock slice does not shorten the 120-step target.** Do not replace it with `--steps 2` and treat the result as the full reproduction. If a session ends during optimization, the uncommitted outer update replays from the saved checkpoint and rollout buffer; individual inner AdamW updates are not separately checkpointed. The margin reduces this risk, but a slow phase or early platform termination can still lose work since the last publish. Mounted Drive persistence is not a transactional local filesystem, so a hard disconnect can also leave an incomplete upload; restore verifies hashes rather than trusting file existence.

Do not run two writers against the same vault or change numerical settings inside one experiment. The driver rejects changes in the saved numerical protocol. If changing hardware requires a precision change, create a separate experiment.

## FP16 implementation

The master parameters, gradients and AdamW moments stay FP32. All forward parameters and activations use the selected low precision. In FP16 mode a dynamic scaler starts at 128. The chunked vocabulary losses are scaled once; their already scaled hidden-state gradients are backpropagated into the transformer without applying the scale again. Parameter gradients are unscaled once before clipping and the optimizer update.

An overflow lowers the scale and retries the **same** logical minibatch, without consuming an AdamW step. Repeated nonfinite gradients fail explicitly. Successful outer updates still contain 16 actual optimizer steps. The scaler state is saved with the model. These safeguards support an engineering FP16 variant; they do not make it numerically equivalent to BF16 or constitute a research-method contribution. The pinned inference stack retains a Turing-compatible backend; upgrading vLLM indiscriminately can remove that support. [vLLM 0.11.2 CUDA backend source](https://docs.vllm.ai/en/v0.11.2/api/vllm/platforms/cuda/).

## Verified here

- **37 CPU tests**, including the eight original migration tests, pass with Torch 2.9.1.
- Setup/run orchestration checks cover automatic defaults, saved configuration reuse and refusal to run after failed installation. Other checks cover interrupted snapshot publication, immutable-file reuse, stale-checkpoint pruning, corrupt snapshot rejection, cross-host generation resume without repeating completed samples, concatenated compressed response evidence, and `PAUSED` status.
- Actual CPU FP16 arithmetic checks compare the split backward path against an unchunked calculation with tied weights; master parameters/moments stay FP32. An injected overflow does not consume an AdamW update.
- A resumable HTTPS LFS download is tested with a simulated ranged response, including separation of repository credentials from object-host requests. Both notebooks' code cells compile. Their outputs are empty, and no credentials are embedded.
- Setup checks cover streamed stdout/stderr, timed heartbeats, failed/interrupted status, child termination on interruption, completed-install reuse with repeated GPU checks, and exclusion of training commands from the restart matcher.
- ETA checks cover calibration, exclusion of other invocations, recent-cycle averaging, session-budget messages, and distinguishing completed training from unfinished evaluation. No cloud timing accuracy has been measured.
- Worker diagnostics checks verify bounded failure output and exclusion of earlier workers' tracebacks.
- Snapshot recovery checks reproduce a missing exported optimizer blob, preserve newer sampling evidence and the manifest pointer, reject unavailable exact hashes and incorrect recovery bytes, detect corruption during restore, and leave existing state intact after an interrupted repair. Real Drive durability remains unverified.
- Worker environment checks launch actual train/infer fixture processes through virtual-environment executable symlinks and verify each isolated import path. Restoring the old symlink resolution makes this regression test fail. These fixtures do not run GPU training.

**Not verified:** a live Colab/Kaggle installation, end-to-end GPU generation/training in the new environments, T4 peak memory, session throughput or platform persistence under a real VM eviction. The local Torch installation currently reports CUDA unavailable. Setup automatically runs the GPU doctor; then run an initial slice before budgeting the full study. A source GPU/CPU timing estimate cannot establish a cloud completion date.
