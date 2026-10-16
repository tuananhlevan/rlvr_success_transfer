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

- **21 CPU tests**, including the eight original migration tests, pass with Torch 2.9.1.
- Setup/run orchestration checks cover automatic defaults, saved configuration reuse and refusal to run after failed installation. Other checks cover interrupted snapshot publication, immutable-file reuse, stale-checkpoint pruning, corrupt snapshot rejection, cross-host generation resume without repeating completed samples, concatenated compressed response evidence, and `PAUSED` status.
- Actual CPU FP16 arithmetic checks compare the split backward path against an unchunked calculation with tied weights; master parameters/moments stay FP32. An injected overflow does not consume an AdamW update.
- A resumable HTTPS LFS download is tested with a simulated ranged response, including separation of repository credentials from object-host requests. Both notebooks' code cells compile. Their outputs are empty, and no credentials are embedded.

**Not verified:** a live Colab/Kaggle installation, end-to-end GPU generation/training in the new environments, T4 peak memory, session throughput or platform persistence under a real VM eviction. The local Torch installation currently reports CUDA unavailable. Setup automatically runs the GPU doctor; then run an initial slice before budgeting the full study. A source GPU/CPU timing estimate cannot establish a cloud completion date.
