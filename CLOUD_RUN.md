# Colab and Kaggle execution

## Choose a notebook

| Runtime | Notebook | Numerical recipe |
|---|---|---|
| Colab L4/A100 or another GPU with native BF16 | [Colab notebook](notebooks/rlvr_colab.ipynb) | BF16 forward, FP32 master parameters and AdamW |
| Colab T4 | Same notebook, explicitly set `PRECISION="float16"` | FP16 variant, FP32 master/AdamW, loss scaling |
| Kaggle T4 x2 | [Kaggle notebook](notebooks/rlvr_kaggle.ipynb) | FP16 variant using GPU 0 only |

**A T4 run is not the original BF16 numerical recipe.** Its settings and result files are labelled accordingly. Do not pool it with BF16 reproduction seeds. The notebook defaults preserve the full population, eight responses per group, 5,120-token cap, logical batch/loss weighting, learning rate, 16 AdamW updates per outer step, and 120-step scientific target. Physical transformer batches shrink to one; inference concurrency and request chunks also shrink. These change floating-point reduction order and generation scheduling.

This remains a reproduction attempt. The local pilot was repaired by ordinary SFT, and persistent failure has not yet been reproduced. Running a notebook slice or passing a GPU check does not change that conclusion.

## First session

1. Upload the appropriate `.ipynb` through Colab's **File → Upload notebook** or Kaggle's notebook import UI.
2. Select a GPU. For the BF16 path, use L4/A100 or another Ampere-or-newer GPU. For Kaggle, select the available T4 option and enable Internet.
3. Configure the first cell. If GitHub is private, create a platform secret named `GITHUB_TOKEN` with repository read access and enable `USE_GITHUB_SECRET`. No SSH key is required. Public repositories need no token.
4. Colab: mount Drive. Kaggle: enable **Files only persistence before running**. Only the vault is written under `/kaggle/working`; code, virtual environments, caches and the active run stay in disposable storage.
5. Run the code and setup cells. The bootstrap installs managed **Python 3.11.14** and the locked training/inference environments without replacing the notebook kernel packages. The GPU doctor must pass before training. On BF16 GPUs it checks the pinned FlashInfer cache/kernel path; T4 checks xFormers attention instead.
6. Choose fresh training or checkpoint continuation, then run a session slice. The Colab notebook defaults to the exported step-1 continuation; Kaggle defaults to a **fresh FP16 seed**. An FP16 continuation of the BF16 checkpoint is explicitly recorded as a precision transition.

If these changes are not yet available from GitHub, use the supplied **`notebook_support.zip`** overlay. Colab prompts for it when the helper script is absent. On Kaggle, attach the ZIP as input and set `CODE_OVERLAY` to its path. The overlay contains code/documentation, not the multi-gigabyte checkpoint. The notebooks then use the already exported model/data resources from the repository.

Checkpoint restoration can fetch existing LFS objects through HTTPS without installing Git LFS. It verifies every part against the manifest and can resume partial downloads. Alternatively, attach the manifest, all six checkpoint parts, and `pending_update_buffer.json.gz` as a Kaggle dataset and set `ARTIFACT_DIR`. Resuming an existing vault does not download the original checkpoint again.

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

- **Colab:** start another GPU runtime, mount the same Drive, and rerun the notebook with the same seed/precision/vault. The latest snapshot is verified and restored into local storage.
- **Kaggle, same editing session:** reuse Files-only persistence and rerun the notebook.
- **Kaggle, clean version/session:** save the previous outputs, attach that notebook output as input, and set `PRIOR_VAULT` to its `vault_float16_seed1` directory. Restoration reads the attached snapshot; the new session publishes to its own `/kaggle/working` vault. A new **Save & Run All** starts clean, so explicitly attach the prior outputs. [Kaggle persistence notice](https://www.kaggle.com/discussions/product-feedback/355440), [notebook version documentation](https://www.kaggle.com/docs/notebooks).

`PAUSED` is a successful session slice with unfinished scientific work. **A shorter wall-clock slice does not shorten the 120-step target.** Do not replace it with `--steps 2` and treat the result as the full reproduction. If a session ends during optimization, the uncommitted outer update replays from the saved checkpoint and rollout buffer; individual inner AdamW updates are not separately checkpointed. The margin reduces this risk, but a slow phase or early platform termination can still lose work since the last publish. Mounted Drive persistence is not a transactional local filesystem, so a hard disconnect can also leave an incomplete upload; restore verifies hashes rather than trusting file existence.

Do not run two writers against the same vault or change numerical settings inside one experiment. The driver rejects changes in the saved numerical protocol. If changing hardware requires a precision change, create a separate experiment.

## FP16 implementation

The master parameters, gradients and AdamW moments stay FP32. All forward parameters and activations use the selected low precision. In FP16 mode a dynamic scaler starts at 128. The chunked vocabulary losses are scaled once; their already scaled hidden-state gradients are backpropagated into the transformer without applying the scale again. Parameter gradients are unscaled once before clipping and the optimizer update.

An overflow lowers the scale and retries the **same** logical minibatch, without consuming an AdamW step. Repeated nonfinite gradients fail explicitly. Successful outer updates still contain 16 actual optimizer steps. The scaler state is saved with the model. These safeguards support an engineering FP16 variant; they do not make it numerically equivalent to BF16 or constitute a research-method contribution. The pinned inference stack retains a Turing-compatible backend; upgrading vLLM indiscriminately can remove that support. [vLLM 0.11.2 CUDA backend source](https://docs.vllm.ai/en/v0.11.2/api/vllm/platforms/cuda/).

## Verified here

- **17 CPU tests**, including the eight original migration tests, pass with Torch 2.9.1.
- New checks cover interrupted snapshot publication, immutable-file reuse, stale-checkpoint pruning, corrupt snapshot rejection, cross-host generation resume without repeating completed samples, concatenated compressed response evidence, and `PAUSED` status.
- Actual CPU FP16 arithmetic checks compare the split backward path against an unchunked calculation with tied weights; master parameters/moments stay FP32. An injected overflow does not consume an AdamW update.
- A resumable HTTPS LFS download is tested with a simulated ranged response, including separation of repository credentials from object-host requests. Both notebooks' code cells compile. Their outputs are empty, and no credentials are embedded.

**Not verified:** a live Colab/Kaggle installation, end-to-end GPU generation/training in the new environments, T4 peak memory, session throughput or platform persistence under a real VM eviction. The local Torch installation currently reports CUDA unavailable. Run the notebook GPU doctor and then an initial slice before budgeting the full study. A source GPU/CPU timing estimate cannot establish a cloud completion date.
