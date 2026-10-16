# Migration verification

Export prepared on 2026-10-16 from the original research workspace. The destination is `tuananhlevan/rlvr_success_transfer`, branch `main`.

## Retained state

| Item | Verified value |
|---|---|
| Training seed | 1 |
| Committed outer step | 1 |
| Optimizer steps | 16, in all 290 parameter states |
| Optimizer moments | FP32 |
| Checkpoint parameter tensors | 290, FP32 master parameters |
| Committed sampler state | epoch 0, cursor 3,072, generation batches 3 |
| Pending outer step | 2 |
| Pending sampler state | epoch 0, cursor 5,120, generation batches 5 |
| Saved pending groups | 1,314; the next update consumes the first 1,024 and retains the remainder |
| Prior execution failure | CUDA unavailable at step 2 model transfer, before any step 2 optimizer update |

The [artifact manifest](artifacts/manifest.json) contains SHA-256 values for every archive part, the concatenated archive, every original checkpoint file, the compressed pending buffer, and its uncompressed contents. The archive includes model and tokenizer files, full AdamW state, carry groups and commit metadata. Six LFS parts contain approximately 5.5 GB of compressed checkpoint data.

The archived [latest sampling batch](studies/rlvr_success_transfer/reproduction/latest_sampling_groups.jsonl.gz) retains the last 1,024 prompts' complete generated groups, including zero-variance groups. It is **not** a complete response archive for every earlier training batch. The per-prompt reward CSV covers those earlier sampling events. The pending update buffer retains all groups required to continue the interrupted update, including earlier carry groups.

## Checks actually completed

- Restored the complete checkpoint into a different directory under `runs/migration_verification/`; verified all archive and checkpoint hashes and the decompressed pending buffer hash.
- Loaded the restored optimizer on CPU with `torch.load(..., weights_only=True)`; confirmed step 16 for all parameter states and FP32 first/second moments. Inspected the safetensors header and tensor dtypes.
- Eight CPU tests passed: cross-directory restore, overwrite refusal, LFS pointer/corruption rejection, archive path traversal rejection, rebased pending-update execution, preservation after worker failure, recovery after commit without double training, buffer corruption rejection, receipt deduplication, and removal of uncommitted sampling rows. Some tests cover multiple assertions.
- Verified all **23** files in the upstream source SHA-256 manifest.
- Resolved separate training and inference dependency locks for Python 3.11 / Linux x86_64. The inference environment uses vLLM's required Torch 2.9.0 family; training uses Torch 2.9.1.
- Checked shell syntax and the portable driver's CLI. Compiled all **1,518** report math expressions with LaTeX and checked report local links. Build products were temporary; no HTML was generated.

The new environments have not been installed and exercised end to end on the destination GPU. No additional training was started during packaging. On the destination, run setup, `doctor.py --gpu`, and optionally the inference benchmark before committing to a multi-day run. A successful dependency resolution is not a GPU execution test.

## Changes from the original driver

1. Project-relative defaults; configurable run/scratch directories and Python executables. Separate training/inference environments replace inherited Conda and `/tmp` paths.
2. Pinned Hugging Face snapshots download automatically when absent. Restored input rows are reused.
3. Durable pending-update buffers survive restarts; paths are reconstructed on the destination machine.
4. Checkpoint receipts recover driver interruption after an optimizer commit without applying the update twice. A missing latest checkpoint is recovered from the previous checkpoint if interruption occurs during promotion.
5. Sampling rows after the last commit are removed before replay when no complete pending buffer exists.
6. CUDA/BF16 preflight runs before long sampling; optimizer state uses Torch's restricted loader.
7. Continuation reports are written into the chosen run directory. Archived scientific results remain separate.
8. `--max-hours` and `--steps` make execution budgets explicit. Source learning hyperparameters and the default 120-step target are retained.

Historical receipts retain their original paths, PIDs, environment and source hash as provenance. `execution_config.json` records the portable script hash and actual runtime settings for a new launch. Changing the runtime, generation backend or reduction order can alter floating-point trajectories; the source fidelity report already describes the single-GPU adaptation.

## Scientific boundary

The subsequent [cloud notebook adaptation](CLOUD_RUN.md) adds bounded sessions, snapshot persistence and an optional FP16/T4 numerical variant. Its tests and limits are reported separately; it does not turn the original BF16 checkpoint into a completed GPU reproduction.

The export enables continuation. It does not complete the reproduction, validate a new method, or establish a representation barrier. The required next evidence remains validation convergence, independent population evaluations, positive-reward exposure accounting, independent cohort confirmation, ordinary-repair controls, and eventual three-seed robustness.
