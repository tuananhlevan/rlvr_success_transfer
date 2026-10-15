# Reproducing persistent RLVR failure

## Material Passport

- Type: code experiment; source-based reproduction.
- Status: implementation and resource checks completed; training reproduction pending.
- Source: Chen, He, and Zhao, [The Unlearnability Phenomenon in RLVR for Language Models](https://arxiv.org/html/2605.16787), ICML 2026; [released implementation](https://github.com/yulinchen99/unlearnability-rlvr).
- Code: [reproduce_failure.py](reproduce_failure.py).
- Evidence: [reproduction/](reproduction/); this is the canonical report for this reproduction. The earlier eight-prompt pilot is a different experiment and does not establish persistent unlearnability.
- Verification status: implementation checks analyzed; no independent end-to-end reproduction completed yet.

## 1. Claim to reproduce

After full-dataset GRPO training has improved and saturated on validation data, some initially hard training problems still have a low independent success rate, despite receiving positive rewards during training. The source reports an intersection over three training seeds, rather than selecting ordinary mixed-reward problems before a short intervention.

Required evidence:

1. Start from the same base model and original training population.
2. Estimate initial per-problem success from 32 independent responses.
3. Train the full population with the released GRPO recipe; log every problem's reward exposure, including discarded zero-variance groups.
4. Measure validation progress during training. A failed or globally ineffective training run cannot establish the claimed selective learning failure.
5. Evaluate training problems independently after training, using 32 fresh responses each.
6. Exclude problems that never received a positive reward. Report positive counts and visits, rather than treating one lucky answer as proof of adequate supervision.
7. Distinguish a single-seed replication from the paper's three-seed intersection. Independently confirm a discovered low-success cohort before arguing that its apparent persistence is more than finite-sample classification noise.

This experiment does not yet test a proposed repair. No new-method claim follows from reproducing a failure.

## 2. Source recovered and pinned

The [source manifest](../../resources/rlvr_success_transfer/source/manifest.json) pins commit `1dce7fec92e7685929a5efbe73ff2091e7d474a1` and hashes retained source files. Downloads are public; no OpenReview API is used.

| Component | Reproduction setting |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Dataset | `hkust-nlp/SimpleRL-Zoo-Data`, revision `50ec8f09045e785fd20c13a508820e77cccb4211`, `simplelr_qwen_level1to4` |
| Eligible population | 8,138 training prompts after the source's 1,024-token prompt filter; 500 validation prompts |
| Training prompts | Exact preformatted dataset strings, including their original chat markers and double braces |
| Response limit | 5,120 tokens |
| Training sampling | Temperature 1, top-p 1, group size 8 |
| Dynamic sampling | Sample 1,024 prompts at a time; discard zero-variance groups; collect 1,024 contributing prompt groups |
| Optimizer | Full-parameter FP32 AdamW, learning rate `5e-7`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, constant schedule, gradient clipping 1 |
| Updates | 512 responses per optimizer step; 16 optimizer steps per 1,024-prompt contributing batch |
| Policy objective | PPO clipping 0.2; KL coefficient `0.0001`, source's clamped low-variance estimator; entropy coefficient `0.001` |
| Reference | Frozen base model |
| Evaluation | Source inference prompt template; temperature 0.7, 32 responses per training problem; separate training and inference scorer outputs retained |
| Validation | Greedy, one response per problem, original dataset prompts and training verifier, matching `ray_trainer._validate` |
| Training horizon | First seed targets step 120 from the source README, bounded by the launcher's 50 data epochs; validation every five steps |
| Seeds | Start with seed 1. Source intersection additionally requires seeds 42 and 37. |

### Material source inconsistencies and adaptations

- **Sampling batch:** Appendix B.1 describes 1,024 sampling prompts for the 0.5B model; `train_baseline.sh` passes 256 as the training/sampling batch and 1,024 as the rollout microbatch. The released CSV resolves which setting better matches the actual reported run: all 146 step row counts are multiples of 1,024, and the first step collects 1,243 mixed-reward groups from 3,072 sampled prompts. The reproduction therefore uses **1,024 sampling and contributing prompts**, following the paper and archived trajectory. This choice was corrected before any training run; the small throughput check did not tune learning hyperparameters.
- **Loss weighting:** the paper describes a response mean after a token mean. The released actor uses token means within 32-response rank microbatches and then averages microbatches and ranks. This implementation follows that code weighting, including the original length-balancing function and four logical ranks.
- **One GPU:** four physical data-parallel ranks become serial gradient accumulation. Transformer batches of four and vocabulary chunks preserve the mathematical loss, with changed floating-point reduction order. SDPA replaces FlashAttention 2 for training; vLLM replaces SGLang for inference. Decode CUDA graphs and longer inference queues improve throughput while preserving the learning recipe. These adaptations prevent a claim of bitwise reproduction.
- **Softmax arithmetic:** the memory adapter computes log-softmax and entropy in FP32 after the BF16 vocabulary projection. The source has backend-dependent cross-entropy arithmetic and a BF16 entropy path. This numerical difference is documented, not silently treated as exact equivalence.
- **Forward precision:** every forward parameter, including embeddings and RMSNorm, is cast to BF16 while master parameters and AdamW remain FP32. This matches the source FSDP parameter-casting policy more closely than autocast alone, which leaves some activations and residuals in FP32. The cast context remains active through gradient-checkpoint recomputation.
- **Evaluation temperature:** the paper gives temperature 1 for sampling; its released `inference/test.sh` uses 0.7. Training uses 1 and independent evaluation uses 0.7, following those scripts.
- **Reward extraction:** training extracts an answer with the source Qwen parser and accepts correct unboxed answers. Source inference parses the full response with Math-Verify. Both outcomes are saved so an extraction mismatch cannot be mistaken for a learning effect.
- **Checkpoints:** the released README refers to final checkpoints and partition exports that are absent from the repository tree. Their labels and models cannot be assumed available. This run must produce its own initial and final evaluations.
- **Evaluation order:** the full-population initial evaluation is collected from the immutable base checkpoint after training, with a fixed independent seed. The questions and sample count are fixed beforehand and are not selected from final outcomes. This allows the training run to start before the expensive population-wide evaluation; it does not change the baseline policy.

## 3. Checks completed before training

The [implementation check receipt](reproduction/implementation_checks.json) records actual CUDA execution on the RTX 5060 Ti.

- A small-model check compared chunked loss/backpropagation against an unchunked calculation, including tied embeddings and the BF16 parameter-casting context. The loss difference was approximately `5.60e-6`; relative gradient L2 difference was approximately `0.00259`.
- Batched accumulation was compared against individual responses with different lengths: relative gradient L2 difference was approximately `0.00295`; the batched loss differed from the unchunked reference by approximately `1.36e-7`. These are floating-point comparisons, not bitwise identity claims.
- Full-model memory was checked with four responses of 5,120 tokens each and full FP32 AdamW moments resident. Two backward passes and an intervening optimizer step completed; peak allocation was approximately 9.43 GiB.

These checks establish implementation feasibility only. Synthetic random-token capacity probes do not reproduce the research failure and are not included in learning results.

## 4. Audit of the authors' released training log

The source CSV's SHA-256 is `af1033f772ba3e353856d9d65ade0e4fc453443952d87dde078f3f083f0ce4ad`. The [audit](reproduction/released_log_audit.json) preserves its raw per-prompt rows and exposure-weighted summaries.

There are 891 distinct parent IDs in the released unlearnable decomposition examples; 890 survive the current source dataset's prompt filter, and 853 appear in the source log. The matched, observed parent IDs have approximately 3.02% reward in steps 1–20, 7.14% in steps 101–120, and 7.69% in steps 127–146. This is consistent with persistent low reward plus some improvement in the source run. It is **a reanalysis of the authors' log**, not a locally trained reproduction or proof that the examples cannot be learned.

The full source log covers 8,058 distinct training IDs, a subset of the current 8,138 eligible prompts. The absent 80 IDs are a population-provenance limitation, not grounds for silently dropping those prompts from the local run.

**ID integrity finding:** the released learnable decomposition examples use parent IDs 200–299 that do not identify their original dataset rows. For example, their coin-jar problem matches original row 7157, whereas original row 200 is a quadratic graph problem. Their IDs must not be used directly to construct a learnable control curve. The missing final partition exports also prevent treating the unlearnable augmentation IDs as independently verified three-seed labels.

Duplicate `(step, prompt_id)` rows are retained as logged sampling groups. Whether any are duplicated logging events rather than fresh draws is not established; they cannot support a naive independent-response confidence interval.

## 5. Interpretation gates

| Observation | Allowed conclusion |
|---|---|
| Implementation runs and fits memory | Reproduction is technically feasible |
| Source log contains low-reward examples with positives | The released evidence contains the reported pattern, subject to provenance limitations |
| Local training produces validation gains and persistent low-success, positive-exposed examples | Single-seed evidence of selective learning failure |
| Same examples remain low in three independent runs and an additional independent evaluation | Stronger reproduction of the paper's operational cohort definition |
| Correct-trace SFT or a standard RL change repairs that cohort | Baseline failure reproduced; stronger resistance-to-repair claim unsupported in this setup |
| Local training fails globally or does not reach validation saturation | Inconclusive reproduction; do not invent a representation barrier |

The paper's operational label is not an impossibility theorem. A successful reproduction would motivate testing ordinary repairs on the same cohort before developing new theory or a new method.

## 6. Execution and retained artifacts

```bash
bash scripts/run.sh benchmark --run-dir runs/benchmark
bash scripts/run.sh run --seed 1 --run-dir runs/reproduction --max-hours 240
```

Use the [repository setup and restore instructions](../../README.md) before these commands. New results and live logs go under `runs/`; the files linked below are the archived original run, not a live process on the destination machine. Persistent update buffers now stay inside the selected run directory, so a restart can reuse the rollouts. Temporary generation files live under that run's `scratch/` directory unless overridden.

The first full-data seed was launched on 2026-10-15. The [archived status](reproduction/status.json), [original driver receipt](reproduction/driver_process.json), and [frozen protocol](reproduction/protocol.json) identify that execution. It **failed at the start of step 2 optimization**, when Torch reported `RuntimeError: No CUDA GPUs are available`. No step 2 optimizer update completed. The step 1 model and full optimizer state are preserved in the checksummed Git LFS archive, together with the pending step 2 rollout buffer. Original PIDs and machine paths in logs are historical provenance.

The completed step-0 greedy validation scored **125/500 (25.0%)**, with 83/500 responses reaching the 5,120-token cap. The first outer step completed 16 AdamW updates. The three-seed reproduction and independent final evaluations remain incomplete. Seeds 42 and 37 have not been run.

### Measured runtime estimate, 2026-10-15 23:52

The first outer GRPO step completed, including 16 optimizer updates and a resumable checkpoint. Sampling took approximately 55 minutes and optimization approximately 52 minutes: about 1 hour 47 minutes in total. Subsequent steps currently require about two sampling batches rather than the first step's three, suggesting roughly 80–105 minutes per outer step if generation lengths and throughput remain similar. Extrapolation from this early measurement gives approximately **7–9 days for 120 training steps**, plus roughly **12–24 hours for independent evaluations and cohort confirmation**: about **8–10 days for the first seed overall**. This is a provisional estimate from one completed outer step; response lengths and dynamic-filtering yield can change during training.

**The 96-hour setting is a termination limit, not a completion estimate.** The original deadline would have been approximately **2026-10-19 21:35 Asia/Bangkok**, but the run failed earlier. The portable continuation instructions use an explicit 240-hour budget; the learning recipe is unchanged.

**Current scientific conclusion:** the persistent failure has not yet been reproduced locally. The next result must come from the full-data run and independent evaluations described above.
