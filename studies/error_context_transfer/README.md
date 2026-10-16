# Research direction screen: when synthetic error supervision transfers

Date: 16 October 2026. Status: **retired in its current formulation after prior-art verification; no new method validated**.

## Decision

The candidate question examined in this screen was:

> When an identical wrong claim appears in an edited correct response and in a naturally generated wrong response, what determines whether a verifier learns correctness rather than an artifact of the surrounding response? Can that distinction predict transfer to a different generator before expensive policy training?

**Decision after the follow-up prior-art check: reject this formulation as the next research proposal.** The broad synthetic-to-natural transfer failure and the proposed consistency-over-correctness explanation already have direct precedents. Our four-cell pilot has a different construction, but it establishes neither a distinct mechanism nor a useful intervention beyond that literature. Do not allocate further experiment time to the current proposal. No conference acceptance likelihood is assigned.

### Follow-up novelty audit, 16 October 2026

| Primary source and reading scope | Overlap with our candidate | Boundary of the comparison |
|---|---|---|
| [Reward Models Identify Consistency, Not Causality](https://arxiv.org/html/2502.14619), February 2025; §§4–6 | Input interventions across reward models and math tasks motivate a consistency-bias explanation; ranking and Best-of-N are evaluated. | Different interventions from our answer-box swaps. The broad mechanism is already studied; their empirical interpretation is not a universal causal proof. |
| [Synthetic Error Injection Fails to Elicit Self-Correction In Language Models](https://arxiv.org/html/2512.02389), December 2025; methods, §3.5 and §4 | Directly examines synthetic-to-natural error transfer despite high measured error coverage. | Self-correction rather than reward ranking; this difference alone does not supply a new mechanism. |
| [Unsupervised Multimodal Reward Modeling](https://arxiv.org/html/2610.12707), October 2026; §3.4, Appendices F/G | Appendix F already contrasts strong planted-decoy discrimination with unchanged natural-caption selection; §3.4 tests verifier-labeled natural candidates. | Multimodal selection rather than our small text probe. Generic natural-error bootstrapping is already a tested remedy. |

The last two sources were already in the earlier report. Keeping the candidate alive on the basis of an unverified exact-design difference was too optimistic. The newly checked 2025 consistency paper makes the mechanism overlap explicit. This audit uses paper text; it does not claim access to OpenReview discussion, reviews, rebuttals or metareviews.

The recorded 94.4% / 66.7% / 6.2% results remain exploratory measurements. They are compatible with a known consistency shortcut, and the edits confound consistency with final-answer correctness. They neither reproduce every prior-paper setting nor establish a new causal explanation. The pair-accuracy metrics here and the multimodal paper's selected-caption metric are different and must not be compared as a numerical replication.

## 1. Evidence and intuition

This is a targeted primary-source screen, not a systematic review. Reading scope is stated below. No new OpenReview reviews, rebuttals or metareviews were obtained: attempted forum access returned a browser challenge. One delegated read-only literature screen supplied leads; the main sources below were subsequently opened by the parent researcher. Abstract-level checks are not represented as full-paper reviews.

| Paper; reading scope | Intuition | Finding that matters | Consequence for this study |
|---|---|---|---|
| [Unsupervised Multimodal Reward Modeling](https://arxiv.org/html/2610.12707), October 2026; method, §§3.1–3.4, Appendices D/F | Recognizing an artificial corruption may teach a different distinction from choosing between real model outputs. | Planted-decoy discrimination does not reliably improve natural candidate selection. Several training variations fail to move benchmark results materially; verifier-labeled policy candidates improve deployment while benchmark scores decline. | A direct failure of synthetic supervision as a deployment proxy. Its own bootstrap is a mandatory successful baseline, not our proposed contribution. |
| [Synthetic Error Injection Fails to Elicit Self-Correction](https://arxiv.org/html/2512.02389), December 2025; methods and analysis, including §3.5 | Covering the possible mistakes is insufficient if mistakes arise in different contexts. | Synthetic errors cover over 95% of the measured natural error support, yet correction does not transfer reliably. | Neither “synthetic errors fail” nor “coverage is insufficient” is new. A contribution must identify and predict a more specific mechanism. |
| [Reward Learning from Best-of-N Preference Data](https://arxiv.org/abs/2605.30619), May 2026; abstract-level collision check | Stronger chosen/rejected separation can make useful comparisons rarer. | Analyzes targets, the margin–connectivity tradeoff, and shaping the sampling distribution around relevant test comparisons. | Generic coverage, hardness, covariance or sampler-design claims are already occupied. |
| [Towards Optimality in Active Preference Learning](https://arxiv.org/abs/2602.01581), v2 October 2026; abstract-level collision check | Select comparisons according to the decision they help resolve. | Gives instance-dependent label complexity and a practical greedy design. | Random-versus-our-acquisition would be an inadequate comparison. |
| [Optimal Design with Biased LLM Judges](https://arxiv.org/abs/2609.38860), September 2026; abstract-level collision check | An apparently informative comparison can mostly reveal the judge's bias. | Accounts for residual judge bias during acquisition and studies nuisance representation error. | Judge-aware acquisition and generic nuisance adjustment are not available novelty claims. |
| [The Value of Being Wrong / SMILE](https://arxiv.org/abs/2610.16977), October 2026; abstract-level collision check | Past mistakes from the same model are useful demonstrations. | Mines model-specific errors for prediction-conditioned visual in-context learning. | “Use the model's own errors” is already a current remedy. |

### Why the experiment's differences do not establish novelty

The proposed interaction between an erroneous claim and its surrounding response fits the existing consistency-bias explanation. A different factorial construction can help diagnose a model, but is insufficient as the contribution of a new paper. No independently useful transfer predictor, unresolved deployment failure surviving existing repairs, or new theoretical result has been established. The current candidate fails the novelty gate.

## 2. Initial experiment actually performed

### Data and scope

We reused the repository's existing 1,024-prompt, eight-rollout-per-prompt bank. No new rollout generation or backbone training occurred. There were 201 prompts with an eligible naturally correct and naturally incorrect response: non-truncated, at most 650 response tokens, agreement between both stored graders, and an unambiguous integer in the final boxed answer agreeing with the stored extraction. We shuffled once with seed 20261016 and retained 160 prompts: 106 train and 54 test. All four variants of a prompt stay in one split; prompt IDs were checked for uniqueness.

For each selected prompt, let A be the correct final integer and B the wrong integer. Construct:

| Cell | Surrounding reasoning | Final integer | Outcome label |
|---|---|---|---|
| Natural correct | From a naturally correct rollout | A | Correct |
| Injected wrong | Same reasoning as natural correct | B, substituted in the last box only | Incorrect |
| Repaired outcome | From a naturally wrong rollout | A, substituted in the last box only | Correct |
| Natural wrong | From that same wrong rollout | B | Incorrect |

**These labels concern final-answer correctness only.** An edited final answer may contradict its reasoning. Calling the repaired outcome a correct solution or a preferable complete response would be wrong. Existing stored correctness labels were reused rather than independently re-annotated.

We extracted mean response-token features at a fixed intermediate layer of the frozen, pretrained Qwen2.5-0.5B. This is a text-side feasibility probe, not a reproduction of the multimodal paper or its trained reward model. Small pairwise linear heads, a 32-unit MLP, a length-only control and an answer-only canonicalization control were fitted. Three fit seeds reuse the same prompt split; they are not three independent datasets. Hyperparameters were fixed rather than selected on test results.

### Results

Accuracy below is averaged over the three fit seeds; ties receive half credit. The training regimes use the same train prompts, but the mixed regime has three pair types per prompt and therefore more supervision. It is a generous repair control, not a matched-budget method comparison.

| Training and scoring | Injected pairs | Repaired-outcome pairs | Natural correct/wrong pairs |
|---|---:|---:|---:|
| Linear head, injected pairs | 94.4% | 6.2% | 66.7% |
| Linear head, natural pairs | 93.2% | 22.8% | 71.0% |
| Linear head, repaired-outcome pairs | 3.7% | 94.4% | 29.6% |
| Linear head, all three pair types | 69.8% | 42.0% | 63.0% |
| MLP, injected pairs | 94.4% | 10.8% | 61.7% |
| MLP, natural pairs | 89.5% | 18.5% | 65.4% |
| MLP, all three pair types | 66.7% | 31.5% | 59.3% |
| Answer-only canonicalization + linear head | 49.4% | 49.4% | 49.4% |
| Length-only control, trained on natural pairs | 50.0% | 50.0% | 53.7% |

The natural-trained versus injection-trained linear head differs by **+4.3 percentage points** on natural pairs. A descriptive paired bootstrap over the 54 test prompts gives **[-9.9, +18.5] points**. This does not establish a reliable advantage for natural training. The estimand is conditional on these trained heads and this eligible prompt subset; the interval does not include uncertainty from redoing the training-data selection.

### Interpretation and limits

The striking reversal across edited pair types is consistent with learning whether the final answer agrees with the surrounding reasoning. A wrong natural reasoning chain and its wrong conclusion can agree with each other. Editing the conclusion alone breaks that agreement. Training on repaired outcomes can therefore teach the opposite shortcut.

This is a **hypothesis**, not a causal result: we did not independently label reasoning consistency. Canonicalizing to question plus final answer removes the context difference by construction, but natural-pair accuracy falls to 49.4%. A larger head also fails to improve the measured natural-pair result in this run. Neither observation proves that normalization or ordinary supervised learning cannot solve a properly specified task.

The honest outcome is that the quartet design runs cheaply and reveals a label-target confound that must be controlled. It does **not** establish a new multimodal failure, a new learning algorithm, or that a repair survives the strongest baselines. Only one backbone, layer, split and restricted integer-answer subset were studied. Best-of-N selection and policy improvement were not measured. This pilot is not evidence for resuming the old RLVR campaign.

GPU: RTX 5060 Ti 16 GB. Feature extraction took roughly nine seconds; the separately added canonicalization pass took roughly three seconds. These measured times exclude model loading, data filtering and earlier environment work. They are not a runtime promise for a VLM or a complete study.

## 3. Mathematical starting point

These are elementary diagnostic identities, **not claimed novel theorems**.

Let h(c,a) be a frozen representation of a response with surrounding context c and final claim a. For the same correct/wrong claims A and B, define

$$
d(c)=h(c,A)-h(c,B),
\qquad
I=d(c_{\mathrm{correct}})-d(c_{\mathrm{wrong}}).
$$

A linear reward r_w(c,a)=w^T h(c,a) changes its preference margin across the two contexts by

$$
m_w(c_{\mathrm{correct}})-m_w(c_{\mathrm{wrong}})=w^T I.
$$

Thus fixing the incorrect claim does not fix its training signal. The model can respond to how the claim relates to the rest of the answer. Measuring I alone does not identify its semantic cause; contradiction, length, difficulty and token position can all contribute.

A simple transport model would posit

$$
d_{\mathrm{natural}}=T d_{\mathrm{synthetic}}+\epsilon.
$$

For invertible T, transporting a source head as w_n=T^{-T}w_s gives

$$
w_n^T d_{\mathrm{natural}}
=w_s^T d_{\mathrm{synthetic}}+w_n^T\epsilon.
$$

Consequently a positive source margin exceeding ||w_n||_2 ||epsilon||_2 stays positive. This algebra is standard. **Assuming a small residual would assume away the proposed problem.** A useful theory would need observable conditions for such transport, sample requirements for estimating only the decision-relevant component, and a relation to held-out within-pool ranking regret. No new rate or theorem is asserted here.

The main theoretical fork is whether matched contrasts share a stable decision direction at all. If they do, ordinary calibration or transport may suffice. If they do not, a global invariant head is misspecified. A publishable contribution must characterize a measured regime and show why an existing nonlinear head, richer features, or natural-error data does not already resolve it.

## 4. Archived contribution plan — not recommended after the novelty audit

The following records the earlier conditional plan; it is not an active proposal.

Working study title: **When Does Synthetic Error Supervision Transfer to Natural Mistakes?**

The intended contribution would be a controlled mechanism study with a deployment-predictive diagnostic, followed by a data intervention only if needed:

1. Separate error-category coverage, response provenance, semantic consistency and feature limitations using paired interventions.
2. Predict natural candidate-selection performance on held-out prompts and generators using training-side measurements, beyond source accuracy and existing design statistics.
3. Test whether selecting or constructing transportable contrasts improves real selection at fixed target-label and feature-extraction budgets.

The method is deliberately unspecified until the mechanism is established. A new weight, regularizer or sampler is not itself the contribution. If a simple baseline repairs the measured failure, close the methods proposal and assess whether the explanatory result independently merits a paper.

## 5. Other candidates closed during this screen

| Candidate | Existing fix or decisive concern | Decision |
|---|---|---|
| Finite-K random-order protein DPO | [ProtAlign](https://arxiv.org/html/2603.06748) averages order-conditioned probabilities, then takes log ratios. [LLaDA 1.5](https://arxiv.org/html/2505.19223) already studies noisy MC preference losses and variance reduction; [earlier protein DPO](https://arxiv.org/html/2410.19471) explicitly discusses multiple-order averaging. | Reject generic MC correction as the new contribution. |
| Quantization safety | [QDPO](https://arxiv.org/abs/2407.03051), [ExSQF](https://proceedings.mlr.press/v306/yang26bg.html) and [Q-realign](https://arxiv.org/abs/2601.08089) provide direct remedies. KV-specific [diagnosis and mitigation](https://arxiv.org/abs/2606.09864), [AnchorKV](https://arxiv.org/abs/2606.17872) and runtime certification also occupy the obvious extension. | Reject generic safety-preserving quantization. Some sources were checked at abstract level only. |
| Reward-bias subtraction under selection | [Reward Bias Substitution](https://arxiv.org/html/2605.27996) already establishes audit/deployment mismatch. [FiMi-RM](https://aclanthology.org/2026.acl-long.133/) learns nonlinear corrections; causal and robust reward methods provide stronger comparisons. | No residual beyond those remedies established. |
| Cross-dimension rubric compensation | [ProRubric](https://arxiv.org/html/2609.38847) documents residual damage, but [graphical aggregation](https://arxiv.org/html/2606.03361) and [robust rubric gates](https://arxiv.org/html/2605.30244) directly address dependencies and essential failures. ProRubric also reports a useful SFT baseline. | Reject another generic gate or dependency graph. |
| Claim splitting to game factual reward | [Claim-Level Rubric Rewards](https://arxiv.org/html/2607.05150) already caps residual claim credit; semantic matching and deduplication are strong ordinary repairs. | No evidence yet that a consequential failure survives them. |

### Protein numerical check actually run

Using pinned original ProteinMPNN code, its base checkpoint and a released MoMPNN alignment checkpoint, we scored two author example backbones (68 and 106 residues), five sequences per backbone and 256 shared decoding orders. The order-weight effective sample size ranged from approximately 11 to 79 out of 256. Estimated aligned/reference log-ratio standard deviations were approximately 0.48–0.84 at K=1 and 0.08–0.38 at K=64. The run took 12.7 seconds after loading.

This establishes numerical sensitivity, not biological harm. Increasing K reduced variation. K=256 was an internal comparison estimate, not ground truth. We did not run a design-quality or property assay, a training-gradient experiment, or a complete baseline comparison. This is a closed exploratory check, not a second proposed study.

## 6. Archived decision gates — stage 0 failed for the current formulation

The follow-up audit failed the novelty requirement below. Later stages are retained as a record of the original plan, not as authorized or recommended experiments.

| Stage | Concrete work | Advance only if |
|---|---|---|
| 0. Novelty and assets | Read the closest synthetic-error, causal-reward and active-design papers fully; verify an available natural candidate bank with grounded labels and reuse rights. No code/cache/candidate bank was found linked from the October multimodal source in this screen. | The narrow mechanism is distinguishable from prior context-mismatch studies and the data are accessible. |
| 1. Valid matched examples | Build 200–400 matched grounded examples. Change the same fact bidirectionally while checking all dependent claims. Label correctness, consistency and relevance separately. Preserve original natural outputs for the primary outcome. | The claimed gap remains when edits are semantically valid, rather than contradictory answer-box substitutions. |
| 2. Strong cheap repairs | Compare error-type matching, formatting normalization, length control, likelihood-based hardness, importance weighting, whitening, a nonlinear head, training-free judgment, natural-only data and mixed data. Include successful policy-error bootstrap and current active-design baselines with equal label budgets. | A material natural-selection failure remains after comparable tuning; an existing repair does not match the proposed intervention. |
| 3. Transfer prediction | Split by prompt and generator. Predict selection accuracy/regret on untouched natural pools, including a generator unseen during diagnostic construction. Report paired prompt intervals and label budgets. | The diagnostic predicts deployment beyond source accuracy, error coverage and sampler geometry controls. |
| 4. Theory | Derive assumptions from the measured interaction; establish identifiability or an obstruction, and prove a decision-relevant guarantee with finite calibration data. | The theorem does not assume invariant error directions or reliable target labels without accounting for their cost. |
| 5. Small policy test | Only after the preceding gates, perform a short policy update or reranking experiment on a second task/model. | Improved natural outcomes persist without sacrificing useful detail or merely changing answer format. |

**Archived budget rule, inactive after rejection:** stage 0 uses no new training. Profile 20 VLM examples before stage 1. Allow at most two GPU-hours for an initial feature-only screen, including extraction; stop at the cap if the required data do not fit it. This is a spending limit, not an estimated completion time. No full RL training is justified by the current results.

## 7. Files and reproduction

- [quartet_probe.py](quartet_probe.py): canonical text feasibility experiment. Uses the existing rollout bank and a pinned model revision; downloads only that model if it is not cached. It writes aggregate results and prompt IDs, not feature caches.
- [results.json](results.json): one canonical result file containing the text probe and the closed protein numerical check, with provenance and limitations.

For the text probe, use an environment with working CUDA Torch, Transformers and NumPy. The recorded run used Python 3.11, Torch 2.13.0+cu130, Transformers 5.14.1 and an RTX 5060 Ti. It does not use or modify the cloud RLVR environment.

```bash
python studies/error_context_transfer/quartet_probe.py --output /tmp/quartet_results.json
```

The default input is the rollout bank already in this repository. The default model revision is fixed in the script. The script fails on a different input hash rather than silently claiming reproduction. Different numerical libraries/hardware can change fit-level details. No human-quality or multimodal validation is implied by a successful run.

## Final self-review

The problem is real, but the current explanation overlaps directly with prior work. Our cheap pilot exposes a confound and remains reproducible; it does not establish the distinctive contribution required to pursue this as the next paper. **Retire the current formulation and retain its results as an exploratory negative screen.** Reopening would require a separately justified unresolved question and evidence beyond the known synthetic-transfer and consistency-bias findings. No further experiments are recommended for this formulation.
