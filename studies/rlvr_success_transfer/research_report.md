# Successful rollouts without reliable learning: a research direction and its gates

**Date:** 15 October 2026. **Decision after GPU testing:** no new-method recommendation. The RTX 5060 Ti completed five full-parameter pilot arms; ordinary SFT learned the sampled cases. The published persistent-failure cohort has not been reproduced. See [the empirical results and research decision](#11-completed-rtx-5060-ti-experiment).

## 1. The research question

> When a hard reasoning problem already yields correct rollouts, why does reinforcing those rollouts sometimes fail to produce reliable improvement on that problem and its valid variants? Can we distinguish an update-estimation problem, an optimization problem, and a failure of transferable computation—and demonstrate which intervention repairs each?

The useful outcome is a **causal explanation and a demonstrated repair of an existing learning failure**. A better difficulty score, a faster example router, or another cosine-based reward would not meet the intended contribution.

This is different from our previous restricted-policy/cardinal-feedback project. Here the correctness verifier is available, the model actually samples successful responses, and the target is learning and transfer in an ordinary reasoning model. No prompt-blind policy restriction is introduced to manufacture a failure.

**Intuition:** a student occasionally getting the right answer does not tell us whether extra practice, a better teaching method, or learning a missing concept is needed. Current measurements do not reliably separate those possibilities. The research must make that distinction experimentally useful.

## 2. Evidence that the problem is real, including contrary evidence

### 2.1 Original observation: do not assume the diagnosis is established

[Chen, He and Zhao, *The Unlearnability Phenomenon in RLVR for Language Models*, ICML 2026](https://proceedings.mlr.press/v306/chen26aq.html), [method and appendices](https://arxiv.org/html/2605.16787), studies hard prompts receiving positive rewards yet improving poorly. Sections 4–5 and Appendices A.2–A.5 test positive replay, larger rollout groups, relaxed clipping, removal of KL regularization, correct-response SFT, augmentation and curriculum variants. These do not close the reported gap. Section 5.4 compares gradient similarity after mid-training; it does not demonstrate an RL intervention repairing the same fixed hard cohort. Low gradient similarity is evidence of association, not proof of a representation barrier.

**Intuition:** obtaining a correct example and fitting more such examples are insufficient in their tested settings. Their explanation still needs a causal test.

### 2.2 A recent reanalysis changes the target substantially

[Chakma et al., *Unlearnable, or Unmeasured?*, September 2026 preprint](https://arxiv.org/html/2609.40115), §§3.2–3.6, reports slow learning and reduced gradient separation after matching rollout counts. Different data and generation caps prevent treating its magnitudes as a matched replication. These measurement corrections are prior work.

**Our inference from §§3.2/A.1:** changing membership in the logged training population can also affect a fitted slope. An independently evaluated fixed-panel contrast remains necessary. This is a validity check, not a demonstrated error or proposed contribution.

**Intuition:** establish the remaining learning effect with matched measurements before explaining it.

### 2.3 Practical consequence and scope

The intended use case is deciding whether a reasoning-training failure needs better updates or additional representation learning, then making the affected skills improve. The practical claim must be a gain on independently held-out problem variants or families, with aggregate capabilities retained. Increasing training accuracy alone would establish memorization, not the intended repair.

The evidence is currently strongest for the studied small reasoning models. It does not establish a universal failure of RLVR or a problem affecting every frontier model. Nor does it establish that 20B tokens of mid-training are necessary. Both would be overclaims.

## 3. Prior-art check before proposing a method

The following are direct competitors or foundations. Their existence rules out several attractive but weak proposal shapes.

| Primary source and inspected scope | Intuition explanation | What is already occupied | What still needs to be established for this question |
| --- | --- | --- | --- |
| [Unlearnability, 2605.16787](https://arxiv.org/html/2605.16787), §§3–5, App. A | Success can occur without effective learning. | Reporting the phenomenon, gradient-outlier association and the listed ordinary interventions. | A causal repair on a fixed cohort with transfer beyond the trained wording. |
| [Unlearnable or Unmeasured, 2609.40115](https://arxiv.org/html/2609.40115), §§3.2–3.6, Apps. C–E | Noisy success estimates can alter the population being explained. | Difficulty-set reliability and matching gradient-estimator sample counts. | Explanation and repair of the residual slow-learning effect. |
| [Barzilai, Wolf and Basri, 2602.07992](https://arxiv.org/html/2602.07992), task-selection model and task-advantage results | A correct intermediate skill must improve eventual success to be reinforced. | Outcome feedback can favor shortcuts or fail to teach compositions; a structural advantage condition can permit learning. | Transfer and feature changes under shared neural parameters, beyond the fixed task library and orthogonal positional embedding model. Merely renaming its advantage ratio is excluded. |
| [GAR, 2609.03342](https://arxiv.org/html/2609.03342v1), §3 and experiments | An expert trace supplies a direction for distinguishing successful rollouts. | Expert-anchor gradient rewards and a decomposition of their signal. | Whether this already repairs the residual cohort under information-matched controls. It is a mandatory baseline, not an innovation we can claim. |
| [Supervised Reinforcement Learning (SRL), ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7ab7fc1278add78fd6eae2da7a14c79b-Abstract-Conference.html), official abstract checked in this continuation | Reward the important intermediate actions while allowing the model to generate its own reasoning. | Expert-action supervision and SRL followed by ordinary RLVR. | Whether this established intervention repairs the fixed cohort. The abstract establishes relevance; it is not evidence of a matched test of the present hypothesis. |
| [LearnAlign, current v4](https://arxiv.org/html/2506.11480v4), scoring and selection method | Select examples with useful success rates and aligned updates. | Gradient/difficulty-based training-data selection. | Repair of the affected skills rather than discarding them to improve average scores. |
| [CROPI, current v2](https://arxiv.org/html/2510.26491), §3 and limitations | Estimate how one example's update affects another query. | First-order influence and off-policy gradient-based selection. | A causal, uncertainty-aware distinction between unavailable useful directions and poorly estimated or poorly executed ones. Its stated first-order SGD scope is not a general neural learnability theorem. |
| [MASPO, 2602.17550](https://arxiv.org/html/2602.17550), §3 and method | Rebalance how existing positive and negative signals enter optimization. | Gradient utilization, probability-dependent treatment and asymmetric reliability corrections. | Whether these ordinary optimizer-side changes close this particular residual failure. |
| [PSRA, 2609.33085](https://arxiv.org/html/2609.33085), §§3–4 | Elicit another existing reasoning strategy, then preserve and transfer its successful behavior. | Strategy-guided exploration, adaptive rollout allocation and guided-to-unguided transfer. | Whether a residual persists after strategy rescue. Guiding another reasoning route and distilling it is already a method, not an available new proposal. |
| [Kakade, natural policy gradient](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html); [TRPO](https://proceedings.mlr.press/v37/schulman15) | Judge an update by how much the policy changes, not just its parameter length. | Fisher geometry and trust-region optimization. | An RLVR mechanism result and effective repair. Replacing a cosine with a Fisher metric is not sufficient novelty. |
| [Bhattacharjee, Campbell and Shome, 2609.33499](https://arxiv.org/html/2609.33499) | Even an exact direction may not reveal a useful finite step. | Statistical and computational difficulty of turning a natural-gradient direction into a useful update. | The proposed study must measure finite updates, not declare a representation problem from a local direction alone. |

**Search-bounded conclusion:** I did not find a verified paper that resolves this exact residual using the proposed matched causal experiment. That is not a claim that gradient-based learning diagnostics or reasoning repair are new fields. Existing GAR, SRL, process supervision, natural-gradient methods and mid-training could still win the practical test.

No new OpenReview discussion or rebuttal was accessed for these papers. This audit uses primary papers and the original authors' public repository; it does not manufacture review evidence.

## 4. The simple-solution check

| Possible repair | Present evidence | Required decision |
| --- | --- | --- |
| More successful rollouts, replay, weaker clipping/KL, ordinary correct-trace SFT, generic augmentation | Tested in the original source; unsuccessful there. | Retest the strongest relevant versions under the matched protocol. Published negative results are scoped, not universal impossibility proofs. |
| More reliable evaluation and matched successful-rollout counts | Recent reanalysis materially changes the interpretation. | Apply these controls first. If the residual disappears, stop this project. |
| Longer training | Can distinguish slow learning from non-learning, but can also memorize. | Compare held-out transfer at equal tokens and report the whole curve. If it closes the gap economically, a new repair method is unnecessary. |
| Natural gradient, Adam/K-FAC-style preconditioning, PCGrad/CAGrad | Established methods; not ruled out by clipping/KL ablations. | Mandatory baseline stage. If one closes the gap and the explanation adds nothing beyond known optimization effects, abandon a methods-paper claim. |
| GAR, SRL, process supervision or student-generated correct-trace training | Closely relevant existing approaches. | Match access to expert traces/process labels and total work. If one solves it, do not repackage its mechanism. |
| Mid-training or a stronger base model | Plausible remedy; the gradient comparison is not a causal repair result. | Directly compare subsequent RL and transfer on the same cohort. A successful existing recipe changes this into a mechanism study, or ends the project. |
| Drop the hard prompts | Can improve average training efficiency. | Does not answer the question of making the affected skills learn. It is a useful efficiency baseline, not a repair. |

**Gate status:** published evidence motivates a residual failure beyond several ordinary fixes. Its magnitude still needs independent fixed-panel confirmation, and the strongest remaining repairs have **not** been ruled out locally. Consequently, this is an investigation to run, not a method to commit a full project to today.

## 5. Theoretical foundation: distinguish observations from explanations

The mathematics below sets up an identifiable target. These identities and local optimization facts are foundations, not claimed new theorems.

### 5.1 Correctness probability and the actual learning signal

For a prompt $x$, let $R_x(y)\in\{0,1\}$ be a fixed verifier and

$$
p_x(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}R_x(y),
\qquad s_x(y)=\nabla_\theta\log\pi_\theta(y\mid x).
$$

Assume differentiability, parameter-independent support, interchange of differentiation and expectation, and finite second score moments. Use the full sequence score, including the specified stopping convention. Then

$$
g_x:=\nabla p_x
=\mathbb E[R_xs_x]
=p_x(1-p_x)(m_{1,x}-m_{0,x}),
\quad m_{r,x}=\mathbb E[s_x\mid R_x=r].
$$

This follows from $\mathbb E[s_x]=0$. Correct rollouts can exist while their mean score is weakly separated from the failure mean. Their mere count is not a complete description of the update signal. This is a hypothesis to measure, not an explanation already established for the source cohorts.

For binary reward, the exact within-prompt score covariance decomposes as

$$
F_x=\mathbb E[s_xs_x^\top]
=p_x\Sigma_{1,x}+(1-p_x)\Sigma_{0,x}
+p_x(1-p_x)\Delta m_x\Delta m_x^\top,
\quad\Delta m_x=m_{1,x}-m_{0,x}.
$$

With $0<p_x<1$, define

$$
\chi_x=\frac{g_x^\top F_x^\dagger g_x}{p_x(1-p_x)}\in[0,1].
$$

The bound follows by projecting the centered verifier outcome onto the linear span of score coordinates in $L^2(\pi_\theta)$. This is a classical projection/information calculation. A small estimated value is not yet evidence of a global representation barrier.

**GRPO boundary:** finite-group standardization, response-length averaging, clipping, replay and Adam change the actual update. The identities describe $\nabla p_x$, not an assertion that practical GRPO is an unbiased estimator of that gradient. The study must log both the intended derivative and the implemented update.

### 5.2 Transfer must be measured on untrained variants

Let $\mathcal V_j$ be a separately held-out distribution of valid variants of a problem family. Define

$$
q_j(\theta)=\mathbb E_{x\sim\mathcal V_j}p_x(\theta),
\qquad h_j=\nabla q_j(\theta).
$$

For a candidate training update $d$, its first-order transfer is $h_j^\top d$. This is different from cosine similarity between two successful training traces. Its realized finite-step counterpart is

$$
T_j(d)=q_j(\theta+d)-q_j(\theta).
$$

Use independently sampled evaluation responses to estimate this quantity. Do not construct the evaluation family by selecting whichever variants improve after seeing the update.

### 5.3 The classical local ceiling and its limitation

Let $F$ be the sequence-policy Fisher under a fixed prompt mixture, restricted to an identifiable tangent space on which it is positive definite. A quadratic approximation to average policy KL is $d^\top Fd/2$. The best common linearized improvement is

$$
C_\theta(\epsilon)
=\max_{d:\,d^\top Fd/2\le\epsilon}\min_{j\le m}h_j^\top d
=\sqrt{2\epsilon}\min_{\lambda\in\Delta_m}
\left\|\sum_{j=1}^m\lambda_jh_j\right\|_{F^{-1}}.
$$

The equality follows from minimax duality and the dual norm of an ellipsoid. It is a Fisher-metric form of established [multicriteria steepest descent](https://doi.org/10.1007/s001860000043), not a new algorithmic contribution. It separates a poor executed direction from the absence of a shared favorable tangent direction under the chosen KL cost.

For twice differentiable family success functions with a uniform remainder bound $|q_j(\theta+d)-q_j(\theta)-h_j^\top d|\le L\|d\|_F^2/2$ in that neighborhood, optimizing over the quadratic ball changes this linear prediction by at most $L\epsilon$. This is a local Taylor statement, not a theorem about the true finite-KL ball or unrestricted neural training.

In particular, $C_\theta=0$ does not prove that feature learning is necessary. Larger steps, different tangent spaces, or different weighting of families may help. Estimating the ceiling in a small update subspace gives a constructive lower bound for the full space, not an upper bound proving no useful direction exists elsewhere.

### 5.4 What would constitute substantial new theory

The useful theoretical target is a **finite-sample, finite-update account of transfer after successful sampling**, coupled to a measured intervention. A serious result should contain:

1. An estimation or impossibility theorem accounting for rare successes, high-dimensional scores and adaptive selection of updates. Plugging a pseudoinverse into the same data used to construct a direction is insufficient.
2. A precise transfer result for shared, nonorthogonal skill representations; it must go beyond a fixed independent task-selection model. It should predict when increasing successful samples helps and when a change of learned features is needed within a specified computational regime.
3. A finite-step test or bound connecting the prediction to realized held-out gains under measured KL. A local metric alone is insufficient.
4. A constructive repair whose advantage survives natural-gradient, process-supervision and expert-trace baselines at matched information and work.

These are **research targets, not results already proved**. If only the elementary identities above remain, there is no theoretical-paper contribution.

## 6. Completed analytical preflight

[One script](experiment.py) produces [one result file](results.json). It checks failure modes of prospective diagnostics; it does not simulate or reproduce the reported NLP failure.

| Check | Result | Consequence |
| --- | --- | --- |
| Invertible parameter-coordinate change | Raw cosine changes from 0.3846 to -0.5039; the Fisher-dual inner product remains 0.150785. | Raw cosine is not an intrinsic representation-barrier certificate. This is a classical geometry fact. |
| Independent Bernoulli prompts with orthogonal parameter directions | Cross-prompt cosine is zero, yet exact ascent raises mean success from 0.0474 to 0.9978. | Low cross-prompt similarity alone does not imply failure to learn. |
| Same-sample empirical Fisher fit, 64 observations, 96 score coordinates, 4 positives, statistically independent outcomes | Estimated $\chi=1$ although population $\chi=0$; held-out linear gain is approximately -0.000079. | A naive natural-gradient diagnostic can report perfect apparent learnability under a null. Simply switching to Fisher geometry does not solve measurement. |
| Tangent max-min duality | Primal/dual discrepancy below $4.7\times10^{-9}$. | The proposed baseline calculation is implemented consistently. It does not establish new theory. |

The interpolation check is exact in this setup: the centered design has rank 63 and spans every centered label vector. The fitted projection explains all in-sample label variance. Independent outcomes make the population gradient zero. This is an overfitting control, not a proposed mechanism for the source results. Ridge regularization, cross-fitting and held-out direction tests are established remedies and must be included.

The coordinate-change control also has a limit: keeping an optimizer fixed in one parameterization gives its Euclidean geometry operational meaning. A coordinate-dependent cosine may still predict that optimizer's behavior. The control rejects interpreting cosine as an intrinsic impossibility certificate; it does not refute every empirical use of cosine.

## 7. The decisive feasibility experiment

### 7.1 Establish a fixed effect before designing a repair

Start with the original Qwen2.5-0.5B/MATH configuration. Freeze generation length, extraction, verifier, optimizer, templates and token budget. Use independent pilot and confirmation samples. Define a continuous slow-learning outcome with uncertainty instead of selecting a cohort from the same final outcomes used to explain it.

Evaluate the same frozen prompt panel at predeclared checkpoints, including prompts with all-correct or all-failed training groups. Estimate each checkpoint's success probability from fresh samples. Keep training-filter exposure as a separate explanatory variable. This prevents a change in the active training population from masquerading as a learning curve.

Use the original three model seeds as a minimum replication anchor; determine additional seeds from measured between-run variance and the smallest practically meaningful gain. Prompts, not individual tokens or rollouts, are the primary resampling units. Account for train/evaluation overlap in MATH variants and avoid selecting favorable checkpoints using the final test.

For the mechanistic subset, obtain matched successful and failed rollout counts and preserve the response strings. If a prompt cannot supply the desired count within a predeclared cap, report the resulting uncertainty and attrition. Do not silently replace it with an easier prompt.

### 7.2 Matched interventions

At identical model snapshots, fork training states and compare:

- Ordinary GRPO/DAPO and longer-training controls.
- Positive replay and student-generated correct-response SFT.
- Expert-trace SFT, GAR and SRL/process supervision, with their extra information recorded; PSRA or an equivalent current strategy-rescue baseline.
- A trust-region natural-gradient approximation and a strong preconditioner, with damping and actual sequence KL measured.
- A gradient-conflict baseline when conflict is actually observed.
- A model linearized at the starting snapshot versus nonlinear training of the same parameter block. A separate frozen-feature control is useful but changes the hypothesis class; it cannot alone establish a feature-learning mechanism. Record approximation error and work rather than assuming these controls cost the same.
- Mid-training initialization followed by the same RL protocol, comparing the same fixed evaluation families rather than freshly relabeling each model's hard set.

Initially use short intervention windows to predict and measure transfer; then run the best standard repairs to convergence. Short windows alone cannot establish persistent failure.

### 7.3 Metrics that determine whether this is useful

Primary: independently measured improvement in verified success on held-out variants/families of the affected skills, at matched generation/training tokens. Secondary: aggregate held-out accuracy, pass@1/pass@k, reasoning validity on a controlled subset, measured policy KL, wall-clock cost and sensitivity to response length.

Create variant families before interventions. Use exact answer-preserving transformations where valid, independently verified parameterized problems, and a separately authored family split. Report these separately: paraphrase consistency is weaker evidence than generalization to new compositions. Do not accept a language model's unverified answer as ground truth.

A concrete provisional success threshold is at least a 5 percentage-point absolute transfer gain over the strongest matched repair baseline, with no more than a 1 point aggregate accuracy loss. These thresholds are proposed practical criteria, not predicted results; revise them before the confirmatory experiment using pilot uncertainty and the intended workload.

The meaningful ablations remove: matched rollout counts; independent confirmation data; Fisher/optimizer correction; transfer evaluation; the feature-learning intervention; and extra process/expert information. A gain that vanishes when information access is matched is not evidence for the proposed mechanism.

## 8. Expected outcomes and stop conditions

There are three legitimate outcomes, and only the third justifies developing a new repair method:

1. **The residual disappears with proper measurement.** Stop; the recent measurement literature already occupies this explanation.
2. **An established repair closes the gap.** Use that method. Continue as a mechanism paper only if a new, consequential causal result remains; do not advertise a renamed optimizer.
3. **A stable transfer failure survives the strong repairs.** Identify the measured bottleneck, derive the corresponding learning condition, and develop a targeted intervention. This is the viable methods-plus-theory branch.

Further stop conditions: gain only on trained wording; no replicated effect outside one tiny model; a diagnostic that fails out-of-sample calibration; a claimed barrier inferred only from a low-rank sketch; or a theoretical contribution subsumed by natural-gradient/multicriteria optimization and task-advantage theory.

**Acceptance prospects:** conditional. A replicated causal explanation with a repair and a substantive shared-representation learning theorem could form a full submission. A cosine plot, a local Fisher identity, or a small synthetic example would not.

## 9. Current deliverable and concrete limitation

Completed: primary-source problem and counterevidence audit; specific overlap checks; documented simple-fix screen; explicit mathematical target and its classical foundations; reproducible analytical controls; and a confirmatory experiment protocol.

The original execution environment hid NVIDIA device nodes and failed DNS resolution. After the user changed the environment permissions, CUDA initialization and a GPU matrix multiplication succeeded. PyTorch 2.9.1 is built with CUDA 12.9; the RTX 5060 Ti exposes about 15.47 GiB. [Current diagnostics](gpu_preflight.json) supersede the earlier failed probe. The model and dataset now download directly; Chrome fallback was unnecessary. Dependencies were installed in an isolated environment rather than modifying the existing conda environment.

A full-parameter pilot completed successfully. Its [fixed inputs](gpu_pilot_inputs.json), [implementation](gpu_pilot.py), [dependencies](gpu_requirements.txt) and [canonical result](gpu_pilot_results.json) are retained. The initial configuration was recorded before inspecting model outcomes; later controls and their declaration times are disclosed separately. It uses 48 sampled MATH training problems and 16 test problems, levels 1–4, with eight rollouts and a 1,024-token response cap. The first eight eligible training groups containing both successes and failures are selected for short interventions. Independently sampled baseline responses avoid evaluating improvement against the same outcomes used for selection.

The three arms are clipped GRPO-style replay on fixed initial rollouts, student-correct-response SFT, and expert-solution SFT. They use the same selected prompts, full parameters, four passes and the same learning rate. Their supervision and response-token budgets differ and are reported; they are not information- or token-matched comparisons. The replay arm is not an on-policy GRPO training reproduction. A same-problem prefix variant measures prompt-format robustness, not new compositional skill. This is a single-seed feasibility test, not the complete repair-baseline gate described in §7.

The previously inaccessible [released reward-statistics CSV](../../resources/rlvr_success_transfer/prompt_reward_stats.csv) also downloaded successfully. It contains 306,176 rows across 146 steps and 8,058 prompt identifiers, including duplicate step/identifier pairs. Its provenance and identifiers have not been matched to this pilot; no original-cohort balanced-panel result is claimed.

Full source replication, strong current repair comparisons and a new method remain unestablished. No paid service, external model call or public upload was used.

## 10. Other candidates screened out in this search

This is a targeted exclusion record, not an exhaustive systematic review. A source limitation is insufficient when the paper or a related current method already offers a plausible practical remedy.

| Candidate | Concrete overlap or simple repair found | Decision |
| --- | --- | --- |
| Cross-tokenizer on-policy distillation alignment | [Rethinking, 2610.08448](https://arxiv.org/abs/2610.08448), [ESCD, 2609.34738](https://arxiv.org/abs/2609.34738), and [CAST, 2610.13257](https://arxiv.org/abs/2610.13257) already address token/span/event alignment and credit. | A new alignment wrapper is insufficient. |
| Privileged-context distillation failing on harder tasks | [Privileged, but Biased, 2608.04794](https://arxiv.org/abs/2608.04794) identifies a failure; [RLSD, 2604.03128](https://arxiv.org/abs/2604.03128), [HERO, 2606.11559](https://arxiv.org/abs/2606.11559), and [AIR-OPD, 2610.02700](https://arxiv.org/abs/2610.02700) offer related repairs. | No demonstrated residual after those methods. These do not all test the same task regime. |
| Flow-model RL trained with stochastic samplers but deployed deterministically | [Understanding Sampler Stochasticity, 2510.10767v2](https://arxiv.org/html/2510.10767v2) already identifies the post-update score/marginal inconsistency and studies its effect. | The proposed central mechanism was already addressed; reject as new. |
| Diffusion-language confidence ordering shortcuts | [Confidence Shortcut, 2605.29123v2](https://arxiv.org/html/2605.29123v2) compares ordinary order changes; [DAPD, 2603.12996](https://arxiv.org/abs/2603.12996) already pursues dependency-aware decoding. | Existing order remedies make a generic dependency fix weak. |
| Visual tool evidence overriding user corrections | [VisualSwap, ICML 2026](https://proceedings.mlr.press/v306/shi26ad.html), [VRE, 2603.26348](https://arxiv.org/abs/2603.26348), and [ReGround, 2608.04385](https://arxiv.org/abs/2608.04385) include message or evidence-grounding interventions. | Need a failure surviving those controls; none established here. |
| Context compression losing jointly useful evidence | [FOCUS, 2609.37590](https://arxiv.org/abs/2609.37590), [TRACE, 2608.06503](https://arxiv.org/abs/2608.06503), and [CICL, 2606.08151](https://arxiv.org/abs/2606.08151) already use task feedback, plans or structured context. | Complementarity alone does not distinguish a contribution from existing dependency-aware selection. |
| Correlated unwanted features amplified by alignment | [Alignment Tampering, ICML 2026](https://proceedings.mlr.press/v306/hahm26a.html) documents a consequential failure. Counterfactual data, explicit feature constraints and output rewriting remain plausible repairs under different information access. | Real problem; no new necessary mechanism or repair established. |

These exclusions explain why the retained question is narrower than a generic proposal to improve RLVR. They do not prove that the retained question has passed its pending baseline gate.

## 11. Completed RTX 5060 Ti experiment

### Material Passport

- Evidence: locally generated full-parameter Qwen2.5-0.5B pilot on MATH; raw responses, token IDs and rewards retained in [one result](gpu_pilot_results.json).
- Verification status: **ANALYZED**. The script completed; no independent training rerun has been performed.
- Model/data revisions, package versions, seeds and configuration are recorded in the result. The [additional on-policy control](gpu_pilot_protocol.json) was declared during initial-policy baseline sampling, before intervention outcomes.

### Outcome table

Each cell is verified success before → after, followed by the absolute change and a descriptive 95% prompt-bootstrap interval. Intervals are not simultaneous, and one training seed does not quantify training-run variability.

| Intervention | Selected prompts (8) | Same-problem prefix variants (8) | Held-out MATH problems (16) | Training response tokens |
| --- | --- | --- | --- | --- |
| Fixed-rollout clipped replay | 43.75% → 46.88%; +3.1 pp [-9.4, +18.8] | 45.31% → 39.06%; -6.2 pp [-26.6, +15.6] | 20.31% → 10.94%; -9.4 pp [-20.3, +0.0] | 91,560 |
| Student-correct SFT | 43.75% → 81.25%; +37.5 pp [+9.4, +65.6] | 45.31% → 75.00%; +29.7 pp [+4.7, +51.6] | 20.31% → 24.22%; +3.9 pp [-5.5, +14.8] | 31,904 |
| Expert SFT | 43.75% → 78.12%; +34.4 pp [+1.6, +67.2] | 45.31% → 64.06%; +18.8 pp [-18.8, +56.2] | 20.31% → 8.59%; -11.7 pp [-24.2, -1.6] | 3,588 |
| Fresh group-relative PG, LR 5e-6 | 43.75% → 51.56%; +7.8 pp [-3.1, +21.9] | 45.31% → 60.94%; +15.6 pp [-1.6, +34.4] | 20.31% → 22.66%; +2.3 pp [-3.9, +8.6] | 80,479 |
| Fresh group-relative PG, LR 5e-7 | 43.75% → 60.94%; +17.2 pp [+0.0, +32.8] | 45.31% → 40.62%; -4.7 pp [-20.3, +14.1] | 20.31% → 21.09%; +0.8 pp [-7.0, +7.0] | 85,095 |

### Low observed-success subset

This split was declared before intervention results. It uses noisy selection counts, not a validated latent-difficulty label. Independent baseline outcomes are not used to relabel the strata.

| Selection stratum | Prompts | Independent baseline success | Intervention | Final success | Change |
| --- | --- | --- | --- | --- | --- |
| 1-2_of_8_selection_successes | 4 | 21.9% | Fixed-rollout clipped replay | 37.5% | +15.6 pp |
| 1-2_of_8_selection_successes | 4 | 21.9% | Student-correct SFT | 71.9% | +50.0 pp |
| 1-2_of_8_selection_successes | 4 | 21.9% | Expert SFT | 81.2% | +59.4 pp |
| 1-2_of_8_selection_successes | 4 | 21.9% | Fresh group-relative PG, LR 5e-6 | 34.4% | +12.5 pp |
| 1-2_of_8_selection_successes | 4 | 21.9% | Fresh group-relative PG, LR 5e-7 | 43.8% | +21.9 pp |
| 3-7_of_8_selection_successes | 4 | 65.6% | Fixed-rollout clipped replay | 56.2% | -9.4 pp |
| 3-7_of_8_selection_successes | 4 | 65.6% | Student-correct SFT | 90.6% | +25.0 pp |
| 3-7_of_8_selection_successes | 4 | 65.6% | Expert SFT | 75.0% | +9.4 pp |
| 3-7_of_8_selection_successes | 4 | 65.6% | Fresh group-relative PG, LR 5e-6 | 68.8% | +3.1 pp |
| 3-7_of_8_selection_successes | 4 | 65.6% | Fresh group-relative PG, LR 5e-7 | 78.1% | +12.5 pp |

### Formatting and truncation check

Our verifier requires a boxed answer. Training changes formatting and response length, so success gains cannot automatically be attributed to better reasoning. Conditional accuracy below is descriptive; conditioning on formatting does not identify a causal reasoning effect.

| Method | Held-out boxed-answer rate | Held-out success among boxed responses | Held-out truncation | Mean held-out response tokens |
| --- | --- | --- | --- | --- |
| baseline | 77.3% | 26.3% | 18.0% | 628 |
| Fixed-rollout clipped replay | 64.1% | 17.1% | 35.2% | 744 |
| Student-correct SFT | 94.5% | 25.6% | 7.0% | 566 |
| Expert SFT | 73.4% | 11.7% | 3.1% | 250 |
| Fresh group-relative PG, LR 5e-6 | 90.6% | 25.0% | 18.0% | 616 |
| Fresh group-relative PG, LR 5e-7 | 78.9% | 26.7% | 16.4% | 607 |

### Gradient prediction and finite updates

Full-sequence reward-gradient estimates were computed from independent baseline rollouts. The following compares their linear predictions along each realized final parameter change with the independently sampled outcome change. The gradients themselves are noisy Monte Carlo estimates; no confidence bound for their predictions was established. Final updates are not constrained to the neighborhood required by the Taylor bound. These figures therefore do not validate a representation barrier or a finite-step guarantee.

| Intervention | Predicted selected gain | Observed selected change | Predicted held-out gain | Observed held-out change |
| --- | --- | --- | --- | --- |
| Fixed-rollout clipped replay | -5.7 pp | +3.1 pp | +13.5 pp | -9.4 pp |
| Student-correct SFT | +54.7 pp | +37.5 pp | +3.1 pp | +3.9 pp |
| Expert SFT | -178.0 pp | +34.4 pp | -42.9 pp | -11.7 pp |
| Fresh group-relative PG, LR 5e-6 | +378.9 pp | +7.8 pp | +15.6 pp | +2.3 pp |
| Fresh group-relative PG, LR 5e-7 | +91.6 pp | +17.2 pp | +5.7 pp | +0.8 pp |

### Execution and interpretation boundaries

- Peak allocated PyTorch GPU memory: **9.49 GiB**. This excludes some driver/display/reserved memory.
- Original pilot wall time: 17.2 minutes; additional fresh-rollout controls: 13.4 minutes.
- Fresh-rollout reward-driven updates / sampled groups: Fresh group-relative PG, LR 5e-6: 23/32; Fresh group-relative PG, LR 5e-7: 28/32. Zero-variance groups were retained in logs and skipped for optimization.
- Normalized text overlap between sampled training and test inputs: 0. Pretraining contamination of MATH is not ruled out.
- The fixed-data replay arm and fresh-rollout arm are small group-relative training implementations, with no KL term. Neither is a reproduction of the original distributed DAPO configuration.
- The [published scorer](https://raw.githubusercontent.com/yulinchen99/unlearnability-rlvr/main/verl/utils/reward_score/hf_math_verify.py) has a default mode that can accept unboxed extracted answers; our boxed-answer requirement is a deliberate stricter pilot choice. Answer-level verification does not certify intermediate reasoning. This is another reason the pilot is not a source replication.
- Truncation, generated lengths, trained tokens and per-prompt outcomes are retained. Differences in information, trajectory length and optimization exposure prevent claiming a controlled algorithm ranking.
- Existing strong repair baselines remain outstanding. This experiment can expose a short-horizon pattern and verify feasibility; it cannot establish persistent unlearnability or the necessity of a new method.

### Statistical self-review: 11/11 checks

| Check | Disposition |
| --- | --- |
| Simpson reversal | Observed-rate strata reported alongside the pooled panel; a broad homogeneous effect is not assumed. |
| Ecological inference | Aggregate success is not attributed to every prompt or every model. |
| Berkson selection | Mixed-reward selection is disclosed; the affected population is not a random hard-prompt sample. |
| Collider conditioning | No causal representation explanation is inferred by conditioning on successful rollouts. |
| Base rates | Selection, baseline rates and all-failure groups are retained. |
| Regression to the mean | Improvement is measured against an independent baseline, not the selection rollouts. |
| Survivorship | Frozen evaluation panel retained; optimizer skips are logged. |
| Look-elsewhere | Every arm/panel is reported; intervals are descriptive and unadjusted. |
| Forking paths | Configuration and added control are disclosed; this remains exploratory. |
| Causation | Interventions change training recipes; unequal information/cost and one seed preclude a representation-mechanism claim. |
| Reverse causality | Updates precede endpoint evaluation; gradient association is not used as proof of a learning barrier. |

### Policy-change diagnostic for the source-rate control

The source-rate control additionally measures full-vocabulary conditional KL at one uniformly sampled response position per independent base-policy trajectory. Weighting by that trajectory's response length estimates the chain-rule sequence KL sum:

$$
\widehat D=\frac{1}{n}\sum_{i=1}^{n}L_i\,D_{\mathrm{KL}}\!\left(\pi_{\theta_0}(\cdot\mid h_{i,U_i})\,\|\,\pi_{\theta_1}(\cdot\mid h_{i,U_i})\right),\qquad U_i\sim\mathrm{Unif}\{1,\ldots,L_i\}.
$$

This is a classical chain-rule Monte Carlo diagnostic, not a new result or an upper bound. It uses 128 baseline trajectories across 16 held-out prompts. The additional probe was added after the first fresh-rollout run started, so it is available for the source-rate control only. Mixed-precision computation and finite gradient samples also affect first-order comparisons. No Taylor remainder constant was established.

| Control | Estimated sequence KL after first applied update (nats) | Estimated sequence KL at endpoint (nats) |
| --- | --- | --- |
| Fresh group-relative PG, LR 5e-7 | 0.199575 | 0.289594 |

### Research decision after the pilot

**Do not commit to a new methods or theory paper on this evidence.** Ordinary student-correct SFT raised selected-prompt success from 43.75% to 81.25%. In the four low observed-success cases it raised the independent baseline from 21.875% to 71.875%; expert SFT reached 81.25%. The held-out student-SFT gain was only 3.9 points, with a descriptive interval spanning zero. Its conditional success among boxed responses did not improve. These results support neither a demonstrated representation barrier nor a broadly effective new repair.

The sampled mixed-reward cohort is not the original paper's persistent, threshold-defined cohort. Even our low observed-rate subset has an independent baseline of 21.875%, above the original hard threshold. Thus this experiment does not refute the published phenomenon or establish that SFT repairs the original cases. It establishes that this pilot has not reproduced the necessary residual failure.

The uncalibrated gradient forecasts also give wrong signs or gains outside the physically possible probability range in several arms. This does not falsify the conditional Taylor theorem: gradient-estimation error, update size and mixed precision were uncontrolled for that purpose. It shows that the proposed point diagnostic is not a validated practical certificate. The underlying identities remain classical foundations rather than a new theoretical contribution.

The next scientific requirement, if this question is retained, is a fixed-cohort reproduction of the published failure with the original response cap, independent evaluation and multiple training seeds, followed by the strongest current repairs. Until a consequential residual survives, developing another weighting, geometric score or optimizer is not justified. The present candidate should remain unendorsed.

