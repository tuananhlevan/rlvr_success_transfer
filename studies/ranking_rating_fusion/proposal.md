# Research proposal: when does cardinal feedback help a restricted alignment model?

Date: 14 October 2026. Status: **conditional comparative-theory proposal**, supported by the [current results and limitations](research_report.md). No experimental results are implied by the proposed hypotheses or success conditions.

**Current progress:** [the research report](research_report.md) records completed scalar structural/statistical theory and CPU experiments. A standard reward-reconstruction baseline succeeds, so the project does not claim a new optimizer. A natural language-task application and publication-level novelty remain unestablished.

## 1. Proposed paper and central question

**Working title:** *When Does Extra Feedback Help? Ranking–Rating Fusion under Misspecified Policy Alignment.*

**Research question:** At a fixed reference policy and deployment KL temperature, when does adding accurate or unbiased cardinal feedback to binary preference fitting improve policy value, and when does its projection error outweigh its statistical benefit?

**Intuition:** a restricted model cannot adjust every response independently. Fitting a large correct score gap may push a shared feature in a direction that also favors an unwanted response. The binary likelihood and the rating likelihood weight this compromise differently. Whether additional ratings help depends on representable directions, which pairs receive weight, and how much observation noise remains—not simply on whether the ratings are correct.

The first target is a **comparative theory of existing objectives**, supported by a causal mechanism experiment. A new optimizer is optional and must earn its place through a proved or measured cost advantage. This revision removes the weakest part of the previous proposal: a generic adaptive trust gate presented as a new method.

### Intended contributions, conditional on completion

1. A substantive characterization of the ranking–rating fusion path under a specified misspecified policy family, including positive-transfer conditions and robust negative-transfer regimes.
2. A sharp statistical/projection tradeoff showing when the same rating channel helps at one sample budget and hurts at another. Its structural terms must be characterized rather than left as unspecified approximation error.
3. Controlled language-task evidence that tests the mechanism and separates representation, feedback error, temperature and optimization.
4. Only if justified: a correction that improves the attainable utility/cost tradeoff over existing value-directed and validation-based alternatives.

Items 1–3 are the minimum intended paper package. They are research objectives, not completed contributions. The three-action example and standard local calculus alone are insufficient.

## 2. Practical decision and use cases

The practical decision is how to use scores already present in an alignment dataset: fit their magnitudes through an implicit-reward objective, use them for value-directed policy updates, or withhold them from that particular fitting step.

| Potential use case | Decision this research could improve | Evidence still required |
|---|---|---|
| Adapter alignment on a fixed backbone | Whether a numeric-feedback objective is appropriate at a given trainable representation | Same-model interventions and a useful improvement over ordinary tuning |
| Response reranking with a small selector | How to use executable task scores without worsening constrained selection utility | Controlled candidate-pool pilot; naturally occurring adverse geometry |
| Reusing an existing scored corpus | Whether offline pair weighting is sufficient or fresh target-policy feedback is worth its cost | Coverage/variance analysis and matched generation/annotation accounting |

These are plausible uses. We have not established deployment prevalence or a realized saving. The source problem is model-dependent score trust in [RDPO, Appendix F.4](https://arxiv.org/html/2602.00603v1); attributing that observation to this mechanism remains a hypothesis.

## 3. Scope, assumptions and observation model

Start with finite actions and one prompt, then extend to shared features over prompts. Let the reference have full support and let

$$
\pi_\theta(a\mid x)=\frac{\pi_0(a\mid x)e^{\theta^\top f(x,a)}}{Z_\theta(x)},\qquad
J_\tau(\theta)=\mathbb E_{x,a\sim\pi_\theta}r(x,a)
-\tau\mathbb E_x\mathrm{KL}(\pi_\theta(\cdot\mid x)\|\pi_0(\cdot\mid x)).
$$

The main structural theory should initially use an identifiable, fixed-dimensional exponential family, bounded rewards/features and a specified compact parameter region. Population minima should be interior for derivative statements. Handle boundary cases separately. No global theorem for arbitrary transformers is promised.

Observe pairs from an explicit design $q(x,a,b)$, ranking bits $Z$ and rating gaps $G$:

$$
\Pr(Z=1\mid x,a,b)=\sigma(r(x,a)-r(x,b)),\qquad
G=r(x,a)-r(x,b)+\xi.
$$

The first statistical setting assumes conditionally mean-zero noise with stated tail/variance bounds; exact ratings set $\xi=0$. The initial independent-channel model may impose conditional independence of $Z$ and $\xi$, but a realistic extension must retain their cross covariance when the same scorer generates both. Biased scores require a separate term and should not be relabeled accurate.

With $z_\theta=\tau\theta^\top[f(x,a)-f(x,b)]$, define

$$
\widehat L_\lambda(\theta)=\frac1n\sum_{i=1}^n
\left[\operatorname{CE}(Z_i,\sigma(z_\theta^{,i}))
+\frac\lambda2(z_\theta^{,i}-G_i)^2\right].
$$

Allow different pair designs for the two channels only when explicitly declared. Sampling changes must be accounted for; the default source-objective comparison uses the same design. Keep $\tau$ fixed while changing $\lambda$, rather than silently altering the deployment reward/KL tradeoff.

Evaluate **restricted policy regret**:

$$
\mathcal E(\theta)=\sup_{u\in\Theta}J_\tau(u)-J_\tau(\theta).
$$

Report raw task utility and KL separately as well. Pairwise accuracy and gap MSE are explanatory metrics, not replacements for deployment value.

## 4. Theoretical program and formula support

### T0. Completed foundation

The [structural theory](evidence/fusion_structural_theorems.md) contains a complete fixed-temperature, full-pair-support example where every positive rating weight decreases population value while reducing rating MSE. It also derives the standard local condition

$$
\left.\frac{dJ_\tau(\theta_\lambda)}{d\lambda}\right|_0
=-\nabla J_\tau(\theta_B)^\top H_B^{-1}\nabla R(\theta_B).
$$

In the exponential family this becomes $\tau v^\top H_B^{-1}m$, where $v$ is the policy-weighted reward-residual covariance and $m$ its rating-pair-weighted counterpart. Current-policy rating pairs give $m=2v$ and a nonnegative local effect. These identities have been numerically checked. They are standard tools and controls, not sufficient novelty.

### T1. Main structural theorem: characterize the path, not just its derivative

**Partially achieved:** the structural note proves scoped scalar transfer classifications and finite-weight certificates. The broader target is to prove nonlocal transfer conditions for $\theta_\lambda$ over an explicit policy family and interval of rating weights. The result should distinguish:

- reward residual direction from residual magnitude;
- binary saturation from missing support;
- rating-pair design from deployment-policy weighting;
- fixed representation from increased representational capacity;
- small-step improvement from fully optimizing a frozen objective.

One concrete starting point is one-dimensional exponential families with ordered features, at least three actions, arbitrary positive references and connected pair designs. Classify reward/feature configurations for which the fusion path moves toward versus away from the value-optimal parameter; identify when the classification survives small perturbations. Then extend to a genuinely multidimensional family, such as controlled feature subspaces with quantified pair/deployment moment mismatch. A scalar dot product rewritten as a theorem does not meet this target.

**Proof obligations:** uniqueness or a specified stable optimizer branch; an explicit path-ordering or curvature argument; a nontrivial sufficient class and counterexamples beyond it; fixed reward units and temperature; coordinate/gauge invariance; quantitative margins. A theorem describing only the handcrafted instance remains a preliminary result.

### T2. Statistical benefit versus projection harm

Let $\theta_\lambda$ denote the population minimizer, $A_\lambda=\mathcal E(\theta_\lambda)$ and $\widehat\theta_\lambda$ its empirical estimate. The proposed analysis must explain the dependence of $A_\lambda$ on the geometry from T1, not merely introduce its name.

For a regular fixed-dimensional interior estimator, a useful scaffold is the sandwich covariance

$$
\Sigma_\lambda=H_\lambda^{-1}C_\lambda H_\lambda^{-1},\qquad
C_\lambda=\operatorname{Cov}(\nabla\ell_B+\lambda\nabla\ell_R).
$$

The cross terms in $C_\lambda$ matter for correlated channels. Under sufficient moment, differentiability and bias-expansion assumptions, write

$$
\mathbb E(\widehat\theta_\lambda-\theta_\lambda)=\frac{b_\lambda}{n}+o(n^{-1}),\qquad
\mathbb E[(\widehat\theta_\lambda-\theta_\lambda)(\widehat\theta_\lambda-\theta_\lambda)^\top]
=\frac{\Sigma_\lambda}{n}+o(n^{-1}).
$$

A Taylor expansion then gives the expected-value comparison

$$
\mathbb E\mathcal E(\widehat\theta_\lambda)
=A_\lambda-\frac1n\left[
\nabla J_\tau(\theta_\lambda)^\top b_\lambda
+\frac12\operatorname{tr}(\nabla^2J_\tau(\theta_\lambda)\Sigma_\lambda)
\right]+o(n^{-1}).
$$

Call the negative bracket $V_\lambda$. It need not be nonnegative at a misspecified surrogate optimum. Dropping the estimator-bias term or assuming the value Hessian is negative semidefinite there would be unjustified. This expansion is a standard research scaffold, not a completed uniform or finite-sample theorem.

The candidate crossover relation is

$$
\mathbb E\mathcal E(\widehat\theta_\lambda)
-\mathbb E\mathcal E(\widehat\theta_0)
=(A_\lambda-A_0)+\frac{V_\lambda-V_0}{n}+o(n^{-1}).
$$

If the structural penalty is positive and the finite-sample correction is favorable, this suggests a sample-budget crossover. Whether such a regime exists broadly, where the approximation is valid, and whether a sharp finite-sample bound matches it are open. Do not extrapolate a large-sample expansion to tiny datasets.

**Desired substantial result:** explicit upper and matching lower comparisons in a specified class, with characterized dependence on saturation, rating noise and pair/deployment mismatch. A generic asymptotic normality statement plus an unknown $A_\lambda$ is insufficient. High-probability guarantees must allow a first-order value term of order parameter error when $\nabla J_\tau(\theta_\lambda)\ne0$; expected $1/n$ expansions do not erase that issue.

### T3. Correction and cost: optional, gated

A diagnostic baseline estimates $S$. A correction baseline reweights rating pairs toward the current policy, using frozen importance weights

$$
w_B(x,a,b)=\frac{\pi_B(a\mid x)\pi_B(b\mid x)}{q(a,b\mid x)}.
$$

This requires known sampling probabilities and overlap. If a moment bound is absent, clipping or fresh sampling becomes necessary and introduces an explicit bias/cost tradeoff. The prompt law also requires correction if it differs from deployment. Self-pairs contribute zero; conditioning conventions need matching normalization.

For $J_\tau$ with an $L$-Lipschitz gradient, a candidate update $d$ satisfies the elementary bound

$$
J_\tau(\theta+\eta d)-J_\tau(\theta)
\ge\eta\nabla J_\tau(\theta)^\top d-\frac{L\eta^2}{2}\|d\|^2.
$$

Estimating the right-hand side from observed gaps requires an audit-error bound, curvature control and honest sample splitting or adaptive inference. This is standard smooth optimization, not a new method guarantee.

**A new algorithm is justified only if** it attains a stronger utility/cost tradeoff than REBEL, direct reward optimization, AuxDPO and ordinary validation under the same information and computation contract. The connected exact-gap benchmark does not justify a lower bound requiring new comparisons: its rewards can already be reconstructed. Any feedback lower bound must specify sparse/noisy observations, coverage or another real restriction. Do not introduce an artificial restriction solely to disable the baseline.

## 5. Experiments: decisive tests before scaling

### E1. Finite-action population and finite-sample study

**Population grid:** action counts 3, 8 and 32; feature ranks from one to the identifiable full rank; reward/feature geometry including aligned, nonmonotone and random controls; multiple reference entropies and pair designs. Sweep reward scale and temperature separately. Report saturation through the actual BT margins, not scale alone.

**Noise/sample grid:** exact gaps, unbiased gap noise, biased-gap control; independently generated BT bits versus bits derived from the same score channel. Use prespecified sample budgets spanning 64 to 16,384 for the first stochastic sweep, adjusting only when convergence diagnostics show these fail to resolve a relevant transition. Use repeated label draws; do not confuse them with independent population instances.

Use compact parameter bounds to avoid finite-sample separability giving infinite binary estimates. Verify that the bounds do not determine the population comparison. In scalar cases compute the restricted value optimum; in multidimensional cases report optimization certificates or honest uncertainty in the oracle reference.

**Primary outcomes:** restricted regret, value, gap MSE, binary CE, empirical transfer sign, predicted transfer sign, optimizer residual and budget. Evaluate local-sign prediction at small weights separately from nonlocal value at larger weights. Report all prespecified random instances, including beneficial-rating cases.

**Go condition:** the effect occupies a meaningful, reproducible region of controlled geometries; the structural characterization predicts positive as well as negative cases; finite-sample tests match an actual derived crossover where claimed. **Stop condition:** the only result is the original toy or a regression explained by optimizer tolerance or changed temperature.

### E2. Controlled language response-selector pilot

Use executable instruction constraints from the [official IFEval implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval) as a starting point. Create or select a training prompt set with separate development and untouched evaluation splits; do not train on a benchmark later described as unseen. Split by prompt/constraint template where possible. Audit checks on a small hand-inspected subset and record utility definitions before fitting.

Cache 8–16 candidates per prompt from one fixed generator and decoding configuration. Define an operational reward as a prespecified weighted count of satisfied constraints. Use known BT bits and exact reward gaps for the mechanism stage; these are synthetic observations, not human labels.

Train a selector on frozen language representations plus transparent feature controls. Vary only its bottleneck dimension or nested feature subspace. Establish that the selector can express different decisions at these settings; nominal rank changes alone do not prove changed task capacity. Compare a full-feature positive control and shuffled-reward negative controls. Keep candidate coverage and reference probabilities identical.

**Baselines:** binary DPO; ML-RDPO weight grid; gap-only regression; direct empirical value optimization; on-policy/importance-corrected signed-gap updates; AuxDPO when implementable in the same family; ordinary holdout model selection. Include a MetaPO-style influence/validation control if the candidate correction uses that mechanism. If an implementation cannot be matched, report the discrepancy instead of marking the baseline beaten.

**Analysis:** split prompts are the main generalization unit. Bootstrap paired prompt differences and report training-seed variability separately. Prespecify the primary value comparison and a practical-effect threshold before seeing test results. Use a pilot variance estimate to determine main-study size rather than inventing an expected win-rate gain. Account for every method's hyperparameter search and audit examples.

**Go condition:** exact-gap fitting improves while selection value worsens on held-out tasks in controlled restricted settings; the predicted geometry explains the direction; existing methods provide informative positive controls. **Stop/demote condition:** no effect beyond artificial features, or a simple baseline removes it at equal cost and leaves no strong theoretical distinction.

This is a reranking experiment. It can validate a useful mechanism without proving that the same effect controls generative fine-tuning.

### E3. Generative adapter study, conditional on E1/E2

Select one accessible small instruction backbone, pin the checkpoint/tokenizer and vary adapter rank, initially 4, 16 and 64, with fixed target layers. Use the same SFT reference, prompt data, candidate/scorer source, reward units, deployment temperature and evaluation protocol. The specific model and hardware budget are not yet verified; do not commit to an unmeasured GPU-hour estimate.

Use three independent training seeds in the pilot and increase to five for a surviving main configuration if seed variability requires it. Compare fixed-temperature ML precision sweeps, optimized DPO, relevant auxiliary/value-directed controls and equal-budget validation. Real corpus scores are noisy proxies; evaluate a verifiable-reward condition separately from judge-based general alignment.

Report independent task utility, empirical KL, instruction success, capability retention, response length, relevant refusal behavior, training time, generated tokens, scorer calls and search cost. Capabilities and benign refusal should be measured where the selected tasks make them relevant, not added as unrelated benchmarks.

**Main-study justification:** a reproducible utility benefit, a demonstrable tuning-cost saving, or a predictive mechanism that changes a practical choice. An isolated MSE improvement is not success.

### Required ablations

| Confound | Intervention | Expected discriminating pattern |
|---|---|---|
| Representation versus model size | Same backbone/features, changed trainable subspace | Outcomes follow geometry; no assumed monotone rank law |
| Optimization | Converged population solves; matched loss scaling and SGD residuals | Projection failure persists after convergence |
| Feedback error | Exact, unbiased noisy, biased and dependent channels | Exact condition isolates projection; other errors are additional mechanisms |
| Sampling | Reference pairs, current-policy pairs, corrected cached pairs | Local transfer follows the derived moment condition |
| Temperature | Fixed deployment coefficient versus changed coefficient | Reported harm cannot be attributed to a different reward/KL target |
| Saturation | Independent reward-scale/temperature sweep | A specified theorem predicts where BT saturation matters |
| Finite step | Small fusion weight versus deep frozen-objective optimization | Local guarantee is not misreported as global safety |
| Cost | Equal feedback, rollout, validation and training budgets | Proposed improvement survives honest baseline tuning |

## 6. Expected results, without fabricated effect sizes

Established: the exact toy exhibits negative transfer, and frozen current-policy rating weights reverse its infinitesimal sign. Expected under stated assumptions: realizable population controls should share their solution; numerical violations indicate a bug or a changed target.

Hypothesized: some naturally occurring restricted language representations show the same reward-fit/value separation; residual and sampling geometry predict it better than rank alone; finite-sample rating benefits can disappear as projection error dominates. These hypotheses may fail. There is no justified forecast of a benchmark win-rate increase or conference acceptance probability.

If direct optimization or REBEL handles the adverse regime at no additional cost, that supports the diagnosis but weakens the case for another optimizer. The project may still support a strong comparative theory if T1/T2 genuinely add new results. Without those results, publishability should be reconsidered rather than rescued by renaming a standard method.

## 7. Current required steps, in dependency order

| Priority | Step and concrete output | Current state | Completion gate |
|---:|---|---|---|
| 1 | Proof-level novelty matrix for AuxDPO, RDPO, REBEL, Boltzmann projection, MetaPO and policy-aware design | Proof-level matrix completed; exhaustive clearance remains open | One precise theorem statement whose assumptions/guarantee differ substantively |
| 2 | Formalize T1 in an explicit model class; proof and boundary counterexamples | Scoped scalar path/classification and multidimensional certificates proved | Nontrivial positive/negative classes and robust margins, not a renamed derivative |
| 3 | Implement the E1 population/finite-sample harness and record all settings | 193 population configurations and paired Monte Carlo study completed | Mechanism survives controls; honest prevalence across prespecified instances |
| 4 | Derive T2 using complete covariance/bias terms; seek a matching lower comparison | Explicit scalar bias/remainder and local-alternative crossover proved; lower comparisons open | Characterized crossover or a clear impossibility result under a justified contract |
| 5 | Build audited candidate cache, train/dev/test splits, executable utility and E2 selectors | Not started | Reproducible held-out task demonstration or an explicit negative result |
| 6 | Reproduce matched baseline objectives and cost accounting | Same-gap reward reconstruction completed; language baselines pending | Baselines converge and receive equal information/tuning opportunity |
| 7 | Decide theory-only versus theory-plus-method; specify correction if earned | Comparative theory selected; no new-optimizer claim | Existing repair's limitations demonstrated; otherwise omit a new-method claim |
| 8 | Freeze main analysis, endpoints, practical threshold and study size | Not started | Pilot variance and mechanism justify the planned main comparison |
| 9 | Run E3 only after earlier gates; measure complete pipeline cost | Not started | Required checkpoint/hardware access verified; generative claims supported |
| 10 | Write full manuscript, proof appendix, reproducibility package and final literature update | Proposal/analysis available | Every major claim has proof, data or an explicit limitation; external technical critique obtained |

Steps 2 and 3 can progress together after the novelty statement is fixed. Steps 4 and 5 are the next substantial theory/empirical pair. Do not fund a large E3 sweep before those gates. The current report-writing request is complete; the table describes future research execution.

### What is needed to execute, rather than plan

- An isolated experiment environment with numerical/ML dependencies and pinned versions. The current default Python environment only supported standard-library checks.
- Verified public-data/checkpoint access and cached data provenance. Paper PDFs alone do not provide a training dataset.
- For E2/E3, a measured hardware profile: accelerator memory, generation throughput, sequence length and one-run training time. A CPU-only E1 can start first.
- A decision on the main paper contract after T1/E1: comparative theory is the default; an optimizer claim requires additional evidence.

No new human study is necessary for the initial mechanism tests. Independent human/LLM evaluation, if later used, must have a declared measurement target and accounted cost.

## 8. Definition of a complete research project

**Minimum scientific package:** a cleared and fully proved substantive T1 result, T2 tradeoff or sharp boundary result, controlled E1/E2 evidence, competent direct/value-directed/validation baselines, complete ablations, uncertainty estimates, limitations and reproducible artifacts. A generative stage is required for claims about improving LLM fine-tuning pipelines; it is not required for a carefully scoped finite-action theory paper with a selector application.

**Paper outline:** motivation and practical decision; observation/policy model; structural transfer theorem; statistical/projection theorem; method only if earned; controlled and language experiments; cost and competing explanations; limitations. Place full proofs, checker definitions, seeds, prompts and hyperparameter-search accounting in the supplement.

For a theory-centered submission to ICML/ICLR/NeurIPS, theorem novelty and nontrivial applicability must carry the paper. For a language-centered ACL/EMNLP submission, a real language-task benefit and strong evaluation must carry it. These are scope choices, not predictions of acceptance.

**Terminate or substantially revise** if no sharp theorem survives the closest-work comparison, no natural task reproduces the effect, or the only remedy is an equally cheap standard baseline. A scientifically useful negative finding is preferable to a proposal whose contribution is mostly terminology.

## 9. Canonical deliverables

- [Current research report](research_report.md): findings, validity limits, executed experiments and next requirements.
- [Structural proofs](evidence/fusion_structural_theorems.md) and [statistical proofs](evidence/fusion_statistical_theory.md).
- [Novelty matrix](evidence/fusion_theorem_novelty_matrix.md).
- [Structural checks](verification/fusion_structural_checks.py), [high-precision moment checks](verification/check_fusion_moment_law.py), and the experiment runners listed in [the index](README.md).

The completed theory is scoped to its explicit finite-action assumptions. A useful new algorithm, natural-task impact and a complete conference manuscript remain unestablished.
