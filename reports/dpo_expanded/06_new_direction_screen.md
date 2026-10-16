# New direction screen: when can implicit feedback justify a policy update?

**Date:** 16 October 2026. **Status: NO-GO as a main research project after the completed public-data audit and expanded prior-art check.** The current anonymous-timing formulation is retired. No replacement main project has passed the practical and novelty gates. The previous conditional-design direction remains a theoretical note.

**Completed artifacts:** [reproducible CPU audit](../../studies/feedback_timing/audit.py), [canonical results and provenance](../../studies/feedback_timing/results.json). This report is the canonical interpretation; earlier candidate language below describes the hypothesis that was screened.

## 1. Candidate and practical decision

**Research question:** Under what observable conditions can anonymous choice-and-time feedback certify an improvement in a language model's preference objective despite heterogeneous decision processes?

The practical decision is whether an alignment pipeline should use recorded annotation times to change its training targets, retain choice-only supervision, or obtain additional explicit feedback. Timing is attractive because collecting it can be inexpensive. Its value depends on whether it adds reliable preference information, not merely another predictor correlated with labels.

The proposed contribution would concern **identifying a beneficial decision under measurement uncertainty**. Neither collecting response times, fitting a drift-diffusion model, adding annotator effects, nor applying a generic robust objective would be a new contribution.

**Why the candidate was screened out:** the documented NLP weakness does not establish our proposed causal mechanism; the available timing field cannot isolate it; the tested ordinary predictors leave no consistent timing benefit; and closer prior work already occupies much of the decision-identification framing. The exact correlated additive-delay model is not proven solved, but an unoccupied model variant alone is insufficient grounds for investment.

## 2. Evidence, intuition, and reading scope

All numbers in this section are **the authors' results**, not our experiments. These are primary manuscripts; searches do not establish exhaustive novelty. The PRISM reward-model paper below is distinct from the PRISM human-feedback dataset.

| Primary source and inspected sections | Intuition | Evidence or contribution relevant to the decision | What it does not establish |
|---|---|---|---|
| [Response Time Enhances Alignment with Heterogeneous Preferences](https://arxiv.org/html/2605.06987), 2026; §§2–5, Appendices B.1–B.2 | Timing may distinguish strong and weak preferences that produce the same choice. | Identifies average drift with anonymous observations under a common decision boundary. Handles a common deterministic non-decision delay. With heterogeneous boundaries, its boundary estimator instead approaches the smallest boundary in the population; observed-stratum conditioning is an offered remedy. Random heterogeneous non-decision time remains open. | Its real-data study uses intertemporal monetary choices, not LLM-response annotations. The results do not demonstrate timing-based LLM policy improvement under heterogeneous nuisance processes. |
| [ResponseRank](https://arxiv.org/html/2512.25023), NeurIPS 2025; §5, Appendix D.2, D.7, discussion | Compare preference strength locally to reduce differences in people's baseline speeds. | On MultiPref, RR-RT remains weak after annotator and length stratification. At full data, RewardBench 2 scores are 52.6 for BT, 47.2 for RR-RT and 55.6 for agreement-based RR-Agree, averaged across 30 splits/seeds. Appendix D.7 finds little benefit from changing length stratification. | The protocol includes other annotation tasks; their contribution to time is an explanation, not a demonstrated causal attribution. These are reward-model benchmarks, not direct generation utility. Explicit slight/clear strength has only a modest, nonsignificant overall advantage over BT; it is not a proven universal repair. |
| [Your Mouse and Eyes Secretly Leak Your Preference](https://arxiv.org/html/2606.20482), 2026; §§3–5, conclusion | Reading behavior contains information, but depends on how text is displayed and consumed. | Actual LLM-response interactions show length, layout, position and individual variation in reading trajectories. The paper also demonstrates benefits from richer behavioral predictors. | It does not show our anonymous timing estimator fails, establish a causal effect of layout on latent utility, or prove that all implicit feedback is unreliable. |
| [Preference Learning with Response Time: Robust Losses and Guarantees](https://arxiv.org/html/2505.22820), 2025; model, policy-loss discussion, Appendix A | Errors in an estimated response-time model need not enter reward estimation at first order. | Orthogonal losses and guarantees; analysis includes unknown barriers and non-decision-time estimation error. | A new nuisance-robust timing loss or generic DPO-plus-time objective would collide with this work. Arbitrary correlated population heterogeneity is a different contract, not automatically an unsolved problem. |
| [Enhancing Preference-based Linear Bandits via Human Response Time](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e2dd2f1efbc6e65b68f17ce6e158b34-Abstract-Conference.html), NeurIPS 2024; abstract and model/estimator passages | A fast, consistent choice can contain strength information even when choices saturate. | Timing-aware utility estimation and fixed-budget best-arm identification already exist. | Timing-aware acquisition and better information per unit time are not new proposals. |
| [Testing the drift-diffusion model](https://pmc.ncbi.nlm.nih.gov/articles/PMC7776861/), 2020; identification and related-work discussion | A good fit to reaction times is not automatically identification of their psychological cause. | Identification/testing under stated restrictions; discusses difficulty separating random non-decision time and boundaries. | Generic nonidentification of timing components is old. It cannot be the main novelty claim. |
| [Time is Knowledge: What Response Times Reveal](https://arxiv.org/html/2408.14872v4), June 2026 revision of a 2024 preprint; §§2.2–2.6, §3.1, Appendix B | A useful decision may be identifiable even if the full latent model is not. | Characterizes properties shared by observationally equivalent models; handles uncertainty over timing functions and studies speed heterogeneity, including preference–speed dependence under stated normalization assumptions. Includes an optimal-nudge application. | Its chronometric model and noise extensions do not solve every stochastic DDM with arbitrary correlated additive delays. Nevertheless, our broad conceptual framing substantially overlaps. |

**Additional calibration collisions:** [Factoring Out Nondecision Time](https://www.ppw.kuleuven.be/okp/_pdf/Verdonck2016FONDT.pdf), 2016, already uses convolution across experimental conditions to remove a shared nuisance distribution. A [revised 2026 eLife study](https://elifesciences.org/reviewed-preprints/109730v2) uses external non-decision-time information and discusses earlier calibration alternatives. Relevant sections were inspected by a worker. These are mandatory comparisons where their shared-nuisance/repeated-condition assumptions apply.

### Interpretation

There is a meaningful tension between an identification result for a specified measurement process and disappointing timing-based learning in actual text annotation. They are **not contradictory findings**: their observation protocols and targets differ. A useful study must determine whether resolving that difference changes the quality of alignment decisions after strong existing remedies.

The initial opportunity was to characterize which uncertainty matters for a policy decision. The expanded search substantially weakened that novelty argument. Merely obtaining a better psychological parameter fit would not satisfy the practical goal.

## 3. Prior-art and simple-remedy gate

| Proposed shortcut | Existing answer / required control | Disposition |
|---|---|---|
| Add response time to preference learning | NeurIPS 2024, robust-loss work, ResponseRank and 2026 anonymous estimator | Occupied. |
| Separate cautious users from users with weak preferences | Established hierarchical DDMs jointly fit drift, boundary and non-decision time; see the primary references above. | Occupied at this level. A familiar random-effects model is a baseline. |
| Subtract a common delay | Appendix B.1 of the 2026 paper already does this. | Occupied. |
| Regress out response length or stratify by annotator | ResponseRank already evaluates this; the 2026 paper also suggests observed strata. | Required baseline, not a contribution. |
| Use a first-order robust nuisance estimator | The 2025 orthogonal-loss paper directly addresses this. | Required baseline. |
| Add one calibration task or measure decision onset | Existing cognitive-model calibration and deconvolution methods must be compared; a calibration interface alone is not new. | Not a novelty claim. |
| Collect agreement or an explicit strength label | ResponseRank evaluates both. Additional annotations may be preferable to recovering timing information. | Compare at matched annotation time and expense, including choice-only with more labels. |
| Learn each person separately | Relevant when repeated observations exist. | Compare with repeated-ID and temporary-session baselines. Do not hide available IDs to manufacture an advantage. |

**Updated novelty assessment:** [Time is Knowledge](https://arxiv.org/html/2408.14872v4) already makes identification of decision-relevant properties, without complete latent recovery, central to its analysis. Its §2.6.1 examines robustness across admissible timing functions; §2.6.3 and Appendix B study speed heterogeneity and baseline normalization. Earlier [Time Will Tell](https://www.journals.uchicago.edu/doi/full/10.1086/713732), JPE 2021, also recovers preference information from choice and timing under explicit assumptions. A stochastic DDM with unrestricted type-correlated additive delays remains technically distinct; no exact solution was located in the inspected sources. That distinction does not establish a useful or sufficiently substantial contribution.

### What would make it too trivial?

If ordinary stratification, nuisance-robust fitting, a standard hierarchical DDM, or spending the same time on more binary labels yields the same decision quality, stop the methods proposal. A theorem about a broader adversarial nuisance class would not rescue it without independent theoretical importance.

## 4. Mathematical formulation and what our earlier theory contributes

For pair features $\psi$, let a labeler's utility difference be $V=\psi^\top\theta$. Consider the unit-diffusion model

$$
dX_t=V\,dt+dW_t,\qquad
T=\inf\{t:|X_t|=B\},\qquad
Z=\operatorname{sign}(X_T),\qquad
S=T+A.
$$

Here $B$ is the decision threshold and $A$ is time outside evidence accumulation. Only $(\psi,Z,S)$ is observed in the anonymous regime. Heterogeneity may correlate $\theta$, $B$ and $A$. Choosing this model is an assumption to test, not a general model of reading.

Specify a fixed utility normalization and target population before treating $\bar\theta=\mathbb E[\theta]$ as meaningful. Timing does not by itself resolve interpersonal cardinal comparability.

Let $P$ be the observable population law and $\mathcal G$ a stated class of admissible latent processes. Define

$$
\Theta(P;\mathcal G)
=\{\mathbb E_G[\theta]:G\in\mathcal G,\ P_G(\psi,Z,S)=P\}.
$$

One standard decision-theoretic template is

$$
\max_\pi\ \inf_{\vartheta\in\Theta(P;\mathcal G)}
\left\{
\mathbb E_{x,y\sim\pi}[\vartheta^\top\phi(x,y)]
-\mathbb E_{x,y\sim\pi_0}[\vartheta^\top\phi(x,y)]
-\tau\mathbb E_x\mathrm{KL}(\pi(\cdot|x)\|\pi_0(\cdot|x))
\right\}.
$$

This robust objective is **standard scaffolding**, not our new method. A full result would need an observable finite-sample uncertainty set, useful rather than vacuous guarantees, and upper/lower bounds for the actual permitted measurements. Parameter nonidentification need not prevent decision identification, but that distinction is already present in the closest literature.

| Earlier finding | Legitimate contribution to this candidate | What cannot be imported |
|---|---|---|
| Unknown-link/cardinal-calibration analysis | Test whether two measurement processes imply different preferred policies despite identical observations. | Its old identification result does not prove a new timing theorem. |
| Policy-capacity and misspecification analysis | Separate measurement error from inability of a restricted learner to implement the desired update. | A linear toy model cannot diagnose a neural policy by itself. |
| Conditional/global information geometry | Once measurement parameters are identified, evaluate whether calibration covers decision-relevant feature directions. | The fixed-conditional $d^2$ lower bound does not apply to human timing or adaptive session queries. |
| Generated-pair versus queried-label accounting | Count candidate generation, explicit labels, calibration trials and participant time separately. | A human-information lower bound cannot be inferred from a generation bound. |
| Caching/relabeling findings | Reusing a timestamp does not add an independent observation; fresh decisions and repeated stored evaluations have different information. | Cached generation tradeoffs do not imply timing sample-complexity rates. |
| Ordinary repairs defeating constructed failures | Test the strongest simple remedies before deriving another hard-instance theorem. | None of our previous closed directions is reopened by analogy. |

### An illustrative boundary, not a claimed novel theorem

The following shows why a decision-level question is logically distinct from fitting observed timings. It is a constructed example, not evidence about actual annotators.

Let $D_1$ be the first-passage-time distribution with drift magnitude 2 and boundary $1/2$, and $D_2$ that with drift magnitude 1 and boundary 1. In either case $B|V|=1$. Conditional on type, the exit sign and exit time are independent for this symmetric constant-boundary DDM, and the sign probability is $\sigma(2BV)$.

Construct two populations, each with two equally probable types:

| Population | Type 1 $(V,B,A)$ | Type 2 $(V,B,A)$ | Mean drift |
|---|---|---|---|
| $+$ | $(2,1/2,\ A\sim D_2)$ | $(-1,1,\ A\sim D_1)$ | $+1/2$ |
| $-$ | $(1,1,\ A\sim D_1)$ | $(-2,1/2,\ A\sim D_2)$ | $-1/2$ |

Each additive delay is sampled independently of that type's DDM trajectory. Both types in both populations have total-time distribution $D_1*D_2$. Both populations have pooled positive-choice probability $1/2$. Thus their entire observable $(Z,S)$ law is identical, while the optimal direction for average utility is opposite. The nuisance distribution is allowed to depend on latent type; restricting it can break the construction.

For binary actions with utility gap equal to mean drift, uniform reference and KL coefficient $\tau=1$, the optimal positive-action probabilities are 0.622459 and 0.377541. The minimax regularized regret of an observation-only rule is attained by probability $1/2$ and equals $\log\cosh(1/4)=0.0309298$. Unlike our earlier dimension-dependent example, this illustration has a fixed nonzero decision gap. That alone does **not** establish significance or novelty.

The numeric identities were checked with Python's standard library. They are neither simulated participant results nor language-model results. The public-data audit below does not reproduce this constructed mechanism.

**Investment implication of our theory:** if both populations remain admissible, the robust improvement objective cannot certify a nonzero beneficial update: their utility directions oppose each other, and the reference policy attains the safe value zero. Additional defensible assumptions or measurements are necessary. This is a useful reason to reject an unsupported method proposal, not evidence that we have a practical timing method. The previous information findings strengthen this diagnosis; they do not supply the missing observation protocol or a new theorem with established priority.

## 5. Executed public-data audit

### Data and target

Downloaded the complete public MultiPref training JSONL at revision `12910233a0238a997ebe425656e9dfed7b0ff031`. The 64,787,194-byte file has SHA-256 `027072c52e39300c55ebc34265dda5ef2cfb1682edbc22b1a7d978c94776aaca`. The runner verifies this hash before extraction; raw data remains in an ignored local cache.

The release contains **10,461 response comparisons and 41,844 annotations**. Excluding 10,997 ties leaves **30,847 non-tie annotations**, covering 5,304 prompt IDs and 227 evaluator IDs. The target is whether an annotator reports a **clear** rather than **slight** preference (43.36% clear). It is not correctness, independently measured cardinal preference, or policy utility. We pool A and B choices for this strength-prediction audit.

Inspected all released annotation field names and the [official ResponseRank MultiPref loader](https://github.com/timokau/response-rank/blob/8bc9b17ca59db5e1502fe3783acef482975d761c/src/responserank/llm/data/datasets/multipref.py) and [stratification implementation](https://github.com/timokau/response-rank/blob/8bc9b17ca59db5e1502fe3783acef482975d761c/src/responserank/llm/data/stratification.py). The released measure is `time_spent`; no separate choice-click timestamp is exposed. The total annotation duration includes the multi-field task described by ResponseRank. A single annotation-level `timestamp` cannot separate reading, decision, justification and auxiliary judgments. Therefore this release cannot establish that heterogeneous decision thresholds or additive delays cause the reported reward-model weakness.

### Protocol actually run

Executed ten CPU evaluations: five fixed seeds for held-out prompts and five for held-out annotators, with 20% of distinct split groups held out. This is an exploratory screening audit, not a preregistered confirmatory experiment. All annotations of a held-out prompt stay together. Annotator holdout has no evaluator overlap, but prompts may overlap; it is not a joint prompt-and-annotator cold-start test.

Compared four ridge-logistic models with fixed regularization and no test-selected hyperparameters:

1. **Length/cohort:** log total pair length, log shorter-response length, log absolute response-length difference, their squares, and normal/expert cohort. Total pair length counts the prompt twice, matching the two conversations.
2. **Length/cohort + flexible time:** adds log duration and a piecewise-linear timing basis with training-only quantile knots.
3. **Calibrated metadata:** adds a smoothed per-annotator clear-label rate. This is an ordinary identity-based predictive baseline, not an anonymous deployment method.
4. **Calibrated metadata + time:** adds log duration centered within annotator-by-eight-length-bucket strata, plus its square. Sparse strata fall back to annotator then global means. Training rows leave themselves out of calibration statistics; test rows use training statistics only. Unseen annotators receive global fallbacks.

We also evaluate the simple score “shorter time implies stronger preference,” raw and normalized. Reported AUROCs, log losses and Brier scores are held-out quantities. The five-split means are descriptive, not independent replications. Per-split 95% intervals in the JSON resample prompt clusters 1,000 times conditional on the fitted model; they do not capture full two-way annotator/prompt or training uncertainty.

Reproduce from the repository root with a Python environment containing NumPy and SciPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studies/feedback_timing/audit.py
```

The executed environment was Python 3.11.14, NumPy 2.4.0 and SciPy 1.16.3; the JSON records the exact versions and runner hash. The measured computation took about four seconds after data download. No GPU is required. Re-running overwrites the single canonical `results.json`.

### Observed results

| Strength predictor / metric | Held-out prompts, mean of 5 splits | Held-out annotators, mean of 5 splits |
|---|---:|---:|
| Raw shorter-time score, AUROC | 0.53025 | 0.52679 |
| Normalized shorter-time score, AUROC | 0.50932 | 0.52679 |
| Length/cohort, AUROC | 0.64545 | 0.63299 |
| Length/cohort + flexible time, AUROC | 0.64517 | 0.63205 |
| Calibrated metadata, AUROC | 0.73815 | 0.62766 |
| Calibrated metadata + time, AUROC | 0.73809 | 0.62731 |
| Log-loss reduction from flexible time, nats/annotation | −0.00003242 | −0.00033284 |
| Log-loss reduction from normalized time after calibration, nats/annotation | −0.00004645 | −0.00030880 |

**Metric intuition:** AUROC 0.5 is chance ordering of clear versus slight labels. The raw timing score has a weak association with stated strength. Metadata is more predictive in these models. A log-loss reduction is baseline loss minus time-extended loss, so positive means adding time helps. Every mean reduction here is negative: time slightly worsens average held-out prediction in the tested models. These tiny differences are not a substantial degradation claim.

For held-out prompts, all ten per-split intervals for the two timing additions include zero. Held-out annotators show mixed signs and some intervals excluding zero in opposite directions; there is no consistently transferable benefit. The normalized timing score equals the raw score for unseen annotators because subtracting the same global fallback cannot change rankings. The identity-based baseline transfers poorly to unseen people; its advantage on held-out prompts is not a general cold-start solution.

### What was established, and what was not

- **Established:** the public release lacks the event boundaries needed for the proposed causal explanation; raw timing weakly predicts stated strength; neither tested timing augmentation provides a consistent held-out improvement over ordinary metadata controls.
- **Not established:** timing is universally useless; all nonlinear or hierarchical estimators fail; the arbitrary-delay construction describes real annotators; or no timing protocol could improve an alignment policy. AUROC and log loss on stated labels cannot establish downstream generation benefit or interpersonal cardinal utility.
- **Not run:** a full ResponseRank reward-model reproduction, hierarchical DDM, orthogonal timing loss, new human study, or policy training. They are not required to decide against this particular investment: the novelty and measurement gates already fail. Their absence limits any broader claim about timing methods.

### Why we stop here

The original follow-up plan was a human measurement intervention followed by a theory/method campaign. It is **not an active work plan** after this audit. Separating choice-click time from later annotation is an obvious instrumentation control, and existing normalization/calibration methods remain plausible alternatives. We cannot justify assuming that a new robust optimizer is needed before these controls have been shown insufficient.

A credible future reconsideration would need independent evidence of a consequential decision error that survives these remedies, a feasible measurement protocol, and a distinct theorem beyond known decision identification. Those conditions describe what would change the decision; they are not a request to fund further work now. No GPU run or participant recruitment is recommended for the current formulation.

## 6. Other directions eliminated during this search

These are bounded screens, not declarations that their whole fields are solved. Worker findings were synthesized in the same model family, not independently peer reviewed. Full source scope varies; rows with only targeted passages must not be read as full-paper audits.

| Candidate | Closest source or ordinary escape | Decision |
|---|---|---|
| More diverse or causally realistic user simulators | [SCOPE](https://arxiv.org/html/2608.12253), [Controllable User Simulation](https://arxiv.org/html/2605.11519); method/theory and relevant appendices examined by a worker | Generic proposal occupied; no distinct residual established. |
| Pair stochastic agent rollouts to separate luck from skill | [Luck Is Not Skill](https://arxiv.org/html/2609.24144); variance analysis and experiments examined by a worker | Direct collision; reward-contrast variance is not automatically gradient variance or learning benefit. |
| Align training with constrained decoding | [Constrained code generation](https://arxiv.org/html/2606.21619), [grammar jailbreaks and CodeShield](https://arxiv.org/html/2606.11817); targeted method/evaluation worker screen | Generic decoder-aware alignment is occupied. |
| Repair dense-reward GRPO collapse | [Dark Room](https://arxiv.org/html/2607.21273v1), §§3–5 read directly | Removing std normalization is already a decisive ordinary remedy in the reported setting. |
| Fix sequence/token policy-composition mismatch | [PoEM](https://arxiv.org/html/2609.30226v1), targeted composition passages; MOD worker check | Mismatch is acknowledged and correction literature exists. No new contribution established. |
| Preserve persistent user identity rather than only marginal reward distributions | [2024 RLC primary paper, indexed relevant section](https://openreview.net/pdf?id=UKlQyRAKPv), [MaxMin-RLHF](https://arxiv.org/abs/2402.08925) | Persistent-identity criticism is already explicit in prior work. Direct PDF access hit an OpenReview challenge; no public review discussion was newly read. |
| Make DuPO masks reliable across candidate shifts | Original DuPO method/selection ablation and [Condition Mask Verification](https://aclanthology.org/2023.findings-emnlp.167.pdf), worker audit | Exact sound inversion cannot accept a wrong answer at the original missing input. Imperfect inversion needs controls against stronger inverse models and all-mask aggregation; no residual demonstrated. |
| Improve protein local stability from unordered assay profiles | [Nature 2026 energy landscapes](https://www.nature.com/articles/s41586-026-10465-z), worker full relevant-method/results inspection | Prediction gaps are real, but a permutation-invariant design objective does not need residue assignment. No evidence that the proposed assignment mechanism is necessary for the practical decision. |

## 7. Investment decision

**NO-GO: retire anonymous timing under arbitrary correlated measurement heterogeneity as our next main project.**

| Investment criterion | Evidence from self-review | Verdict |
|---|---|---|
| Real problem | ResponseRank documents a real timing weakness; the CPU audit finds no consistent incremental benefit in its stated-strength prediction task. | A weakness exists, but our proposed mechanism is unverified. |
| Distinct contribution | Closest timing/decision-identification work substantially overlaps the framing; robust losses and calibration already exist. | Insufficient novelty demonstrated. |
| Theory strengthens the method | Our indistinguishable-world example rules out an assured beneficial update under unrestricted nuisance. | Useful negative constraint, not a useful new learning method. |
| Nontrivial practical necessity | The data cannot isolate decision time; simple instrumentation and existing estimators have not been ruled out. | Necessity not established. |
| Cost of resolving uncertainty | Would require new measurement evidence and independent utility evaluation before a defensible policy experiment. | Current evidence does not justify that investment. |

This is a decision about the current formulation, not a proof that the whole response-time field is solved. The audit and theoretical note remain reusable screening evidence. We should not extend training or broaden an adversarial theorem merely to preserve this candidate.
