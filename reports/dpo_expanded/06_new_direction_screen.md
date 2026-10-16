# New direction screen: when can implicit feedback justify a policy update?

**Date:** 16 October 2026. **Status:** one candidate for a bounded falsification study; **no new main project has passed all gates**. This report replaces neither the original literature analysis nor the rejection ledger. The previous conditional-design direction remains a theoretical note, not the recommended main study.

## 1. Candidate and practical decision

**Research question:** Under what observable conditions can anonymous choice-and-time feedback certify an improvement in a language model's preference objective despite heterogeneous decision processes?

The practical decision is whether an alignment pipeline should use recorded annotation times to change its training targets, retain choice-only supervision, or obtain additional explicit feedback. Timing is attractive because collecting it can be inexpensive. Its value depends on whether it adds reliable preference information, not merely another predictor correlated with labels.

The proposed contribution would concern **identifying a beneficial decision under measurement uncertainty**. Neither collecting response times, fitting a drift-diffusion model, adding annotator effects, nor applying a generic robust objective would be a new contribution.

**Why this is only a candidate:** a real NLP timing failure is documented, and the 2026 theory explicitly leaves heterogeneous measurement processes unresolved. However, the failure has not been attributed to our proposed mechanism. A method that reliably improves deployed generation has not been established. We must not infer that missing theorem conditions explain an observed experimental failure.

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

**Additional calibration collisions:** [Factoring Out Nondecision Time](https://www.ppw.kuleuven.be/okp/_pdf/Verdonck2016FONDT.pdf), 2016, already uses convolution across experimental conditions to remove a shared nuisance distribution. A [revised 2026 eLife study](https://elifesciences.org/reviewed-preprints/109730v2) uses external non-decision-time information and discusses earlier calibration alternatives. Relevant sections were inspected by a worker. These are mandatory comparisons where their shared-nuisance/repeated-condition assumptions apply.

### Interpretation

There is a meaningful tension between an identification result for a specified measurement process and disappointing timing-based learning in actual text annotation. They are **not contradictory findings**: their observation protocols and targets differ. A useful study must determine whether resolving that difference changes the quality of alignment decisions after strong existing remedies.

The main opportunity is conditional: characterize which remaining uncertainty matters for a policy decision, and determine whether a small amount of additional evidence can resolve it. Merely obtaining a better psychological parameter fit would not satisfy the practical goal.

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

**Search-bounded remaining question:** anonymous population-level **policy identification and improvement guarantees** under correlated measurement heterogeneity, together with a demonstrably useful observation protocol. No inspected source provided that entire package. This is weaker than establishing priority, and classical partial-identification, decision-theory and mixture-identification literature still require a dedicated audit.

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

This robust objective is **standard scaffolding**, not our new method. A full result would need an observable finite-sample uncertainty set, useful rather than vacuous guarantees, and upper/lower bounds for the actual permitted measurements. Parameter nonidentification need not prevent decision identification; that distinction is the target.

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

The numeric identities were checked with Python's standard library. They are neither simulated participant results nor language-model results. No empirical failure has been reproduced in this search.

## 5. Bounded falsification study before theory investment

### Gate A: verify the residual on existing human annotations

Use [MultiPref](https://huggingface.co/datasets/allenai/multipref) and the [released ResponseRank implementation/results](https://github.com/timokau/response-rank). Confirm released timing fields, measurement boundaries, annotation order and available IDs before interpreting time as decision time.

1. Reproduce the **data-level** relation between time, stated strength, disagreement and text properties; inspect held-out annotators as well as held-out prompts. Avoid counting each rater's opinion as an independent prompt.
2. Compare raw timing, per-annotator normalization, annotator-by-length strata, flexible observed-context adjustment, existing orthogonal losses and an appropriate hierarchical DDM. Preserve choice-only, explicit-strength and agreement baselines.
3. Diagnose whether residual timing variation provides information beyond those baselines. Held-out log loss and preference accuracy measure prediction, not true cardinal utility.
4. Evaluate response selection where matched human judgments actually support it. Do not invent rewards for unobserved alternative responses or reinterpret observational annotations as randomized policy evaluations.

**Immediate stop:** no reliable residual timing information; standard adjustment recovers it; agreement/extra labels dominate at matched resource cost; or the released data cannot support the necessary target. MultiPref failure alone is not a reason to construct an increasingly elaborate timing estimator.

### Gate B: establish a measurement intervention if Gate A survives

If existing data cannot isolate the process, a new, appropriately authorized human study is necessary. Prepare a small within-participant protocol that distinguishes reading, choice and additional annotation time. Randomize layout/order and use matched content; separately check whether the intervention changes the preference itself. A layout-induced behavioral change cannot automatically be attributed to pure measurement noise.

Compare a simple choice interface, the original multi-field protocol, and a conventional calibration baseline. Evaluate target judgments independently from the signal used to fit the model. Permanent identity is not necessary for a temporary session experiment, but repeated observations must be counted explicitly.

**First and cheapest escape to test:** record the choice-click timestamp separately from subsequent justification and auxiliary labels. The MultiPref duration used in the cited analysis includes those later tasks. If isolating the event repairs timing's usefulness, that is a measurement fix; it would not justify our more elaborate proposal. Agreement also targets consensus rather than necessarily the same cardinal population mean, so compare practical decisions and state any target mismatch.

**This human study has not been run or authorized as participant recruitment.** Existing public-data analysis can proceed without it. No participant contact, external messages or sensitive-data collection occurred in this search.

### Gate C: only then develop a full theory/method package

The potential package is: an empirically defensible nuisance class; a characterization of decision identification within that class; finite-sample bounds; a matching or near-matching observation strategy; and downstream gains against the strongest repairs. A lower bound over arbitrary delays plus a generic maximin objective would be insufficient.

Practical success must mean better independent preference decisions or generation quality at comparable total annotation resources. A better fitted RT distribution, lower training loss or a new confidence score is not enough. There is no justified numeric improvement expectation yet.

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

## 7. Recommendation

The timing/measurement candidate has a concrete source-documented failure after a relevant simple repair and a mathematical connection to our previous information findings. It merits **a bounded data audit**, not another expensive training run or a declared main proposal. Its distinctive contribution, practical mechanism and independent priority remain unresolved.

The success condition for this search is not to rename another standard remedy. If the candidate fails its baseline or measurement gate, record that result here and continue to a different problem. Do not use the illustrative impossibility example to keep it alive.
