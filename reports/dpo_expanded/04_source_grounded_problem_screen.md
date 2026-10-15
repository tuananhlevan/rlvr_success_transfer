# Research problems grounded in the papers' actual methods and limitations

**15 October 2026. Status updated after user review: generator-routing proposal rejected as a main paper direction. No new method or full-paper proposal validated.**

**Reason for rejection:** the proposed mechanism is expected-learning-progress prediction plus cost-aware routing, using established acquisition and bandit tools. We have not identified a consequential failure of ordinary methods, a necessary new mechanism, or a substantive theoretical separation. An explicit future-work statement and a large compute bill establish relevance, not research necessity. The screening notes and formulas below are retained as historical analysis.

## 1. Revised selection rule

A retained problem must have an exact source location or a reproducible derivation from an actual implementation. A missing theorem, an incorrect formula, a hypothetical adversary, or the authors' suggestion of future work is insufficient on its own.

For each problem, distinguish:

1. What the source actually reports or assumes.
2. What we infer from its engineering setup.
3. What existing remedies already address.
4. What would have to fail on real data before the problem justifies a new paper.

The earlier fusion, verifier-transfer mechanism and informative-delay method proposals remain rejected or deprioritized. This screen does not revive their constructed examples as evidence of an unmet practical need.

**Read scope:** fresh focused reads of relevant methods, limitations and appendices, plus current primary related work. Public-review statements use the [discussion audit already read through the website](03_openreview_discussion_audit.md); they were not newly retrieved in this pass. A fresh OpenReview supplement request reached its verification page; the official ICLR supplement was accessible. No OpenReview API was used. Native research workers shared this session; their criticism is not external independent peer review.

## 2. Screened problem: generation before preference-pair selection

### Source evidence, with exact locators

[ActiveUltraFeedback v2](https://arxiv.org/html/2603.09692v2), §4.1, §5.3, §6, Appendix F.4/Table 22 and G.6/Table 26, uses a 30-model response pool before pair acquisition. It estimates 600 GPU-hours for generation, 600 for precomputed annotation and 32 for the active loop. Annotation precomputation supports retrospective experiments; deployment only needs selected annotations. Section 6 explicitly identifies choosing generators before producing responses as a priority. G.6 already tests smaller pools: DPO mean improvement is .112 for all 30 models, .017 for the 15 largest, and .090 for five large plus five small models. These are reported point estimates, not our replication or a significant-difference claim.

**Precise problem:**

> Can a preference-data pipeline choose which generators to query, before seeing their responses, and achieve better downstream learning per total generation, feature-scoring and annotation cost than a well-chosen fixed diverse pool?

This is directly anchored in a stated engineering limitation. Its theoretical counterpart is supported by the supplied Coverage paper, discussed in Section 3.

### Intuition

Selecting two useful responses from thirty completed responses can save annotation cost while still paying for all thirty generations. Moving selection earlier changes the available information: the selector knows the prompt and generator identities, but cannot inspect response quality, disagreement or response embeddings until it pays to generate them.

This creates a decision about which information to acquire. Merely substituting a cheaper acquisition formula after all responses exist does not change that decision.

### What we derive from the setup

Let $\mathcal G$ be a generator collection. Querying generator $g$ for prompt $x$ returns a random response $Y\sim P_g(\cdot\mid x)$ and incurs generation cost. The history $\mathcal H_t$ contains only previously queried responses and observed feedback. A valid next-query policy must be measurable with respect to the prompt, allowed metadata and $\mathcal H_t$.

Its resource budget should include

$$
C=\sum_{\text{queried }(x,g)}c_{\rm gen}(x,g,Y)
+\sum_{\text{scored responses}}c_{\rm feat}(x,Y)
+\sum_{\text{annotated pairs}}c_{\rm ann}(x,Y,Y').
$$

This is our proposed accounting contract, not a theorem from the source. If output length makes realized cost unknown before generation, the policy must use a prediction or a stopping rule rather than observing future cost for free.

The quantity to optimize must be defined separately: final policy quality, reward-model risk or information relevant to a specified target. A large response-quality gap is a surrogate; obtaining the best single answer is another surrogate. Neither is automatically the downstream learning objective. We have not proved that a sophisticated alternative is necessary.

### Existing remedies and the hard baseline gate

The following are mandatory comparators:

- The source's smaller diverse fixed pool, not merely its 30-model pool.
- Its cheap fixed small/large-model pairing baseline, with appropriate task-specific evaluation.
- A fixed generator pair chosen on a separate validation split.
- Uniform generator sampling at the same total budget.
- A prompt-conditioned quality/cost router and ordinary contextual-bandit generator selection.
- The source's response-level DRTS/DeltaUCB after generating a budget-matched pool.

ActiveUltraFeedback already studies acquisition objectives and pool reductions. Adding a cost term to UCB, taking fewer samples, or querying a large and a small model is not sufficient novelty. An adaptive method must improve a measured cost–quality frontier beyond those remedies.

### Why source provenance did not establish research necessity

The source explicitly identifies the unresolved stage, supplies a cost breakdown and releases useful artifacts. A fixed strongest-model pool is not an adequate sole control: its own ablation performs poorly. However, the diverse ten-model pool is competitive. We have not shown that this strong ordinary remedy leaves a substantial failure.

**Current decision:** rejected as a main paper direction. A cost comparison could be a useful engineering experiment, but the present idea does not justify a substantial theoretical research program.

### Available next test

Inspect the released artifacts to determine whether full per-prompt candidate pools, generator identities, lengths and scores are available. A chosen/rejected-only dataset cannot evaluate unselected generators fairly.

If full pools exist, run a causal acquisition replay: hide each response and its score until its generator is queried, hide unqueried response embeddings, separate train/validation/test prompts and include selection overhead. Existing generated outputs can support this screen, but they do not measure new deployment GPU throughput or a changing generator distribution. Compute proxies must be labeled as proxies.

Evaluate both a preference-trained policy target and reward-model prediction. If cheap fixed diverse pools or ordinary contextual-bandit routing achieve the same frontier, reject a new-method proposal. Only after a surviving failure should we decide which theory is needed: generator-access sample complexity, exploration under missing response features, or policy-targeted acquisition with a counted oracle. A synthetic pair-selection example is not a substitute for this test.

## 3. Supporting theoretical problem: a sampler whose acquisition oracle is costly

**Source:** [Coverage Improvement and Fast Convergence of On-policy Preference Learning v1](https://arxiv.org/html/2601.08421v1), §2.1, §4, Algorithm 2, Remark 4.3 and Appendix C.2. Algorithm 2 uses known fixed features, conditional designs for all prompts and a global design with a freely chosen prompt marginal. Remark 4.3 already permits efficient approximate convex/Frank–Wolfe optimization, but explicitly leaves avoiding all-prompt conditional preprocessing open. Its near-optimal two-round statement does not provide a counted language-generation/feature-query oracle.

**Our engineering derivation:** producing an unseen response, computing its features and finding a high-leverage response in an implicit language-model output space are resources. Counting design optimization iterations without charging those operations may misrepresent an implementable alignment budget.

**Precise question:** can comparable preference-learning efficiency be obtained with only paid response/feature queries, under the prompt access available to the actual pipeline?

This question must distinguish free prompt allocation from fixed incoming prompts. Importance weights cannot create an unavailable prompt or response, and may increase estimator variance. However, this observation alone is elementary and is not our proposed contribution.

**Obvious repairs:** Frank–Wolfe, cached information matrices, candidate subsampling and classical design on response-pair contrasts. A global pair-design construction may bypass the source's per-prompt preprocessing; do not promote removal of that loop as new without checking the classical design argument.

**Close related work:** [ADPO](https://arxiv.org/html/2503.01076v1), §§4.1/6.2; [NAOD](https://arxiv.org/html/2609.38860v1), §§4.3–4.4/Appendix C; and [GSHF](https://weixiongust.github.io/WeiXiongUST/gshf.pdf). Finite-pool design and policy-based exploration are strong existing approaches. They must be checked under the same charged acquisition contract, not dismissed because they use different notation.

**Decision:** use this as theoretical motivation for the concrete generation-stage problem in Section 2. A faster design solver or a dimension-factor improvement alone does not establish an impactful NLP research direction.

## 4. A real reported failure that should be distinguished from our failed explanation

**Source:** [Proof-Carrying Cognition v1](https://arxiv.org/html/2609.09776v1), §§8–8.1. The paper reports an on-policy label-efficiency failure on real code. Its equal-budget half-random mixture gives .148 AUC versus .147 selected-only and .170 random at 720 labels. Its competence and mixture interventions fail to explain the discrepancy, and cross-problem transfer is explicitly left open. A feature-based head also fails to improve selection across problems. These are source-reported, small-suite results; we have not reproduced them or independently validated their causal conclusions.

**Precisely grounded question:** why does selecting code candidates for correctness audits fail to improve cross-problem selection efficiency in that experiment, and which established audit/fitting recipe resolves it?

The source's failed mixture experiment rules out only that tested mixture as a sufficient repair. It does not rule out importance weighting, richer representations, different regularization or pairwise fitting.

Our earlier executable construction showed that standard methods repair our proposed task-offset mechanism. That rejected explanation is not evidence that the real source result has been resolved. Conversely, an unexplained result in one preprint does not establish that all standard remedies fail.

**Decision:** retain as a source-documented reproduction problem. It is not a new direction or justification for reviving our earlier method proposal. A reproduction with strong existing fitting/acquisition baselines would be necessary before pursuing new theory.

## 5. Other source-grounded problems screened and rejected as proposals

| Source location | Genuine issue or derived engineering question | Existing remedy / collision | Disposition |
|---|---|---|---|
| AuxDPO §4.1–4.2, Proposition 9; [author scope reply](https://openreview.net/forum?id=btEiAfnLsX&noteId=kAKQcHkdED); [official supplement](https://proceedings.iclr.cc/paper_files/paper/2026/file/8f6c674a7edf57912f1a5830f4a66e3f-Supplemental-Conference.pdf), B.2 | Can the fixed-reference nuisance geometry still target the intended policy component outside its local neighborhood? Batchwise constraints also differ from population constraints. | Relinearization, appropriately weighted/unbiased moment estimation and reward fitting followed by RLHF are strong ordinary remedies. | Legitimate audit; no demonstrated residual practical failure. Do not promote a global-proof extension alone. |
| CATTO v2 §4.1, §5 and Limitations p.9 | Whole-answer correctness confidence does not follow from next-label/token calibration. | [CALM](https://arxiv.org/html/2608.07419v1) trains free-form calibration; [Sem-ECE](https://arxiv.org/abs/2605.08432) evaluates semantic confidence. Answer-level proper-loss calibration is an immediate baseline. | Reject generic extension to open-ended calibration. |
| Domain-shift study v2 §§3.3, 6–7 and Limitations | Target teacher preferences can carry teacher style as well as task knowledge. | The source reports online methods with better diversity; [DivPO](https://arxiv.org/html/2501.18101v2) already targets quality–diversity selection. | Reject a generic diversity weight or teacher mixture. |
| MixDPO Eq.3 and Limitations p.9 | Positive sensitivity mixtures cannot encode conflicting preference directions. | Personalized reward models and [representation/aggregation theory](https://arxiv.org/html/2606.00291v1). | Explicit limitation, but broad repair is occupied. |
| AW-DPO §4, Eq.4; [weighting-theory reply](https://openreview.net/forum?id=OuMNJoKJBQ&noteId=WhIqcfumRT) | Independently scored segments need not supply valid nonnegative normalized weights. | Simple weight constraints; [SaLT-DPO](https://arxiv.org/html/2609.15517v1). [Seg-OPD](https://arxiv.org/html/2610.02703v1) also tests student continuations after segment revisions. | Reject basic signed-weight repair or generic counterfactual-prefix proposal. |
| ECLIPTICA Appendix E.3 and §6.4 | Absolute refusal separation does not establish correct conditional safety. | The paper discusses a signed/harm-aware diagnostic; [MOSAIC](https://arxiv.org/html/2603.16210v1) addresses selective composition and benign preservation. | Reject metric correction as a paper direction. |
| DuPO §3.1–3.3, Appendix A/E; public-review audit | Finite candidate-pool inverse uniqueness does not establish universal correctness; forward/inverse competence differs. | Frozen/independent verifiers, complementary-task selection and existing verification/audit methods are immediate comparators. | The issue is sourced, but generic consistency-versus-truth arguments do not establish a new contribution. |
| RLHF generalization v1 §5.2, Assumptions 4–5 | Raw-feature coverage can include a policy-invariant direction, producing an incompatible coverage requirement. | Center identifiable response contrasts and apply established covariance analysis. | Retain a mathematical audit finding; reject as a standalone research direction. |

No numeric novelty or publication-confidence scores are assigned. An author's future-work statement establishes provenance; it does not certify that the problem is open throughout the literature.

## 6. Decision after user review

**Selecting generators before responses exist** is explicitly named in a 2026 paper and supported by a concrete engineering budget. That satisfied the provenance requirement, but it did not establish that the proposed solution was necessary or scientifically substantial.

**Reject the current routing proposal.** No extended experiment or theorem program is recommended for it on the present evidence. Reconsideration would require a consequential source-backed failure that established methods cannot adequately resolve, or a substantive new characterization with practical implications. The generic possibility of reducing generation cost is insufficient.

This screen executed literature and engineering analysis. It did not run a new language-model experiment, acquire a complete candidate bank, or demonstrate that adaptive generator selection is necessary.

## 7. Concrete method idea: select generators by expected learning progress

**Status:** rejected proposal, retained to document the reasoning. Expected learning progress, gradient alignment and cost-aware acquisition do not establish a necessary new method in the current setting.

### Central hypothesis

A generator can produce excellent answers while producing comparisons that teach the current learner little. Conversely, a less expensive specialist may expose exactly the learner's remaining mistake. We should test whether predicting this difference before generation provides a useful advantage over a fixed diverse pool or ordinary quality routing.

The intended method selects a generator pair using the prompt, learner state and previously acquired examples. It does not inspect the current prompt's ungenerated responses, their embeddings, lengths or judge scores.

### Objective

Let $L_{\rm val}(\theta)$ be a held-out target loss. For generated responses from generators $g,h$ and an observed preference label $Z$, let $v$ be the gradient of the training loss for that pair. A conceptual pair utility is

$$
U_t(x,g,h)=\mathbb E\left[
L_{\rm val}(\theta_t)-L_{\rm val}(\theta_t-\eta v)
\mid x,g,h,\mathcal H_t\right].
$$

The expectation covers unseen responses and feedback. This one-step utility is not the final multi-step training benefit. It is an estimable local objective whose shortcomings must be measured.

For a twice-differentiable validation loss, its local approximation is

$$
U_t(x,g,h)\approx
\eta\nabla L_{\rm val}(\theta_t)^\top\mathbb E[v]
-\frac{\eta^2}{2}\mathbb E\left[v^\top H_{\rm val}(\theta_t)v\right].
$$

The approximation comes from ordinary Taylor expansion. Gradient alignment, influence approximations and cost-sensitive acquisition are established tools; none is claimed as our innovation.

For DPO, a chosen/rejected pair with reference-relative margin $z$ has gradient

$$
v=-\beta\sigma(-z)
\left[\nabla\log\pi_\theta(y_w\mid x)
-\nabla\log\pi_\theta(y_l\mid x)\right].
$$

A large teacher-assessed quality gap does not ensure a large useful update: when the learner already has $z\gg0$, the sigmoid factor is small. A large gradient can also point away from the held-out target. These identities motivate a learner-dependent utility; they do not prove superiority over existing acquisition scores.

### Implementation to test

1. **Collect a charged pilot.** Query a small, diverse set of generators on training prompts, with randomized coverage and recorded selection probabilities. Count generation, scoring and annotation costs.
2. **Measure observed pair utility.** After obtaining a pair and feedback, estimate its update alignment on separate validation minibatches. Use adapter-gradient sketches if full gradients are too costly; count sketch computation. Occasionally compare the surrogate against actual short training updates.
3. **Fit a cheap predictor.** Predict the distribution of pair utility from prompt features, generator identities and summaries of the learner's current errors. Train only on acquired outputs. Track uncertainty and refresh as the learner changes.
4. **Select the next queries.** Allocate a batch under the total budget using predicted positive learning benefit and uncertainty. Include exploration and coverage controls. A benefit-per-expected-cost score is a baseline implementation, not a novelty claim.
5. **Test a staged variant.** When an inexpensive first response helps predict which second generator would provide a useful contrast, acquire it first and condition the second query on that paid observation. Compare against simultaneous pair selection and ordinary correction-teacher routing. Any paid first response belongs in the cost ledger.

A simple example is a learner with remaining arithmetic mistakes but strong general prose. An expensive general model might yield polished comparisons the learner already ranks correctly; a cheaper arithmetic specialist paired with the learner might supply useful corrective examples. This is an illustrative hypothesis, not an observed result. Cheap fixed specialist routing could achieve the same benefit and is a mandatory comparator.

### What would have to be new

The candidate contribution is a demonstrated improvement in acquiring useful **training comparisons before their response features are observed**, together with theory that charges learning the generator distributions and acquiring candidate outputs. Existing [NAOD](https://arxiv.org/html/2609.38860v1) is an important policy-targeted design baseline; response-level acquisition cannot be dismissed as irrelevant merely because its pool already exists.

General routing and teacher selection are occupied topics. [RouteLLM](https://arxiv.org/abs/2406.18665) routes inference requests using preference data. [BRTS](https://arxiv.org/abs/2605.09725) selects teacher rollouts for on-policy distillation. These are close comparisons; adding a predictor or calling it a router does not establish a contribution.

### Decisive test and rejection condition

At equal total acquisition and training cost, compare against the diverse ten-model pool, fixed validated generator pairs, fixed specialist routing, quality/cost routing, ordinary contextual-bandit routing and budget-matched DRTS/DeltaUCB. Ablate learner-state inputs, staged querying and uncertainty separately. Separate a frozen-learner acquisition replay from a real changing-learner training experiment.

Reject the proposal if those methods match its frontier, if its surrogate fails to predict useful updates, or if computing utility costs more than the generations it saves. No arbitrary compute-saving percentage or acceptance probability is assigned. The problem is source-documented; the necessity and novelty of this solution remain to be established.
