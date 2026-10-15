# Expanded literature analysis: 26 distinct preference-learning papers

**Research-direction update, 15 October:** the subsequent fusion and verifier studies failed their baseline gates. The [latest research screen](../../studies/informative_delays/research_report.md) records rejected alternatives and a preliminary theoretical lead. Earlier priority recommendations below are historical.

**Prepared:** 14 October 2026. **Input:** [the supplied spreadsheet](https://docs.google.com/spreadsheets/d/1vwuYzjRLaazjo5bUhDtF3rsvBE7CeKxykNyT8eMOW7w/edit?gid=1488308827). **Deliverables:** [theoretical proposals and sprint 0](02_theoretical_research_proposals.md), [public review audit](03_openreview_discussion_audit.md), [PDF inventory](../../resources/README.md), [reproducible analytical checks](verification/analytic_checks.py).

## 1. Scope and evidence

All 27 listed entries are available locally: eleven supplied PDFs reused and sixteen arXiv PDFs downloaded automatically through connected Chrome. AW-DPO appears in both tabs, giving 26 distinct papers and 27 PDF files. The [manifest](../../resources/download_manifest.json) records titles, source URLs, versions, page counts and SHA-256 hashes. SafeDPO's spreadsheet link is a 2025 submission despite its placement in the ICLR 2026 tab; the final 2026 paper and its discussion were checked separately. Do not infer publication from a sheet label.

Methods, relevant derivations, experiment definitions, limitations and relevant appendices were analyzed. Long repeated illustrations and qualitative examples in CATTO, ECLIPTICA, SPINAL and some engineering appendices were selectively inspected. This is a deep mechanism review of a supplied corpus, not an exhaustive systematic search or a reproduction of training. Public discussions were inspected for ten of the eleven original papers; ProtAlign's discussion remains unchecked because Chrome was disconnected on resumption. Read scope and specific note links are in the review audit.

Source claims, our deductions and proposed theorems have different evidence status. Formula objections below concern the expressions printed in the inspected version; they do not establish which objective released or unpublished code actually ran. Relevant sections/equations are used as locators; no certified PDF page anchors are claimed. No language-model training or human annotation was performed.

## 2. Paper map, with intuition

The last column identifies the deeper assumption or mechanism that should influence research design. Detailed equation-level audits follow the synthesis.

| Paper / source                                                                                                                                         | Local PDF                                                | Intuition explanation                                                                            | Main idea and contribution                                                | Deep insight / weakness                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [Property-Driven Protein Inverse Folding with Multi-Objective Preference Alignment](https://openreview.net/forum?id=m826DekCpp)                         | [PDF](../../resources/4249_Property_Driven_Protein_I.pdf) | Separate competing objectives before treating a pair as evidence for one property.               | Scalarized preference margins and shared-randomness likelihood estimates. | Auxiliary reward gaps explain part of a policy log-ratio; proxy outcomes do not establish measured outcomes.            |
| [Why DPO is a Misspecified Estimator and How to Fix It](https://openreview.net/forum?id=btEiAfnLsX)                                                     | [PDF](../../resources/24413_Why_DPO_is_a_Misspecifie.pdf) | A limited model can fit preferences well while moving toward the wrong reward compromise.        | Misspecified DPO projection; local nullspace reward offsets.              | The local policy uses only a projection of reward; comparisons also fit irrelevant residual directions.                 |
| [Token-Importance Guided Direct Preference Optimization](https://openreview.net/forum?id=cMEnMVvMw9)                                                    | [PDF](../../resources/6576_Token_Importance_Guided_D.pdf) | Concentrate training on decisive tokens, but blend noisy salience with a position prior.         | Logit-gradient token weights plus an anchor triplet loss.                 | Logit sensitivity is not preference importance; variance reduction alone does not order logistic risk.                  |
| [Mitigating Mismatch within Reference-based Preference Optimization](https://openreview.net/forum?id=k79Un1LSXy)                                        | [PDF](../../resources/362_Mitigating_Mismatch_within.pdf) | Relative improvement can still leave the bad answer more likely.                                 | Clip pessimistic reference pair margins.                                  | Clipped edges can lose global potential consistency; stronger pair pressure is not an objective guarantee.              |
| [DuPO: Enabling Reliable Self-Verification via Dual Preference Optimization](https://openreview.net/forum?id=SD8Z231C45)                                | [PDF](../../resources/6741_DuPO_Enabling_Reliable_Se.pdf) | A partial problem plus its answer may let the model reconstruct the missing part.                | Conditional inverse-task feedback; GRPO implementation.                   | Pool uniqueness reduces ambiguity but does not establish verifier soundness against coherent errors.                    |
| [Importance Sampling for Multi-Negative Multimodal Direct Preference Optimization](https://openreview.net/forum?id=HEFPwoGtTj)                          | [PDF](../../resources/24535_Importance_Sampling_for_.pdf) | Several distinct wrong images expose more types of confusion than one negative.                  | Multi-negative visual loss, SAE selection and importance sampling.        | Normalized/logarithmic estimators and greedy top-k selection require distinct bias analyses.                            |
| [SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](https://openreview.net/forum?id=MoJSnVZ59d)                         | [PDF](../../resources/5041_SafeDPO_A_Simple_Approach.pdf) | An absolute safety label says whether either response is acceptable, beyond ranking them.        | Safety-oriented pairing and shifted preference margins.                   | Hard-support ideal and finite softmax learning differ; 2025 draft and 2026 final are distinct.                          |
| [Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](https://openreview.net/forum?id=G7DBGlgjjp)                                   | [PDF](../../resources/11934_Uni_DPO_A_Unified_Paradi.pdf) | Prioritize reliable unfinished pairs while preventing good answers from losing likelihood.       | Quality, focal pressure and conditional chosen-only likelihood.           | Differentiating a dynamic weight changes the gradient; scalar scores supply extra information.                          |
| [P^2 -DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization](https://openreview.net/forum?id=ekOwxTn65Y) | [PDF](../../resources/8851_P_2_DPO_Grounding_Halluci.pdf) | Improve use of visual evidence by contrasting enhanced and erased contexts.                      | Focus/robustness preferences and evidence-gain calibration.               | Data are generated once from the initial model; evidence dependence need not be factual correctness.                    |
| [Alignment-Weighted DPO: A principled reasoning approach to improve safety alignment](https://openreview.net/forum?id=OuMNJoKJBQ)                       | [PDF](../../resources/15113_Alignment_Weighted_DPO_A.pdf) | A safe rationale should not compensate for an unsafe final answer.                               | Separate reasoning and response preference losses.                        | Segment consistency requires calibrated nonnegative weights; safety intervention is not a universal causal explanation. |
| [Is On-Policy Data always the Best Choice for Direct Preference Optimization-Based LM Alignment?](https://openreview.net/forum?id=tz9mJmgrdM)           | [PDF](../../resources/24230_Is_On_Policy_Data_always.pdf) | The most useful source depends on what the learner already knows.                                | A relative alignment-stage diagnostic for on/off-policy source selection. | Rank agreement is not distribution identity; cross-source quality does not isolate diversity or update value.           |
| [Coverage Improvement and Fast Convergence of On-policy Preference Learning](https://arxiv.org/pdf/2601.08421)                                          | [PDF](../../resources/2601.08421v1.pdf)                   | Fresh sampling can uncover the feature directions needed near the target.                        | Feature-covariance coverage and iterative/two-round sample bounds.        | Realizability, sufficiently large batches and a coverage-growth condition drive convergence.                            |
| [Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning](https://arxiv.org/abs/2601.03027)                                            | [PDF](../../resources/2601.03027v3.pdf)                   | Repair factually reversed preferences, then require a larger margin for factual winners.         | Factuality-aware pair flipping and hallucination-gap margin.              | Verifier errors reverse update direction; label repair sometimes beats the extra margin.                                |
| [MixDPO: Modeling Preference Strength for Pluralistic Alignment](https://arxiv.org/pdf/2601.06180)                                                      | [PDF](../../resources/2601.06180v1.pdf)                   | People can agree on quality but differ in how sharply they discriminate it.                      | Learn a mixture of preference sensitivities inside the likelihood.        | Positive sensitivity mixtures model strength, not conflicting value directions; joint scale is unidentified.            |
| [From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models](https://arxiv.org/pdf/2601.06108)           | [PDF](../../resources/2601.06108v1.pdf)                   | Many direct losses approximate or reshape the same KL-regularized reward problem.                | Theoretical organization of RLHF/direct alignment objectives.             | Shared ideal variational roots do not imply equal restricted minimizers or finite optimization paths.                   |
| [An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift](https://arxiv.org/pdf/2601.05882)                             | [PDF](../../resources/2601.05882v2.pdf)                   | Learning the target domain can matter more than choosing another loss.                           | Factorial transfer/diversity study and target teacher pseudo-labeling.    | Teacher quality, domain relevance and style interact; lower diversity and better scores can coexist.                    |
| [SPINAL - Scaling-law and Preference Integration in Neural Alignment Layers](https://arxiv.org/pdf/2601.06238)                                          | [PDF](../../resources/2601.06238v1.pdf)                   | Monitor where alignment changes activation spectra and predictive distributions.                 | Spectral tails, effective dimension and Fisher–Rao layer diagnostics.    | Exponent/protocol inconsistencies and an invariant negative control limit causal conclusions.                           |
| [Direct Preference Optimization with Rating Information: Practical Algorithms and Provable Gains](https://arxiv.org/pdf/2602.00603)                     | [PDF](../../resources/2602.00603v1.pdf)                   | A near-certain win cannot distinguish a modest gap from an enormous one; calibrated ratings can. | RDPO/RIPO and combined ranking–rating likelihood, finite-class bounds.   | Cardinal units, channel dependence and oracle trust matter; sparse ratings are already addressed.                       |
| [Tangent Space Fine-Tuning for Directional Preference Alignment in Large Language Models](https://arxiv.org/pdf/2602.01128)                             | [PDF](../../resources/2602.01128v1.pdf)                   | Make network updates additive to improve control when combining preferences.                     | Train and combine frozen-base JVP directions.                             | Linear logits do not imply linear sequence margins, behavioral tradeoffs or reward sufficiency.                         |
| [YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation](https://arxiv.org/abs/2601.08441)                                            | [PDF](../../resources/2601.08441v1.pdf)                   | Steer through a learned concept dictionary while preserving reconstruction residuals.            | Frozen-SAE latent intervention and bidirectional preference training.     | Sparse activations do not guarantee a sparse steering vector; ReLU can make directions asymmetric.                      |
| [SP^2DPO: An LLM-assisted Semantic Per-Pair DPO Generalization](https://arxiv.org/pdf/2601.22385)                                                       | [PDF](../../resources/2601.22385v1.pdf)                   | Assign each pair its own saturation curve based on teacher-estimated strength.                   | Offline semantic per-pair temperature from teacher ensembles.             | Larger temperature does not monotonically increase positive-margin pressure; edgewise scales may not integrate.         |
| [ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection](https://arxiv.org/pdf/2601.09195)                                 | [PDF](../../resources/2601.09195v3.pdf)                   | Avoid forcing imitation of unlikely surface wording that may erase useful pretrained behavior.   | Detached high-probability token selection in SFT.                         | Rare but crucial knowledge can be masked; gradient scale and token count are confounds.                                 |
| [CATTO: Balancing Preferences and Confidence in Language Models](https://arxiv.org/pdf/2601.23096)                                                      | [PDF](../../resources/2601.23096v2.pdf)                   | Train answer confidence alongside preference so confident selection becomes more useful.         | Preference objective plus confidence-alignment penalty.                   | The printed affine confidence loss is not proper; low aggregate ECE does not establish conditional correctness.         |
| [ECLIPTICA - A Framework for Switchable LLM Alignment via CITA - Contrastive Instruction-Tuned Alignment](https://arxiv.org/pdf/2601.06157)             | [PDF](../../resources/2601.06157v1.pdf)                   | Keep instructions as a common anchor while making preference controls switchable.                | CITA contrastive instruction alignment and modular control framework.     | The printed offline anchor differs from forward KL and can be unbounded below; code remains unverified.                 |
| [Clipping-Free Policy Optimization for Large Language Models](https://arxiv.org/pdf/2601.22801)                                                         | [PDF](../../resources/2601.22801v1.pdf)                   | Give an overshooting policy ratio a restoring gradient instead of a clipped plateau.             | Advantage-weighted quadratic ratio penalty.                               | A soft sampled penalty does not enforce global TV; stability does not resolve reward identification.                    |
| [Towards a Theoretical Understanding to the Generalization of RLHF](https://arxiv.org/pdf/2601.16403)                                                   | [PDF](../../resources/2601.16403v1.pdf)                   | Training corrects parameter error only in directions visible through response covariance.        | Known-reward log-linear stationary-point and generalization analysis.     | Realizable reward and strong unseen-feature coverage assumptions exclude general noisy neural RLHF.                     |

## 3. The common mathematical object

Let $\pi_0$ be a fixed reference, $r$ a reward and $\tau>0$ its cardinal KL tradeoff. The unrestricted optimum is

$$
\pi_r(a\mid x)=\frac{\pi_0(a\mid x)e^{r(x,a)/\tau}}{Z_r(x)}.
$$

Under a correctly scaled Bradley–Terry link, reward differences become reference-relative policy log-ratios. With pair margin $d_\theta$, ordinary DPO gives

$$
\ell=-\log\sigma(\tau d_\theta),\qquad
\nabla\ell=-\tau\sigma(-\tau d_\theta)\nabla d_\theta.
$$

This standard identity explains the original [DPO derivation](https://openreview.net/forum?id=53HUHMvQLQ). It does not guarantee that the same optimum belongs to a restricted policy class. Four independent decisions intervene: what reward the measurements identify; which policy class can represent it; which pair distribution weights its projection; and which finite optimization procedure follows that moving distribution. Most corpus methods change one of these decisions while holding the others implicit.

### 3.1 Pressure, direction and information are different contributions

HyPO, SafeDPO, F-DPO, RDPO and SP²DPO change margin pressure. Uni-DPO also differentiates through a dynamic weight and adds chosen-only likelihood. TI-DPO and AW-DPO change token/segment directions. MISP-DPO and P²-DPO change the available evidence directions. DuPO changes the feedback channel. MixDPO changes the statistical choice link. AuxDPO changes the representable reward fitted by the likelihood. Coverage theory changes the sampling law and controls its feature geometry. These interventions can all help, but they cannot be evaluated as interchangeable variants of a scalar sample weight.

A reliable factuality flag adds information; a larger margin applied to the same unreliable flag does not. A diverse negative set adds feature directions; a confidence score attached to existing pairs does not necessarily do so. Ratings can measure cardinal magnitude; ordinary ordinal scores do not automatically have those units.

### 3.2 Coverage and misspecification coexist

The coverage paper assumes the target Boltzmann policy is realizable. Under that contract, sampling near the target can improve covariance coverage and estimation. AuxDPO studies when the target cannot be represented and comparison likelihood projects reward differently from policy utility. Both can be correct: better coverage can reveal an incompatible residual more accurately. On-policy refitting can then change the pseudo-true target each round. Our exact feedback example in Proposal A preserves full support and correct BT labels while violating realizability; it does not contradict the coverage theorem.

The relevant coverage quantity is a response-feature covariance ratio, not semantic diversity, rank consistency or factuality. The domain-shift study's response variety, the stage paper's ordering consistency and SPINAL's activation effective dimension measure different objects. A new theory must state which one its guarantee controls.

### 3.3 Cardinal calibration precedes policy optimization

MixDPO's positive sensitivity mixture can maintain the same winner while making the aggregate logit nonadditive across three answers. RDPO can avoid saturated binary inference by adding gaps, but relies on gap units. For an unknown shared monotone link, even all population pair probabilities need not identify the cardinal utility needed at a fixed KL temperature. Proposal B converts this into an explicit identified-set and calibration-budget problem.

CATTO illustrates a separate distinction: preference confidence and correctness probability need not coincide. Its printed affine loss does not have a calibrated probability as its minimizer. Even a proper loss with zero aggregate calibration error need not give optimal selection across subgroups. A new paper cannot use ECE alone as the target theorem.

### 3.4 Endpoint composition can lose information before interpolation begins

Tangent-space training reduces network-output curvature. YaPO moves interventions into a dictionary. ECLIPTICA seeks switchable controls. None alone shows that an optimized component checkpoint encodes every reward residual needed by a later mixed objective. In a restricted policy family, two different component rewards can have identical exact component optima and different mixed optima. Proposal C isolates that obstruction in an already log-linear action policy. Better numerical merging cannot reconstruct unobserved reward information.

Full reward-realizing component policies remain a positive case: a common-temperature product of experts recovers the unrestricted scalarized optimum. The proposed obstruction concerns restricted optima and a specified information model, not all policy composition.

## 4. What peer review changes

The [discussion audit](03_openreview_discussion_audit.md) records reviewer concern, author response and our adjudication separately. Its most consequential corrections are:

| Paper           | Review/rebuttal evidence                                                                                                            | Consequence for interpretation and new research                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| AuxDPO          | Authors explicitly restrict the equivalence argument to local/small movement; balancing one example does not repair all references. | A global neural nullspace guarantee is an open extension, not an established result.                           |
| HyPO            | Authors defend gradient pressure rather than a derivation of a new global KL optimum.                                               | Graph-potential coherence is an assumption audit; avoid crediting a global variational theorem.                |
| TI-DPO          | Discussion clarifies attribution target and roughly doubled training iteration cost.                                                | Test preference-specific attribution and equal compute; cheaper algebra is not cheaper full training.          |
| DuPO            | Current-policy inverse filtering and oracle curve comparisons are clarified.                                                        | Filtering quality matters; similar average curves do not prove item-level reward soundness.                    |
| MISP-DPO        | Top-k selection and estimator choices are clarified; some benchmark cells regress.                                                  | Full-support IID importance arguments cannot automatically justify greedy selection or a normalized estimator. |
| SafeDPO         | The 2025 rejected draft and 2026 accepted oral version differ; unsafe–unsafe handling was revised.                                 | Analyze versions separately; finite harmlessness tests do not prove zero risk or zero benign refusal.          |
| Uni-DPO         | Scalar scores provide supervision that some baselines do not receive.                                                               | Match information as well as compute; self-scoring ablations do not equalize every baseline.                   |
| P²-DPO         | Authors clarify one-time initial-model preference generation and extra training cost.                                               | Its results are not a test of iterative deployed-policy feedback.                                              |
| AW-DPO          | Segment judge reliability and prior CoT training are discussed; weight theory remains limited.                                      | Rationale/answer repeatability is not factual label validity; measure benign-request utility separately.       |
| On-policy stage | Added stronger-model/SPPO evaluations and prospective small-sample diagnosis; mixed sources already tested.                         | Generic source mixing is crowded; compare intervention outcomes and account for diagnosis cost.                |

Reviewer approval is evidence of how the contribution was assessed, not proof of mathematical correctness. A rejection identifies concerns in that version, not a timeless verdict on later work. The original forums were read through the website, without the OpenReview API.

## 5. Formula and experiment findings that affect proposal design

1. **A changed pair margin may cease to encode one global reward.** HyPO's clipped gaps and SP²DPO's edgewise scales can violate cycle additivity. Empirical usefulness can survive; an exact KL interpretation then needs another argument.
2. **Nested estimation has bias beyond the inner estimator.** MixDPO marginalization inside a logarithm, MISP normalization and ProtAlign log probability ratios require analysis after the nonlinear transform. Unbiased probabilities or sums do not imply unbiased losses or gradients.
3. **A monitoring distance is not a behavior certificate.** Identical uniform and identical point-mass distributions both have Fisher–Rao distance zero. SPINAL token-row permutation preserves the Gram spectrum exactly. Such diagnostics need matched controls and outcome validation.
4. **An implemented confidence objective must match its proof.** CATTO's printed loss is affine in a fixed confidence target; the proper-loss discussion analyzes another expression. ECLIPTICA's printed offline anchor is not the theoretical forward KL. Reconcile paper and code before using either as a guaranteed building block.
5. **Changed optimization is not added identifiability.** CFPO can improve sample reuse without recovering a missing reward scale; ProFit can stabilize imitation without learning masked unfamiliar facts; a tangent chart can improve smoothness without retaining invisible reward residuals.
6. **A headline improvement has multiple possible mechanisms.** CoT SFT, teacher-generated target answers, reference prealignment, label repair, decoding, token normalization, selection, extra annotations and stronger evaluators all need isolated controls.

The [independent checks](verification/analytic_checks_results.json) reproduce the feedback gradient from all ordered pairs, its two cycles, the inner-depth derivative, endpoint-information separation, unknown-link regret floor and several printed-expression counterexamples. These are finite-action diagnostics, not full source reproductions or LLM results.

## 6. Reassessment of the earlier proposals

The earlier broad weighting, verifier, token, visual-contrast and safety ideas are retained as background hypotheses, but are not the recommended main theory claims. Primary related work already covers validation influence, policy-aware response sampling, category safety, minimal visual counterfactuals and contrastive self-reward. Newer work also covers latent ternary heterogeneity, local policy-risk information design, performative stabilization, gradient/curvature-aware merging and reward-realizing product composition.

The new recommendation is to pursue one precise information or dynamics theorem package:

- **A — over-solving fresh misspecified preference batches:** exact phase/depth mechanism, then stochastic escape and a target-aware finite-sample correction.
- **B — cardinal calibration under unknown choice links:** identification and regret with admissible calibrated contrasts, then budget/noise lower and upper bounds.
- **C — information discarded by component policy optimization:** endpoint/fixed-summary lower bounds, then feasible residual recovery over a control family.

Each has a checked mathematical kernel. Each still needs substantial proof work to become a full theoretical research paper. Novelty is bounded by inspected related works, and acceptance cannot be inferred from a proposal. See the companion sprint report for proof obligations, experiments, expected patterns, ablations and stop criteria.

## 7. Detailed equation-level paper audits

The following integrates the substantive reading notes into this report so the analysis can be read without reopening every paper. The original eleven-paper material is retained for its mechanism analysis; references there to numbered proposals describe the **earlier sprint**, whose prioritization is superseded by Section 6 and the new companion report. Statements about missing public discussions are superseded by Section 4 and the dedicated audit.

### Part I — Original eleven papers

### 1. The common mechanism: DPO controls a gradient coefficient and a direction

The [original DPO paper](https://openreview.net/pdf?id=53HUHMvQLQ) starts from KL-regularized reward optimization:

$$
\max_\pi\mathbb E_\pi r-\beta\mathrm{KL}(\pi\|\pi_0),
\qquad
\pi^*(y|x)=Z(x)^{-1}\pi_0(y|x)e^{r(x,y)/\beta}.
$$

Under a Bradley–Terry preference model, response-independent $\log Z(x)$ cancels. With $m_\theta=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$,

**Equation D1.**

$$
\ell_{\mathrm{DPO}}=\operatorname{softplus}[-\beta(m_\theta-m_0)],\qquad
\nabla\ell_{\mathrm{DPO}}=-\beta\sigma[-\beta(m_\theta-m_0)]\nabla m_\theta.
$$

This decomposition clarifies the eleven papers. Quality weights, focal factors, reference clipping, and safety margins modify the scalar pressure. Token/segment weights change the direction. Candidate generation changes which directions exist. AuxDPO changes the reward space fitted by the likelihood. DuPO changes the feedback signal and uses GRPO experimentally.

The absolute margin $m_\theta$, reference-relative margin $m_\theta-m_0$, preference accuracy, and generated policy reward are different quantities. A pair can be correctly classified by its implicit reward while its preferred sequence remains less likely; neither ordering alone determines greedy decoding or downstream utility. Sequence length and shared parameter conflicts complicate both.

### 2. Uni-DPO: three controllers, not just sample weights

**Source:** [paper](https://openreview.net/pdf?id=G7DBGlgjjp), [supplied PDF](../../resources/11934_Uni_DPO_A_Unified_Paradi.pdf). Locators: main Eqs. (4), (5), (10), (11); Appendix B.2 Eqs. (20)–(26), B.3 Eq. (27).

**Intuition.** Study trustworthy examples the model still needs, while preventing pairwise ranking improvements from making excellent chosen responses less likely.

Let $a_w=|y_w|^{-1}\log\pi_\theta(y_w|x)$, similarly $a_l$, and define

$$
z=\beta[(a_w-a_l)-(a_w^0-a_l^0)],\quad
u=\beta(a_w-a_l)-\tau_{\mathrm{ref}},
$$

$$
w_q=\sigma[\eta(S_w-S_l)],\quad
w_p=[1-\sigma(u)]^\gamma,
$$

**Equation D2.**

$$
\ell_{\mathrm{Uni}}=-w_qw_p\log\sigma(z)-\lambda I a_w.
$$

$I$ activates when the chosen response is judge-rated sufficiently good and its likelihood has fallen below the reference.

**Innovation behind the formula.** The focal controller uses an *absolute length-normalized policy margin* $u$, whereas the pair loss uses reference-relative $z$. These measure different forms of success. A reference that strongly separates a pair can make ordinary relative training keep pushing an already easy absolute ranking; an absolute controller can reduce that pressure. The chosen-only term addresses another ambiguity: raising the chosen/rejected ratio can happen by lowering both probabilities, with the rejected falling faster.

Crucially, Uni-DPO differentiates through $w_p(\theta)$. Its effective coefficient includes

**Equation D3.**

$$
-w_q[1-\sigma(u)]^\gamma
\{[1-\sigma(z)]-\gamma\sigma(u)\log\sigma(z)\}\nabla z.
$$

The extra derivative term is positive inside the braces. A detached focal weight implements a different method. This is an objective modification, not simply resampling pairs.

**What remains assumed.** Judge score gaps must be reliable; large margins do not prove transfer utility. A sigmoid quality weight is at least .5 for correctly oriented nonnegative gaps, so ambiguous pairs are downweighted rather than eliminated. The conditional NLL indicator is a local switch, not a hard global likelihood constraint. Static data with a policy-dependent weight is not equivalent to fresh on-policy generation.

**Evidence.** Appendix A reports weak correlation between judge-score and implicit-reward margins (textual Spearman .08), supporting distinct signals. Table 3 component benefits vary by model/benchmark; length normalization is especially important in some base-model settings. Math gains quoted against the initial model should not be mistaken for gains over SimPO. The method's preprocessing includes scored/generated candidates and filtering; negligible extra arithmetic during training does not mean negligible total cost. Some task cells regress.

**Deep transferable insight.** Label reliability, unmet absolute behavior, and relative improvement should be measured separately. The missing fourth signal is whether the update direction improves the intended policy objective under shared capacity. This motivates sprint proposal #1. Appendix B.3's expanded gradient line has a sign inconsistency; use its unexpanded objective and Eq. (26) when reproducing.

### 3. Is on-policy data always best? Data quality is relative to the learner

**Source:** [paper](https://openreview.net/pdf?id=tz9mJmgrdM), [supplied PDF](../../resources/24230_Is_On_Policy_Data_always.pdf). Locators: §4, Definitions 5.3/5.5, Eqs. (7)–(11), Algorithm 1, Appendices E/F.

**Intuition.** A learner lacking preference knowledge can benefit from broad examples from other models; a more aligned learner may benefit from comparisons around its own behavior. The same corpus can therefore be useful at one stage and unhelpful at another.

The paper defines a diagnostic reward-derived distribution $\pi_G\propto e^{r^*}$ and

$$
P^*(y_1\succ y_2|x)=\sigma[\log\pi_G(y_1|x)-\log\pi_G(y_2|x)],
$$

**Equation D4.**

$$
C(P_1,P_2)=\mathbb E\,\mathbf1\{\mathbf1[P_1>.5]=\mathbf1[P_2>.5]\}.
$$

Its stage diagnostic uses cross-source comparison counts. Algorithm 1 generates two candidates from each source, compares all four cross-source pairs, and chooses off-policy when $V_{\mathrm{off}}>V_{\mathrm{on}}$.

**Innovation.** Candidate provenance is contextual: model ability, data source, and judge jointly determine whether fresh sampling helps. This is more useful than an unconditional “online beats offline” rule. The stage is relative to the comparison corpus, not an intrinsic binary property of the model.

**Limits of the derivation.** Thresholded rank consistency does not identify probability distributions: $(.7,.3)$ and $(.6,.4)$ have identical ordering and different probabilities. Theorem 5.1's reward uniqueness needs the usual gauge $r+c(x)$. The finite-data discussion shows absence of a universal recovery guarantee, not literal impossibility of any finite exact recovery. The source-self-likelihood assumption is only partly checked, and Table 3's ratio orientation conflicts with Algorithm 1. Use actual counts when reproducing.

**Evidence.** On the tested first iteration, Llama gains more from on-policy than off-policy data; Zephyr does the reverse, then benefits from on-policy after an off-policy stage. These refute a universal empirical ordering. A third data source helps explore quality versus diversity but changes several factors together. A reported stage-prediction failure and an SLiC-HF collapse limit broad generalization.

**Deep insight.** A source's answer quality does not identify its training value. Coverage, learnability, label reliability, and gradient compatibility are separate. A better stage detector should predict the outcome difference between interventions, not just their judge win counts. This supports #6, while AuxDPO challenges the assumption that preference-distribution agreement is enough for policy utility.

### 4. AuxDPO: separate reward estimation from reachable policy movement

**Source:** [main paper](https://openreview.net/pdf?id=btEiAfnLsX), [supplied PDF](../../resources/24413_Why_DPO_is_a_Misspecifie.pdf), [official supplement](https://proceedings.iclr.cc/paper_files/paper/2026/file/8f6c674a7edf57912f1a5830f4a66e3f-Supplemental-Conference.pdf). Locators: Proposition 1/Eq. (4), §3.1, §4.1 Eqs. (5)–(6), §4.2 Eq. (7), Proposition 9.

**Intuition.** Shared parameters cannot independently satisfy all preferences. Fitting comparisons and maximizing policy reward can require different compromises, even with perfect labels and infinite data.

For implicit reward $r_\theta^\beta=\beta\log(\pi_\theta/\pi_0)$, DPO projects onto the realizable reward family:

**Equation D5.**

$$
\arg\min_{r\in\mathcal R^\beta}
\sum_e n_e\,D_{\mathrm{KL}}[
\operatorname{Bern}(p_e^*)\|
\operatorname{Bern}(\sigma(r_w-r_l))].
$$

**The consequential insight:** pair counts define the projection metric. Under misspecification, more data of the same distribution estimates the wrong compromise more accurately; changing frequencies changes the limiting policy, not merely its variance.

Let $A$ contain reference score gradients as columns, $D=\operatorname{diag}(\rho\pi_0)$, $G=AD$, and $F=ADA^\top$. Locally,

**Equation D6.**

$$
r_\theta^\beta\approx\beta A^\top d,\quad
J(d)\approx J(0)+d^\top Gr^*-\frac\beta2d^\top Fd,
\quad d_R=\beta^{-1}F^\dagger Gr^*.
$$

The policy only uses $Gr^*$. Reward components in $\ker G$ can matter for comparison fitting while being irrelevant to the local natural-gradient target. AuxDPO fits $r_\theta^\beta+\delta$, constraining $\delta\in\ker G$.

The empirical penalty places the square *outside* the mean:

**Equation D7.**

$$
-\frac1n\sum_i\log\sigma(z_i+\delta_{wi}-\delta_{li})
+\lambda\left\|\frac1{2n}\sum_i[
\delta_{wi}\nabla\log\pi_0(y_{wi}|x_i)+
\delta_{li}\nabla\log\pi_0(y_{li}|x_i)]\right\|^2.
$$

This permits cancellation across offsets, essential to nullspace fitting. A mean of individual squared penalties is different.

**Guarantee boundary.** The theorem concerns a constrained population/local construction with sufficiently small policy movement; it does not make penalized neural minibatch training globally equivalent to RLHF. A fixed parameter neighborhood cannot keep $\beta\log(\pi_\theta/\pi_0)$ bounded as $\beta\to\infty$; a shrinking neighborhood is required. The RLHF comparison assumes accurate reward and successful objective optimization, not arbitrary PPO.

**Evidence.** Main Table 1 measures relative changes in pair-ranking accuracy; Table 2 separately tests generated answers. The supplement includes capacity ablations and already demonstrates that balancing comparisons can be insufficient. Batch null constraints, offset scaling, and reference-gradient cost differ between theory and implementation descriptions; code verification remains necessary.

**Deep insight.** Do not force unrepresentable reward residuals to become policy movement. This leads to a sharper question than another weighting heuristic: can any allowed weighting reach the desired policy target? Sprint #1 derives a feasibility condition and local regret consequence.

### 5. HyPO: “less wrong than the reference” is not “correct”

**Source:** [paper](https://openreview.net/pdf/c575b079c37f32a6ab4870366802e118e41ce299.pdf), [supplied PDF](../../resources/362_Mitigating_Mismatch_within.pdf). Locators: §3 Eqs. (7)–(13), Figure 3/Appendix A.5, Tables 1–2, Appendix D.

**Intuition.** If the reference strongly prefers the bad response, DPO can ease off after relative improvement even while the model still prefers that bad response.

**Equation D8.**

$$
\ell_{\mathrm{HyPO}}
=\operatorname{softplus}[-\beta(m_\theta-\max(m_0,\gamma))],
\quad\gamma\ge0.
$$

For $\beta=1,m_0=-3,m_\theta=-1$, ordinary pressure is $\sigma(-2)=.119$, versus $\sigma(1)=.731$ with $\gamma=0$. The per-pair scalar improvement is exact. At initialization $\theta=\mathrm{ref}$, ordinary DPO assigns .5 to every pair; attenuation develops during relative improvement.

**Innovation.** Conditionally editing the reference changes the update's implicit stopping criterion while retaining the reference on optimistic pairs. It occupies a useful middle ground between a fixed reference and eliminating the reference entirely.

**Global caveat, our derivation.** Reference scores imply additive gaps. With scores $(2,0,1)$ on responses $A,B,C$ and preferred edges $A>B,B>C,A>C$, clipped gaps $(2,0,1)$ violate $b_{AB}+b_{BC}=b_{AC}$. Hence clipping need not correspond to any single normalized reference. The empirical loss remains legitimate, but preserved logistic form alone does not prove a global KL/proximal guarantee.

Larger coefficients do not guarantee better whole-model updates when gradients conflict. Noisy labels can amplify the wrong correction. Absolute sequence ordering is also not generated-task quality.

**Evidence.** Figure 3/Appendix A.5 isolates clipping with matched initialization/reference and no home advantage: absolute agreement 60.4 versus 55.9 and head-to-head preference 55.9%. Main-table gains combine clipping, a SimPO-prealigned reference, and home advantage. Some capability cells regress; reference prealignment has an upstream cost.

**Deep insight.** A reference is both a prior and an example-weighting mechanism. Pairwise editing has a global coherence cost when multiple answers are compared. The sprint's extra hypothesis tests that cost without assuming graph repair improves utility.

### 6. TI-DPO: attribution is a noisy estimator, not a preference oracle

**Source:** [paper](https://openreview.net/pdf?id=cMEnMVvMw9), [supplied PDF](../../resources/6576_Token_Importance_Guided_D.pdf). Locators: §4.2 Eqs. (5)–(12), §4.3 Eq. (13), §4.4 Eqs. (14)–(17), Appendix A.

**Intuition.** One decisive token may drive preference while many shared stylistic tokens dilute training. Ordinary DPO already updates all autoregressive tokens; its limitation is a single pair-level coefficient, not absence of token gradients.

The attribution target is the maximum final-position next-token logit:

$$
I_t=\|\nabla_{e_t}\max_v L_{T-1,v}\|_1,\quad
W=\lambda I/\sum_jI_j+(1-\lambda)P,
$$

**Equation D9.**

$$
P_t\propto e^{-(t-\mu)^2/(2\sigma^2)},\quad
\mu=(T-1)/2,\quad\sigma=T/4.
$$

With token log ratio $b_t=\log[\pi_\theta(y_t|x,y_{<t})/\pi_0(y_t|x,y_{<t})]$, the weighted margin is

**Equation D10.**

$$
\Delta r=\sum_t w_t^w b_t^w-\sum_t w_t^l b_t^l.
$$

The method also introduces an anchor response with log ratios $d_t$:

**Equation D11.**

$$
\mathcal L_{\mathrm{triplet}}=
\mathbb E\left[\sum_t(d_t-b_t)^2-\sum_t(c_t-d_t)^2+\alpha\right]_+,
\quad \mathcal L_{\mathrm{TI}}=\mathcal L_{\mathrm{weighted}}+\gamma\mathcal L_{\mathrm{triplet}}.
$$

Here $b,c$ correspond to chosen/rejected answers. Position matching and unequal lengths need explicit handling; scalar token-score proximity is not semantic correspondence.

**Innovation.** The Gaussian is a shrinkage prior: it prevents noisy attribution from concentrating all update mass on accidental positions. The anchor attempts to constrain score geometry beyond a static pair. Both are inductive biases. The attribution measures logit sensitivity, not sensitivity of human preference; the anchor is not guaranteed intermediate in quality.

**Theory audit.** Independent zero-mean irrelevant-token noise with fixed weights has variance $\sigma_\epsilon^2\sum_t w_t^2$, versus $|N|\sigma_\epsilon^2$ unweighted. This does not account for signal bias, correlated tokens, or noise-dependent weights.

The loss-superiority proof uses variance ordering where a stronger coupling/convex-order argument is needed. Our executed counterexample has mean-zero margins: 99% mass at 0 plus .5% at each of ±10 gives variance 1 and logistic loss .736216; equal mass at ±.9 gives smaller variance .81 but greater loss .791154. Therefore smaller variance alone does not establish lower expected logistic loss. The actual total objective also includes the nonnegative triplet penalty.

The KL-efficiency claim additionally assumes correct identification of relevant positions and unavoidable DPO waste on irrelevant tokens. These do not follow from logit-gradient salience; in a realizable factorized task DPO can already leave irrelevant-token probabilities unchanged.

**Reproduction issues.** Gaussian normalization, weight mass/effective temperature, whether weights are detached, and Eq. (12)'s omitted $\beta$ versus its algorithm must be resolved. A policy-dependent attribution weight contributes extra derivatives unless detached.

**Evidence.** Table 1 reports average 62.3 versus DPO 57.7 and GRPO 62.1; small aggregate differences require uncertainty. Component/noise ablations support a useful recipe. Appendix D acknowledges roughly doubled training time and weaker holistic reasoning cases.

**Deep insight.** Credit allocation is an estimation problem with bias, variance, and causal-target mismatch. A new token method must test whether its attribution identifies the preference-changing evidence, match total weight mass/temperature, and compare compute. This informs sprint #3.

### 7. Alignment-Weighted DPO: prevent one segment from compensating for another

**Source:** [paper](https://openreview.net/pdf?id=OuMNJoKJBQ), [supplied PDF](../../resources/15113_Alignment_Weighted_DPO_A.pdf). Locators: §3 interventions, §4 Eqs. (3)–(4), Appendices E/H/J/O.

**Intuition.** A response can have a safe-looking rationale and an unsafe answer, or vice versa. One whole-sequence score can hide the defective component.

For reasoning/answer segment $s$,

$$
q_s(x,y)=\sum_{t\in s(y)}
\log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_0(y_t|x,y_{<t})},
$$

**Equation D12.**

$$
\mathcal L_{\mathrm{AW}}
=w_{\mathrm{rs}}\,f(z_{\mathrm{rs}})
+w_{\mathrm{rp}}\,f(z_{\mathrm{rp}}),\quad
f(z)=\operatorname{softplus}(-z).
$$

Printed weights use segment harmfulness-score gaps $d_s=h_s^{\mathrm{chosen}}-h_s^{\mathrm{rejected}}$, normalized by their sum.

**Innovation behind separate logistic losses.** For nonnegative normalized weights, convexity gives $\sum_s w_sf(z_s)\ge f(\sum_sw_sz_s)$. An excellent margin in one segment cannot cancel a bad margin in another as readily. This is a mathematical explanation for component consistency, not a policy safety theorem.

Mixed-sign gaps can produce negative or singular printed weights, exactly when segments disagree. The source does not fully resolve this in the displayed formulas. A reproduction needs code inspection or an explicitly labeled alternative handling; silently inserting clipping changes the method.

**Causal insight and its limit.** Removing putatively reasoning-critical heads reduces reasoning while basic safety persists. This supports partial functional separation under that intervention; it does not prove that all jailbreaks result from absent reasoning. Probe datasets can differ in difficulty, source, and lexical cues, and a decodable feature need not be causally necessary. Matched random-head controls and trace-semantic interventions would strengthen the claim.

**Evidence.** The full pipeline includes teacher-generated CoT SFT data, new candidates, and the objective. Appendix O's matched objective comparison is cleaner: Llama-3.1-8B attack success 1.83% to .81%, utility 57.66 to 58.27. Some models trade utility for safety. The DPO construction uses harmful WildJailbreak examples, so MMLU preservation does not directly establish low false refusal on benign lookalikes. Reasoning-judge paraphrase correlation is lower than response-judge correlation.

**Deep insight.** Multi-component preference learning must prevent error compensation and calibrate each component label. Visible rationales are a training channel, not proof of the mechanism of safe behavior. For a new method, compare behavior after changing trace meaning while preserving style/length; evaluate legitimate-request utility separately.

### 8. SafeDPO: safety labels define support, margins define finite-training pressure

**Sources:** [supplied old draft](../../resources/5041_SafeDPO_A_Simple_Approach.pdf), [final ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/447494b03345309e0ce196c78c7e5c83-Paper-Conference.pdf). Old locators: §4 Eqs. (10)–(17), Appendix A; final: §3.2/Eq. (12), Propositions 4.2–4.4, §5.2.

**Intuition.** Ranking two answers by harmlessness does not tell whether either is acceptable. Absolute binary labels establish a safety boundary that helpfulness preference alone cannot identify.

With unsafe indicator $h$, the ideal hard-support optimum is

**Equation D13.**

$$
\pi^*(y|x)=
\frac{\pi_0(y|x)e^{r(x,y)/\beta}\mathbf1[h(x,y)=0]}
{\sum_{z:h(x,z)=0}\pi_0(z|x)e^{r(x,z)/\beta}}.
$$

This is a truncated Boltzmann distribution: helpfulness is optimized within safe support. Safe reference support and bounded rewards are important feasibility assumptions.

Pairs are reoriented to put safe ahead of unsafe. The final version discards unsafe–unsafe pairs; the old draft retained them, although subtracting two $-\infty$ safety rewards is undefined. Safe–safe pairs retain helpfulness preference.

**Equation D14.**

$$
\ell_{\mathrm{Safe}}=
\operatorname{softplus}[-z+(\tilde h_l-\tilde h_w)\Delta].
$$

For safe/unsafe pairs, subtracting $\Delta$ from the preferred logit delays saturation and increases pressure. It is not merely adding a constant reward to the safe answer.

**Innovation and assumption boundary.** The direct loss removes a separate reward/cost optimization pipeline by exploiting the support-constrained variational solution. The ideal target can remain invariant to finite margins while finite shared-parameter training changes markedly. A softmax LLM normally assigns positive mass to all token-supported sequences; an exact zero-unsafe-support boundary is an ideal limit, not a universal attained guarantee.

Appendix A's finite reward-penalty bound depends on safe reference mass and reward spread. A reward penalty $C$ and the surrogate margin $\Delta$ are different quantities; one cannot transfer that bound to a chosen training margin without analysis.

**Evidence.** The final paper adds SafeDAA and benign-use evaluation; XSTest reports 12.4% over-refusal with 100% harmless ratio on its finite test. Excessive margins can harm helpfulness. The draft's human study also highlights a measurement ambiguity: literally satisfying a harmful request and providing useful safe redirection are different notions of helpfulness.

**Deep insight.** Hard-support theory, surrogate classification, finite-model optimization, and distributional generalization need separate claims. Safety-budget estimates must account for evaluator error and sampling. The original sprint #4 idea is directly crowded by Cat-DPO and calibrated safety methods.

### 9. DuPO: conditional inverse problems create feedback, but need soundness

**Source:** [paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/332aa4b65f636f7187982ae5bb16e50a-Abstract-Conference.html), [supplied PDF](../../resources/6741_DuPO_Enabling_Reliable_Se.pdf). Locators: Definition 2, Eqs. (3)–(4), §4.2.5, Appendices A–C.

**Intuition.** An answer may not identify the entire original problem. Knowing part of the problem can make the rest reconstructible: the sum alone cannot recover both addends, but the sum plus one addend can recover the other.

$$
x=(x_k,x_u),\quad T_{\mathrm{cd}}(y,x_k)\mapsto\hat x_u,
$$

**Equation D15.**

$$
R(x,y)\propto e^{-\lambda d(x_u,T_{\mathrm{cd}}(y,x_k))},
\qquad \max_\theta\mathbb E_{\pi_\theta}R.
$$

The experiments use GRPO, not a standard pairwise DPO loss. Its central innovation is the *conditional verifier task*, which reduces information ambiguity and verifier difficulty.

Appendix A selects dual questions for which exactly one candidate in a sampled pool successfully reconstructs the unknown component. This is empirical pool uniqueness, not mathematical injectivity over all answers. Filtering uses additional model calls and pretrained competence, so annotation-free does not mean cost-free or supervision-free in a broad historical sense.

**Soundness gap, our interpretation.** Consistency of a correct primal output does not establish that every consistent output is correct. A wrong addition solver $G(A,B)=A+B+1$ and inverse $V(A,y)=y-A-1$ have perfect cycles. This is a theoretical counterexample, not an observed DuPO attack. In the actual math reranker, the final answer enters the dual question, so reasoning-trace leakage must not be used as a substitute test.

**Evidence.** Seed-X COMET 87.0→89.1 across 756 directions; Qwen3-4B four-task average 67.9→74.4. The math evaluation averages 32 sampled responses, which is neither pass@32 nor best-of-32. The separate +9.3 reranking gain uses extra dual computations. Removing component filtering worsens results, showing construction quality matters. Similar oracle training curves do not certify item-level reward correctness.

**Deep insight.** A verifier's information channel, difficulty, and soundness are separate. Probe task relations can exclude specified coherent errors, but shared-model agreement alone can preserve them. Sprint #2 develops an explicit error-family result; Co-Reward and prior cycle learning are required comparisons.

### 10. P²-DPO: change the visual evidence, not just the answer text

**Source:** [paper](https://openreview.net/pdf?id=ekOwxTn65Y), [supplied PDF](../../resources/8851_P_2_DPO_Grounding_Halluci.pdf). Locators: §4 Eqs. (3)–(8), Appendix A Eqs. (15)–(24), Appendix E Eqs. (25)–(31).

**Intuition.** The model may attend to the right region yet fail to use it. Generate responses with enhanced and impaired visual evidence to expose that failure directly; separately train robustness to noisy images.

Focus pairs come from original-plus-crop $I_{\mathrm{aug}}$ versus salient-region erasure $I_{\mathrm{deg}}$. Robustness pairs come from contrastively amplified clean-image decoding versus noisy-image decoding. Standard focus DPO scores both responses under the original image; robustness DPO scores both under noise.

Define confidence gain

$$
\Delta_\pi(y)=\log\pi(y|I_{\mathrm{aug}},P)
-\log\pi(y|I_{\mathrm{deg}},P).
$$

Its calibration term is

**Equation D16.**

$$
\mathcal L_{\mathrm{cal}}
=-\mathbb E\log\sigma\{\beta[
\Delta_{\pi_\theta}(y_w)-\Delta_{\pi_0}(y_l)]\}.
$$

**Innovation behind the formula.** This is not a standard trainable chosen/rejected DPO difference. The losing gain is frozen at the reference, so gradients directly increase the winning response's enhancement-versus-erasure contrast. It ties training to an evidence manipulation, rather than merely rewarding preferred text. Dynamic deficit weights use a CLIP crop/original score ratio to balance local perception and global robustness.

**Theory boundary.** Appendix A correctly exposes gradient differences as a target, but its gradient–mutual-information association is an assumption, and the Fisher/posterior-KL link is explicitly heuristic. Small textual edit distance bounds gradient differences only under a specified Lipschitz metric. Larger gradients do not establish more correct evidence use.

The exact conditional mutual information is

$$
I(Y;F_v|P)=\mathbb E\log[p_\theta(Y|F_v,P)/p_\theta(Y|P)].
$$

The degraded-image likelihood in $\Delta_\pi$ is not automatically the marginalized denominator $p_\theta(Y|P)$; fixed training winners are not the same expectation as generated $Y$. Therefore maximizing that gain is not, without further assumptions, exactly maximizing mutual information or factual correctness. Dependence on a spurious visual cue can also be strong.

The off-policy vanishing-gradient argument considers $\pi_0(y_w)\to0$ while holding a positive $\pi_\theta(y_w)$ fixed. It is not a universal startup effect: at $\theta=\mathrm{ref}$, every implicit reward is zero and the scalar weight is .5. Finite softmax models generally have nonzero sequence support; low probability and zero support should be distinguished.

**Evidence.** TextVQA attention focus rises 14.73→18.71 and high-focus processing accuracy 66.29→70.10; noise tests and multiple VLM backbones improve many hallucination metrics. These support useful intervention-based training, not proof of causal isolation. Attention-selected crops, prompts, reference-margin filtering, and contrastive decoding may contribute jointly. Some benchmark cells regress.

Calling the data “on-policy” means self-generated by a reference under manipulated contexts. It is not continuous fresh sampling from the evolving target under exactly the training observation distribution. This distinction matters when comparing with the on-policy-stage paper.

**Deep insight.** Intervention contrast is stronger than answer-only correlation but still requires intervention validity. Does erasure remove the claimed fact, or introduce an artifact? Does increased evidence dependence improve truth? Sprint #3 focuses on semantic edits plus nuisance controls, rather than claiming this existing calibration idea anew.

### 11. MISP-DPO: a negative set defines the error geometry

**Source:** [paper](https://openreview.net/pdf?id=HEFPwoGtTj), [supplied PDF](../../resources/24535_Importance_Sampling_for_.pdf). Locators: §3 Eq. (5), §4 Eqs. (6)–(12), Algorithm 1, Appendix A.3.

**Intuition.** One wrong image can teach only one distinction. Several semantically distinct wrong images can expose errors in different visual factors, with stronger pressure on those the model currently confuses.

For positive image $m_p$, negatives $m_i$, and shared positive response $y_p$, set

$$
a_i=\beta\left[
\log\frac{\pi_\theta(y_p|x,m_i)}{\pi_0(y_p|x,m_i)}
-\log\frac{\pi_\theta(y_p|x,m_p)}{\pi_0(y_p|x,m_p)}
\right].
$$

The consistent negative-log-likelihood interpretation is

**Equation D17.**

$$
\mathcal L_{\mathrm{img}}=\log(1+\sum_i e^{a_i}),\quad
\nabla\mathcal L_{\mathrm{img}}
=\beta\frac{S}{1+S}\sum_i\frac{e^{a_i}}S
[\nabla\log\pi_\theta(y_p|m_i)-\nabla\log\pi_\theta(y_p|m_p)],
$$

where $S=\sum_i e^{a_i}$. Printed Eqs. (6)/(9)/(10) omit the negative sign of the loss, while Eq. (7) matches this minimization convention. This must be resolved against implementation, not silently attributed as corrected source notation.

**Innovation.** The listwise gradient forms a mixture of visual correction directions. Softmax probabilities automatically emphasize current confusing negatives; SAE-guided selection attempts to cover distinct error factors rather than merely repeat perturbations of one image.

CLIP image/text embeddings are fused into difference vectors. SAE reconstruction error and latent activation magnitude score candidate novelty/difficulty; greedy selection adds a diversity term. Large representation distance or SAE reconstruction error need not correspond to a human-meaningful false claim, so semantic validity remains a separate measurement problem.

**Importance-sampling audit.** Appendix A.3 proves that

$$
\hat g=\frac1K\sum_{k=1}^K
\frac{p_\theta(m_k)}{q(m_k)}\Delta(m_k),\quad m_k\stackrel{\mathrm{iid}}\sim q
$$

is unbiased for $g=\mathbb E_{p_\theta}\Delta$, with MSE ≤$L^2\|p/q\|_\infty/K$, given exact probabilities, full proposal support, bounded gradients, and iid draws. If the full-set prefactor $S/(1+S)$ is known, multiplying it preserves unbiasedness.

This does not prove that recomputing $S$, its normalization, and its nonlinear prefactor from a small selected set is unbiased. Algorithm 1 is deterministic greedy top-K and does not specify full-support random proposal probabilities. The exact full-target gradient, sampled-subset objective, and implemented selected-slate training must be distinguished.

**Executed diagnostic.** Two negative exponential scores .1 and 10 have full loss $\log(11.1)=2.406945$. Uniformly sample one and multiply its value by two: the denominator estimate is unbiased, but expected log loss is 1.613422. This Jensen effect is generic; it does not prove the specific code uses that estimator.

**Evidence.** Three VLM backbones and five benchmarks show broad gains; on LLaVA the POPE result is still below the untouched base, despite improving over DPO. Baseline fine-tuning configurations differ in some details. SAE selection and multiple negatives add preprocessing and per-sample cost; measure efficiency at equal total budget, not just selected negative count.

**Deep insight.** Negative coverage changes the objective geometry; importance sampling estimates a fixed objective only when its target/proposal are precisely specified. “Informative negatives,” “unbiased gradient,” and “efficient selection” are three distinct claims.

### 12. ProtAlign: multi-objective margins account for competing rewards

**Source:** [paper](https://openreview.net/pdf/413fa4fa5263ae1d6e5d226f6bc2b7fbed2ed628.pdf), [supplied PDF](../../resources/4249_Property_Driven_Protein_I.pdf). Locators: §4 Eqs. (2)–(5), Appendix B.1 Eqs. (12)–(15), conclusion.

**Intuition.** Preference for one predicted property can damage another. A comparison loss should account for what its chosen/rejected pair already implies about auxiliary objectives.

The policy target is a scalarized objective:

**Equation D18.**

$$
\max_\pi\mathbb E_\pi\sum_k w_k r_k-\beta\mathrm{KL}(\pi\|\pi_0).
$$

Its Boltzmann relation gives $\sum_k w_k r_k=\beta\log(\pi/\pi_0)+\beta\log Z$. Solving for one property's pairwise reward difference gives

**Equation D19.**

$$
\Delta r_k=\frac1{w_k}\left[
\beta\Delta\log(\pi/\pi_0)-\sum_{k'\ne k}w_{k'}\Delta r_{k'}
\right].
$$

This explains the adaptive margin: the aggregate policy movement is partly explained by other rewards, so the component likelihood must subtract their contribution.

**Formula audit.** Main Eq. (4) places $w_k$ as a multiplier of the bracketed logit and introduces a tunable auxiliary scale; Appendix B.1 derives $1/w_k$. The reciprocal follows from the algebra above. These can define different finite objectives; verify the implemented convention before asserting exact equivalence to the scalarized reward target. Equal property weights can hide the discrepancy.

**Second innovation: common random numbers.** The order-agnostic model averages probabilities over sampled decoding permutations. Evaluating policy and reference on the same permutations aims to stabilize their log ratio:

**Equation D20.**

$$
\hat p_\theta(y|x)=K^{-1}\sum_k p_\theta(y|x,\sigma_k),
\quad \hat p_0(y|x)=K^{-1}\sum_k p_0(y|x,\sigma_k).
$$

For paired random estimates, $\operatorname{Var}(A-B)=\operatorname{Var}A+\operatorname{Var}B-2\operatorname{Cov}(A,B)$. Shared randomness helps when covariance is positive; it is not unconditional. An unbiased probability average does not make its log or ratio unbiased. This connects to MISP-DPO's nested-estimator issue.

Semi-online iterations regenerate candidates from the updated policy, then train offline on a scored batch. Thresholded score gaps improve label separation but can discard informative near-boundary examples or preserve confidently wrong predictors.

**Evidence boundary.** Experiments on observed/de-novo structures and a separate application setting show improvements in predicted properties while preserving many structural metrics. The authors explicitly report no wet-lab validation. Additional computational descriptors support plausibility but do not substitute for independent measured outcomes. Proxy-label calibration under policy shift remains unresolved.

**Deep insight.** A weighted reward optimum is not a guarantee that every objective improves or that the full Pareto frontier is recovered. Predictor uncertainty, scalarization conflict, Monte Carlo estimation, and distribution change are distinct risks. Sprint #5 addresses the general statistical audit problem at a high level, not additional biological design capability.

### Part II — Coverage and unification

### 1. What the coverage paper actually proves

Primary source: [arXiv:2601.08421v1](https://arxiv.org/abs/2601.08421), locally resources/2601.08421v1.pdf. Locator: Section 2.1.

#### Setup and assumptions

Sections 2.1–3 use finite prompt and action spaces, fixed deployment prompt law D, exact Bradley–Terry feedback, known bounded features, a compact linear softmax class, fresh independent batches conditional on the current policy, and exact constrained empirical minimization:

$$
p^*_{ab}(x)=\sigma(r^*(x,a)-r^*(x,b)),\qquad
\pi_\theta(a\mid x)\propto\pi_0(a\mid x)e^{\theta^\top\phi(x,a)},\qquad
\|\phi(x,a)\|_2\le1.
$$

The KL coefficient is denoted gamma, and the target is

$$
\pi^*(a\mid x)\propto\pi_0(a\mid x)e^{r^*(x,a)/\gamma}.
$$

Assumption 2 requires this unrestricted Boltzmann target to belong to the policy family, with $\|\theta^*\|_p\le R$, $p\in[1,2]$. Equivalently, reward differences must satisfy $r^*(x,a)-r^*(x,b)=\gamma(\theta^*)^\top(\phi(x,a)-\phi(x,b))$. This is much stronger than true labels satisfying BT. Assumption 3 requires $V(\pi^*)\succeq\lambda I$ on the identifiable subspace, where lambda is at most order $1/d$. The main text permits a gauge-restricted subspace. Singular dimensions should be removed rather than inverting a singular covariance.

Equation (4) defines feature coverage:

$$
V(\pi)=\mathbb E_{x\sim D}\operatorname{Cov}_{a\sim\pi(\cdot\mid x)}[\phi(x,a)],
$$

$$
C_{\pi\to\pi'}=\inf\{C:V(\pi')\preceq C V(\pi)\},\qquad
C_*^p(r)=\sup_{\|\theta-\theta^*\|_p\le r}C_{\pi_\theta\to\pi^*}.
$$

This measures the worst *policy-relevant linear variance ratio*, not semantic diversity, preference rank agreement, coverage of every text response, or probability mass on correct answers. Lemma B.8 shows $C_{\pi_\theta\to\pi_{\theta'}}\le\|\pi_{\theta'}/\pi_\theta\|_\infty\le e^{2\|\theta-\theta'\|_2}$, so the feature quantity may be much smaller than density-ratio coverage.

Assumption 4 imposes a connected-sublevel growth condition on $C_*^p(r)/r^2$; convexity or a suitable convex upper envelope is sufficient. The later fixed-point proof uses square-root convexity for its simple convexity argument, and the broader connected-sublevel condition handles more general profiles. Do not silently assume the empirical coverage curve always satisfies this condition.

#### Offline lower bounds

Proposition 2.1 and Appendix E.1 construct near-uniform tabular target policies with adversarially skewed references. With $d=\operatorname{poly}(R)$, coverage grows as $e^R$ for $p=1$ and $d^{-1}e^{2^{1-1/p}R}$ for $p>1$. Theorem 2.2 must be read together with Appendix E.2: its main text accidentally says “for p > 1” immediately before displaying the p=1 inequalities. The appendix distinguishes cases unambiguously.

For p=1, the appendix gives, up to constants,

$$
\inf_{\widehat\theta}\sup_{\theta^*}\mathbb E\|\widehat\theta-\theta^*\|_1
\gtrsim d\sqrt{\frac{C_*^1(R)}n}\wedge1,\qquad
\inf_{\widehat\pi}\sup_{\pi^*}\mathbb E\operatorname{KL}(\pi^*\|\widehat\pi)
\gtrsim\frac{C_*^1(R)}n\wedge\frac1{d^2}.
$$

For p>1 it gives $\sqrt{dC_*^p(R)/n}\wedge1$ in parameter p-norm. This is an information-theoretic lower bound for samples from the fixed initial policy, not a lower bound against arbitrary active comparison queries. Appendix E.2 writes a unit-scaled BT logit in its construction; a uniformly explicit gamma-dependent lower bound would require restoring that scaling. The broad exponential coverage phenomenon remains, but do not infer matching constants across their gamma-scaled upper and lower displays.

#### On-policy upper bound and its actual mechanism

Algorithm 1 samples both actions independently from the current policy and optimizes DPO against the **fixed initial reference**, not against the previous policy. Proposition 3.1, Equation (5), gives a one-step confidence radius:

$$
r_{k+1}^2=
\frac{52e^{4\gamma R}d^{2/p}}{\lambda\gamma^2n_k}
C_*^p(r_k)\log\frac{9\gamma Rn_k}{\delta},
$$

provided the resulting radius lies within $R$ and $1/(2\gamma)$. Appendix B.1 establishes the result via a likelihood/Hellinger generalization argument, a tight sigmoid inverse-slope inequality, and iterative tightening of the same estimator's radius. That tightening is an analytical bootstrap, not several optimization updates.

Theorem 3.2 defines

$$
\xi_n=\frac{52e^{4\gamma R}d^{2/p}}{\lambda\eta^2 n}
\log\frac{9\gamma RKn}{\delta},
$$

requires $\xi_n\lesssim1/C_*^p(R)$ and the stated radius/temperature condition, and obtains

$$
\operatorname{KL}(\widehat\pi_K\|\pi^*)
\le\|\widehat\theta_K-\theta^*\|_p^2
\le\frac{e^2\xi_n}{\gamma^2}\vee R^2\eta^{2K}.
$$

The appendix also controls forward KL and local chi-square. The exponential decay in rounds has a finite-sample floor and requires batches large enough to enter the improving-coverage regime. It is not a guarantee that arbitrary small batches improve monotonically.

Corollary 3.3 suppresses dimension and conditioning as polynomial factors in R and assumes $\gamma\asymp1/R$. Offline complexity carries an accuracy multiplier $1/\varepsilon^2$ on the initial coverage, whereas iterative learning needs sufficient initial coverage plus a final precision cost. This is not permission to claim a literal exponential total-sample gain in all regimes. Corollary 3.6 improves batch scheduling under the extra super-quadratic coverage condition: early batches shrink geometrically, then a final precision batch suffices. That scheduling condition is an additional assumption, not a measured law of neural training.

#### Preferential design: what is controllable

Algorithm 2 **allows a chosen joint prompt/response law**. It forms a global prompt marginal from a G-optimal design, combines it with per-prompt centered conditional designs, and samples from

$$
\widehat\pi_k'(x,a)=\tfrac12D(x)\widehat\pi_k(a\mid x)+\tfrac12\pi^g(x,a).
$$

Thus its exploratory prompt marginal need not equal deployment D. Proposition 4.2 guarantees squared leverage at most $d^2$ for these centered features. Theorem 4.4 assumes $\gamma\asymp1/R$, $\varepsilon\lesssim1/d$, and obtains two-round target covariance-norm error $\varepsilon$ at batch order $\widetilde O(dR^2/\varepsilon^2)$. Appendix C.2 reveals a warmup requirement of order $d^3R^2$, hidden by the small-epsilon condition: first fit controls a design norm, leverage converts that to a uniform log density ratio, and second fit transfers its error to the target norm. The theorem does not show the two-round procedure is optimal in dimension, computationally implementable on unrestricted text, or applicable under a fixed exploratory prompt marginal.

#### Distillation is fast because of exact fit, not because a frozen judge is correct

Section 5, Assumptions 5–6, and Theorem 5.1 use a bounded function class F, finite pseudodimension of pair differences, realizability of both target and reward model, and convex local **mean absolute deviation coverage**:

$$
\operatorname{MAD}_\pi(h)=\mathbb E_{x,a\sim\pi}|h(x,a)-\mathbb E_{b\sim\pi(\cdot\mid x)}h(x,b)|,
$$

$$
C_F(r)=\sup_{\operatorname{MAD}_{\pi^*}(f-f^*)\le r}
\inf\{C:\operatorname{MAD}_{\pi^*}(h)\le C\operatorname{MAD}_{\pi_f}(h),\ h\in F-F\}.
$$

This is different from variance coverage with p=1. Exact empirical reward-difference interpolation plus the VC epsilon-net bound gives a linear-error rate of order $n^{-1}$ and geometric round decay, with irreducible reward-model error $\varepsilon_{\mathrm{RM}}=\operatorname{MAD}_{\pi^*}(\gamma^{-1}\widetilde r-f^*)$. Mere determinism of reward-model output is insufficient: when the output is unrepresentable, their exact-fit proof no longer applies.

Algorithm 3 offers fixed-reference regression or moving-reference regression with reward calibration. The two are algebraically equivalent in the exact setting when calibration coefficient equals gamma; the experiments tune a different calibration coefficient. Theorem 5.1 should not be automatically applied to arbitrary tuned coefficients. Appendix D.1 starts with radius 2R although Assumption 6/main theorem discuss an envelope on [0,R]; a careful restatement should enlarge the envelope's interval or redefine R consistently.

#### Empirical evidence is narrower than theorem validation

Sections 6.1–6.2 test TL;DR and general chat, fixed reward models, five summarization runs, and small numbers of online iterations. Theorem labels are stochastic BT; summarization uses reward ordering, and PDFR uses sharp scaled soft labels. Experiments support practical benefit but do not identify all theoretical conditions. General-chat on-policy DPO beats off-policy on AlpacaEval, while the off-policy comparator is stronger on ARC and TruthfulQA in Table 1. RDRC's overall win rate grows, but individual capability scores fluctuate. “Monotonic” should be restricted to the plotted headline metrics.

### 5. Unification survey: conceptual map with serious qualifications

Primary source: [arXiv:2601.06108v1](https://arxiv.org/abs/2601.06108). Locator: Section 3.

The useful contribution is an organizing map: preference model, regularization, and data distribution. It assembles older methods and practical failure modes, rather than proving a new rigorous unification of arbitrary PPO/GRPO and pairwise losses.

Important source checks:

- Section 2.2 calls BT reward differences scale-free. Only additive shifts are invariant; multiplying reward differences changes probabilities.
- Its introductory claim that overly strong KL causes mode collapse conflicts with its later weak/no-KL explanation and Table 2. For exact unconstrained Boltzmann policies, large beta approaches the reference, while beta approaching zero concentrates on maximizing rewards.
- Equation (25) attributes likelihood displacement to a larger rejected-response partial derivative. Direct differentiation gives $\partial\ell/\partial\log\pi_w=-\beta\sigma(-z)$ and $\partial\ell/\partial\log\pi_l=+\beta\sigma(-z)$: equal magnitudes. Parameter-space coupling and normalization can cause displacement, but this displayed “asymmetry” does not show it.
- Theorem 6.4's hybrid sample-complexity expression lacks the required explicit model, oracle, complexity, and proof conditions. Use the original cited work to verify any rate.
- A margin is not generally a hard policy KL bound. The taxonomy's three axes are informative but not mathematically independent in misspecification: changing the regularizer changes the reachable implicit reward, and changing sampling changes its projection.
- The survey's HyPO is the Song et al. online-KL hybrid, **not** the supplied similarly named reference-clipping HyPO. Keep the acronyms disentangled.
- Empirical Table 3 combines different models and sources without a common matched evaluation protocol. Its numbers cannot establish an algorithm ranking or support a universal default.
- Reference metadata contains obvious inconsistencies, e.g. a 2025 arXiv identifier assigned a 2024 date. Verify originals before exporting bibliographic claims.

Use this survey as a search/taxonomy aid, not the authority for theorem assumptions or novelty clearance.

### Part III — Preference strength, ratings, confidence and switchable alignment

### 1. MixDPO: what is actually modeled

Source: [MixDPO](https://arxiv.org/abs/2601.06180), Sections 2–3, Equations (1)–(6), Appendices B–C.

#### Model and innovation

Let the implicit score be the policy/reference log ratio, with a sensitivity variable called beta in the paper. To prevent confusion with the KL temperature, use $B$ for this variable here:

$$
t_\theta(x,a,b)=\log\frac{\pi_\theta(a\mid x)}{\pi_{\mathrm{ref}}(a\mid x)}-\log\frac{\pi_\theta(b\mid x)}{\pi_{\mathrm{ref}}(b\mid x)}.
$$

$$
F_\eta(t)=\mathbb E_{B\sim P_\eta}[\sigma(Bt)],\qquad
\mathcal L(\theta,\eta)=-\mathbb E_D\log F_\eta(t_\theta).
$$

The substantive change is to marginalize a stochastic sensitivity **inside the logarithm**, then learn its distribution jointly with the policy. This is a marginal likelihood, not an average of DPO losses. Lognormal sensitivity uses reparameterized Monte Carlo with 16 draws; Gamma sensitivity uses a Lerch-transcendent expression approximated numerically by a truncated special-function series. Learning-rate and fixed-distribution ablations distinguish learning the distribution from merely injecting random temperatures.

**Intuition:** someone who distinguishes quality sharply and someone who weakly distinguishes it can agree on which answer is better. The model lets these different sharpness levels contribute to one aggregate choice probability. It does not model disagreement about the direction of quality: for every positive $B$, the sign of the preference is the sign of the same $t$.

The paper explicitly acknowledges this restriction in Limitations: sensitivity to a shared scalar quality is modeled; differing weights on attributes and systematic preference-direction heterogeneity are not. Therefore, pluralistic alignment results should be interpreted as subgroup performance observations, not recovery of each subgroup's value system.

#### The gradient reveals the mechanism

For a positively oriented pair:

$$
\frac{d[-\log F(t)]}{dt}
=-\frac{\mathbb E[B\sigma(Bt)(1-\sigma(Bt))]}{\mathbb E[\sigma(Bt)]}.
$$

Equivalently, this is the posterior average of $-B(1-\sigma(Bt))$ under the latent sensitivity posterior weighted by the observed winning event. Thus the distribution changes the response of a single shared scalar loss to margin; no example-specific distribution or annotator posterior is inferred unless extra conditioning is introduced. At zero margin the derivative is $-\mathbb E[B]/2$. A larger mean sensitivity increases the initial gradient. Consequently, the learned raw variance is not an invariant measure of dataset heterogeneity: utility scaling and optimization can change it.

#### Exact proposition M1: sensitivity mixtures generate nonadditive BT logits

Assume $B>0$ has a bounded distribution, with moments $m_j=\mathbb E[B^j]$, and define $h(t)=\operatorname{logit}F(t)$. The same third-order conclusion holds locally when the relevant moment/remainder conditions are satisfied. Expanding the sigmoid and logit at zero gives

$$
F(t)=\frac12+\frac{m_1}{4}t-\frac{m_3}{48}t^3+O(t^5),
$$

$$
h(t)=m_1t+\frac{m_1^3-m_3}{12}t^3+O(t^5).
$$

Hence

$$
h(2t)-2h(t)=\frac{m_1^3-m_3}{2}t^3+O(t^5).
$$

Strict convexity gives $m_3>m_1^3$ for nondegenerate positive $B$. For three alternatives with shared utilities $0,t,2t$, their population aggregate pair logits have a nonzero cycle residual. No scalar BT score can match all three edges, because scalar differences must add around every cycle.

**What this establishes:** strength heterogeneity alone can cause scalar BT misspecification, even when every annotator agrees on the ordinal ranking. This occurs before restricted neural capacity, optimization error, or conflicting values are introduced. It does not establish that the underlying population preference matrix is cyclic in its majority ordering; its majority ordering remains transitive.

A deterministic calculation with $B\in\{0.5,2\}$ equally likely gives $h(1)=1.107315$, $h(2)=1.786814$, and residual $h(2)-2h(1)=-0.427816$. These are toy calculations, not paper experiments.

This proposition is an explanatory building block. General mixed-logit misspecification and graph integrability are longstanding ideas; the proposition alone is not a sufficient new paper.

#### Exact proposition M2: joint sensitivity/utility scale is a gauge

Replacing $t$ by $ct$ and $B$ by $B/c$ preserves every choice probability. Lognormal and Gamma families are closed under this transformation. Raw mean and variance change, even though observations do not. A normalization such as $\mathbb E[B]=1$ defines a coordinate convention; it does not prove that a substantive cardinal reward scale was measured. Distribution shape can sometimes be identifiable under further structural assumptions and sufficiently rich repeated design, so this is not a claim that all shape parameters are universally nonidentifiable.

#### Exact proposition M3: finite Monte Carlo inside a log is generally biased

Appendix B approximates $F(t)$ with $\widehat F_K(t)=K^{-1}\sum_k\sigma(B_kt)$ and optimizes $-\log\widehat F_K(t)$. Although $\widehat F_K$ is unbiased for $F$, Jensen gives

$$
\mathbb E[-\log\widehat F_K(t)]\ge-\log F(t),
$$

with strict inequality when the estimator varies. Reparameterization gives an unbiased gradient of the **finite-sample surrogate's expected loss**, under interchange conditions, not generally of the exact marginal loss. The paper's unqualified unbiased-gradient language therefore needs qualification.

For the preceding two-point distribution, $K=1,t=1$ has a loss bias of 0.014989 and derivative bias of 0.004265. Bias can shrink with $K$; this calculation does not quantify the practical error at $K=16$.

#### Empirical boundary and numerical audit

- Section 4 uses Pythia-2.8B and Llama-3.2-1B, three datasets, and 425 sampled held-out examples per dataset; subgroup margins are observed-pair log likelihood differences, not subgroup utility on fresh generated answers.
- Main Table 1 gives Gamma HH win rates 58.24 versus DPO 69.29 for Pythia and 54.85 versus DPO 71.58 for Llama. Thus broad language about no aggregate tradeoff does not hold uniformly across variants/settings.
- Main and appendix tables have differing baseline values; these should not be merged into one experiment without explanation.
- Appendix C's derivation has indexing/parameter typographical inconsistencies relative to the main Gamma formula. The correct main expression for positive $t$ is $1-(\lambda/t)^k\Phi(-1,k,1+\lambda/t)$ under the conventional Lerch definition starting at zero.
- At argument $-1$, alternating convergence can hold for positive shape even though the paper's general absolute-convergence wording invokes a stronger exponent condition. A 1,000-term empirical error at one setting is not a uniform numerical guarantee near zero margin or over all learned parameters.
- No finite-sample identification theorem, latent-strength consistency theorem, subgroup regret guarantee, or uncertainty interval for learned heterogeneity is established.

### 2. Rating DPO: additional information can remove saturated comparison ambiguity

Source: [Direct Preference Optimization with Rating Information](https://arxiv.org/abs/2602.00603), Sections 2–4, Appendices B–F.

#### Three objectives and their different targets

The paper assumes a cardinal latent reward $r^*$ generating BT comparisons, $P(Z=1\mid x,a,b)=\sigma(\Delta r^*)$. Ratings provide a gap $g$ in approximately the **same reward units**. Write the policy/reference difference as $d_\theta$.

$$
\mathcal L_{\mathrm{RDPO}}=-\sum_i\log\sigma\left(\beta d_\theta^i-\frac{\beta}{\beta_1}g_i\right).
$$

$$
\mathcal L_{\mathrm{RIPO}}=\sum_i\left(\beta d_\theta^i-\frac{\beta}{\beta_1}g_i-\frac12\right)^2.
$$

$$
\mathcal L_{\mathrm{ML-RDPO}}=\sum_i\left[-\log\sigma(\beta d_\theta^i)+\frac{(g_i-\beta d_\theta^i)^2}{2V}\right].
$$

RDPO follows from an RLHF objective with reward $r+(\beta/\beta_1)\widehat r$. Its gap appears as a **subtracted margin** in the contrastive loss: a larger supplied gap asks the policy to separate the pair more strongly. ML-RDPO instead adds a Gaussian rating likelihood to a BT ranking likelihood. These are different constructions: shifting a likelihood versus combining measurements of the same latent gap.

**Intuition:** a nearly deterministic win says little about whether the reward difference is 5 or 50, because both give almost identical BT probabilities. A cardinal gap can measure that difference without waiting for extremely rare preference reversals. The acceleration comes from an added measurement channel, not a clever reweighting of the original bit.

#### What the theorems assume and actually bound

Assumptions 4.1–4.2 require i.i.d. candidate sampling from a data policy and a policy class expressive enough to realize the Gibbs solution for every bounded reward in scope. Optimization is treated as exact empirical minimization in a restricted policy class. Finite classes are written through $\log|\Pi|$; extending to continuous classes needs covering control. The results do not establish these assumptions for a finite neural model.

Define

$$
E_{\mathrm{rat}}=\mathbb E_{x,a,b\sim\pi_{\mathrm{data}}}[(\Delta r^*-\Delta\widehat r)^2],\qquad
E_{\mathrm{rank}}=O\left(\frac{R^2e^{4R}\log(|\Pi|/\delta)}{N}\right).
$$

Theorem 4.3 tunes $\beta_1=\beta E_{\mathrm{rat}}/E_{\mathrm{rank}}$, using unknown population errors, and bounds a regularized policy gap by order

$$
\sqrt{C_{\max}\min(E_{\mathrm{rank}},E_{\mathrm{rat}})}.
$$

The target's effective KL temperature is

$$
\beta'=\frac{\beta\beta_1}{\beta+\beta_1}.
$$

This is essential: an oracle-tuned robustness statement about $J_{\beta'}$ is not automatically a comparison with ordinary DPO at the original $\beta$. Appendix C.2 makes this change explicit using a convex combination of the ranking and rating reward estimators.

ML-RDPO Theorem 4.4 retains the original $\beta$ and assumes, conditional on the unordered prompt/candidate pair, independent ranking bits and Gaussian rating gaps centered at the true cardinal reward gap. Its main statement has a reward-range factor written as a minimum of an exponential term and $R^2+V$, times $\log|\Pi|/N$, followed by a square root and concentrability. Appendix C.4 displays the concrete conservative squared-gap estimate

$$
E_{\mathrm{out}}\le
\min\{16e^{4R}R^2,\;2(R_{\max}-R_{\min})^2+4V\}
\frac{2\log(|\widetilde\Pi|/\delta)}{N}.
$$

The main statement prints an $e^R$ term whereas the displayed proof uses $e^{4R}$. Both express exponential-versus-polynomial improvement, but this exponent discrepancy should be resolved before reproducing the theorem literally; a big-O constant cannot generally absorb an extra exponential in an unbounded range parameter.

**Statistical rate:** policy-gap order is generally $N^{-1/2}$ in these displayed bounds; the claimed acceleration is a better dependence on reward range or an accurate external rating estimate. It is not a generic change from $N^{-1/2}$ to $N^{-1}$.

Appendices C.3 and C.5 add KL plus chi-squared regularization, require data policy equal reference, and use oracle tuning involving error and optimal-policy concentrability to replace all-policy coverage with optimal-policy coverage. The practical chi-squared variant can be numerically unstable according to the main text.

#### Missing ratings are already covered

Appendix B.2 and Appendix D explicitly consider nonoverlapping/partly overlapping ranking and rating datasets. RDPO first learns a rating estimator on available gaps and imputes the missing gaps. ML-RDPO directly sums the ranking and rating losses over their separate datasets. Appendix D.2 models independent missingness with probabilities $p_{\mathrm{rat}},p_{\mathrm{rank}}$ and gives a minimum of the channel-specific error bounds divided by their observation probabilities. Therefore, simply adding sparse ratings or handling partially scored data is not novel.

Appendix C.2 also recognizes dependence when ranking and rating come from the same source and uses covering arguments for the RDPO reward class. This does **not** remove the separate Gaussian/conditional-independence assumption used for the ML-RDPO joint likelihood.

#### Practical limitations and actionable gaps

1. A Likert gap or evaluator score gap need not be cardinally calibrated to the BT reward. Unknown annotator-specific affine or monotone transformations violate the measurement assumption even if ordering is accurate.
2. Ranking deterministically derived by comparing the same scores is not independent of the scores. Correlated likelihood factors should not be interpreted as two independent measurements.
3. The oracle trust coefficient depends on unknown true rating error. A hyperparameter sweep on one model is useful empirical tuning, but not an adaptive oracle theorem.
4. Coverage, expressivity, optimization, and neural parameter coupling are not eliminated by adding ratings.
5. Figure 1 variation across three reference model choices is not three training seeds of one model. Appendix F distinguishes evaluation-sample error bars from variation across references; neither establishes training-seed robustness.
6. Main experiments include score swaps and zero-mean Gaussian noise, but not adversarial monotone scale distortions or annotator-dependent scoring functions. Those are targeted tests of the new calibration problem.
7. Small-model optimal trust differs from 7B/8B trust; Appendix F.4 explicitly notes that current theory neglects optimization and does not explain model-size dependence.

### 3. CATTO: confidence calibration claims require the printed loss to be proper

Source: [CATTO](https://arxiv.org/abs/2601.23096), Sections 3–4, Appendix A and E.

#### Mechanism and intuition

CATTO computes top-token confidence $c=\max_y p_y$, a ground-truth-versus-best-other probability margin, and a differentiable target

$$
\widetilde z=\sigma(p_{y^*}-\max_{y\ne y^*}p_y).
$$

Its actual Equation (6) uses

$$
L_{\mathrm{CATTO}}(c,\widetilde z)=\widetilde z(1-c)+(1-\widetilde z)c.
$$

This token-averaged term is added to DPO on chosen and rejected responses, with the target inverted on rejected responses in Section 3.2. Confidence@k selects the candidate with the largest reported correctness confidence.

**Intuition intended by the authors:** pairwise preference training can improve answer ranking while inflating absolute confidence. A token signal tied to which answer is correct is meant to constrain that inflation during training instead of repairing it afterward. The idea addresses a real distinction between preference quality and correctness confidence. Whether the proposed surrogate estimates probabilities is a separate mathematical question.

#### Exact proposition C1: the written CATTO loss is not a proper probability loss

Holding $z\in[0,1]$ fixed,

$$
L(c,z)=z+c(1-2z),\qquad
\operatorname*{arg\,min}_{c\in[0,1]}L(c,z)=
\begin{cases}
\{0\},&z<1/2,\\
[0,1],&z=1/2,\\
\{1\},&z>1/2.
\end{cases}
$$

For $z=0.7$, $L(0.7,0.7)=0.42$ whereas $L(1,0.7)=0.30$. The optimum is not the correctness probability 0.7. This directly contradicts Appendix E.1.3's claimed unique minimizer $c^*=z$ for the displayed affine loss.

The appendix partially recognizes that binary-target L1 regression seeks a median (E.2.1). However, its subsequent robustness argument studies **another loss**, $|c-\widetilde z|$ on a continuous target, whose minimizer is $\widetilde z$. That statement does not establish the calibration consistency or median robustness of Equation (6). Bounded gradients for an affine loss do not make it a strictly proper scoring rule.

The target is also endogenous: it changes with the same logits producing $c$. A derivative holding the target fixed is not the full training derivative unless a stop-gradient is actually used. Appendix E.11 assumes a fixed target while the main method defines a model-dependent target; this needs implementation confirmation.

The impropriety persists in an exact endogenous-target example. Consider binary labels with true $P(Y=1)=q>1/2$ and model probability $p\ge1/2$, so top confidence is $c=p$. For label 1, set $s=\sigma(2p-1)$ and $L_1=s+p-2sp$; for label 0, the opposite margin gives $L_0=1-L_1$. Therefore the expected CATTO term is $(1-q)+(2q-1)L_1$. Its derivative is negative for $p>1/2$ because $dL_1/dp=1-2s+2s(1-s)(1-2p)<0$. Its optimum is $p=1$, not $p=q$. At $q=0.7$, the posterior-calibrated $p=0.7$ is not the minimizer. Adding DPO can change the joint optimum; this counterexample establishes that the calibration term itself is not a proper probability estimator.

The probability margin lies in $[-1,1]$, hence $\widetilde z\in[\sigma(-1),\sigma(1)]\approx[0.269,0.731]$. Appendix E.3.3 says $\widetilde z\to1$ when the correct-token probability tends to one. For the printed target the limit is $\sigma(1)\approx0.731$, so that approximation-gap claim is false.

#### What remains valid, and what does not follow

- A population bound $\mathrm{ECE}\le\mathbb E|C-Z|$ follows from conditional Jensen. The bound is loose because it includes irreducible outcome noise. Minimizing this upper bound need not minimize ECE: for $Z\sim\mathrm{Bernoulli}(0.7)$, constant confidence 0.7 has zero ECE, while confidence 1 minimizes expected absolute binary error and has ECE 0.3.
- The sigmoid margin has the sign of whether the correct token beats its best competitor. Sign consistency is a ranking/correct-class condition, not identification of a correctness probability.
- Proposition A.3 is a bounded scalar perturbation argument conditional on existing positive preference margins. Turning log-probability partial derivative bounds into neural parameter update bounds requires learning rate and cross-Jacobian control. It is not a global guarantee that joint training preserves every pair ordering.
- Proposition A.4 assumes $c(x,y)=P(y\text{ correct}\mid x)$ exactly. Its argmax conclusion is valid under that posterior assumption. Small marginal ECE does not establish it.

#### Exact proposition C2: zero ECE need not yield Bayes-optimal candidate selection

Suppose two equally frequent prompt strata each offer candidates A and B. Report confidence 0.8 for A and 0.7 for B in both strata. True correctness probabilities are $(0.6,0.9)$ in stratum 1 and $(1.0,0.5)$ in stratum 2. Both confidence levels are marginally calibrated: A averages 0.8, B averages 0.7. Confidence argmax always chooses A with accuracy 0.8. A selector using the prompt stratum chooses B in stratum 1 and A in stratum 2, achieving 0.95. Thus zero marginal ECE permits a 0.15 selection regret.

This is a standard distinction between marginal calibration and conditional decision sufficiency. It provides a diagnostic, not novelty by itself.

#### Empirical boundary

Section 4 explicitly restricts the study to class/choice labels predicted as next tokens, allowing probabilities to correspond to a verifiable label. Open-ended semantic correctness and sequence correctness are left for future work. The base model is Qwen3-4B; data include reformulated multiple-choice settings and synthetic reasoning traces. ECE/accuracy gains over DPO across tested settings are empirical observations and remain meaningful even though the stronger consistency proof has the gaps above.

The out-of-distribution setting is MMLU training followed by other benchmark tests, not all possible shifts. Five random data splits are reported. Confidence@k also uses temperature selection and $k\in\{4,8\}$; sampling, calibration, and selection effects should be disentangled. Low ECE alone should not be used as evidence that generated free-text answers have reliable semantic truth probabilities.

### 4. ECLIPTICA/CITA: policy switching and the reference-anchor mismatch

Source: [ECLIPTICA](https://arxiv.org/abs/2601.06157), Sections 2–4, 6–7, Appendices C, D, E, F, L, N.

#### Contribution and intended mechanism

ECLIPTICA fixes the user prompt and varies an alignment instruction: 300 prompts, ten instructions, 3,000 cases. This paired design isolates the effect of the supplied instruction on a fixed model's output distribution more cleanly than comparing unrelated prompts. Instructions cover style, stance, and constraints, synthesized from preference examples using five model judges plus a semantic-agreement filter and human quality gate.

Main Figure 3 and Appendix C define

$$
L_{\mathrm{CITA}}=\mathbb E[-\log\sigma(\beta\Delta_\theta(I,x))]
+\lambda\mathbb E_{I,x}\mathrm{KL}(\pi_\theta(\cdot\mid I,x)\|\pi_0(\cdot\mid I,x)),
$$

where the main gap is the raw conditional log-likelihood difference, not the usual policy/reference difference. Appendix C derives the familiar logistic gradient, conditional forward-KL score-function gradient, Fisher quadratic expansion, and local natural-gradient step

$$
\delta\theta^*=-\lambda^{-1}(\mathbb E F_{I,x})^{-1}g.
$$

**Intuition:** one model should adapt its behavior to an explicit policy instruction while the task stays fixed. A reference constraint is intended to keep each instruction-conditioned behavior near a stable starting policy. This is conditional preference learning plus a local trust-region argument, not an identification theorem for unobserved human values.

#### Exact proposition E1: the described offline anchor is not the theoretical KL

Appendix D, Figure 14 and the subsequent Unified Loss Function instead describe the implemented term as the average of policy/reference log ratios on **fixed chosen and rejected data**:

$$
A(\pi)=\mathbb E_{Y\sim q}\log\frac{\pi(Y)}{\pi_0(Y)}
=\mathrm{KL}(q\|\pi_0)-\mathrm{KL}(q\|\pi).
$$

This identity holds for a fixed offline empirical response distribution $q$. It is not forward KL, is not generally nonnegative, and its minimization favors decreasing likelihood under the empirical distribution.

For $q=\pi_0=(1/2,1/2)$ and $\pi=(p,1-p)$:

$$
A(\pi)=\frac12\log[4p(1-p)]\le0.
$$

It is zero at the reference and tends to negative infinity as $p\to0$ or $p\to1$. Therefore the printed offline anchor does not inherit the Fisher trust-region interpretation of the main forward-KL objective. The discrepancy must be resolved with code or author clarification before explaining empirical gains as consequences of that KL geometry.

There is also an unbounded-likelihood-displacement direction for the full printed implementation objective. Let the reference be uniform over three responses and let the sole observed preference compare responses 1 and 2. Set $\pi=(\epsilon,\epsilon,1-2\epsilon)$. The reference-adjusted DPO pair margin stays zero, so the DPO term stays $\log2$, while the average offline anchor becomes $\log(3\epsilon)$. For every positive anchor weight, the total objective tends to negative infinity as $\epsilon\to0$. Thus this literal objective rewards transferring mass out of both observed responses into an unobserved response. This algebraic finding concerns the printed formula; bounded parameterization, clipping, or a different implemented estimator could change actual training behavior and must be checked.

The same appendix says training stacks on a DPO checkpoint and reuses TRL's reference-adjusted DPO loss; the main presentation says a shared SFT branch and uses a raw log-likelihood gap. Section 4 also mentions an SFT component, whereas Appendix D explicitly excludes it after reported interference. These are substantive specification differences. They should not be collapsed into a single verified mathematical recipe.

#### Exact proposition E2: a trust region does not eliminate instruction interference

With instruction gradients $g_i$, weights $w_i$, and positive definite aggregate Fisher $F$, the local anchored step is

$$
\delta=-\lambda^{-1}F^{-1}\sum_jw_jg_j.
$$

The first-order change for instruction $i$ is

$$
g_i^\top\delta=-\lambda^{-1}\sum_jw_jg_i^\top F^{-1}g_j.
$$

Cross inner products can have either sign. In one dimension, $g_1=1,g_2=-2,w_1=w_2=1/2,F=1$ gives $\delta=1/(2\lambda)$ and **increases** instruction 1's loss. KL reduces step size; it does not prove all regime losses decrease or prevent collapse under arbitrary incompatibility and shared capacity. The familiar logistic gradient's saturation also does not itself bound total margin growth over infinitely many steps.

#### Metric and empirical boundaries

- Appendix E defines ECLIPTICA as embedding Fidelity times embedding Shift. Distinct wording/structure can affect these quantities; this is not automatically satisfaction of a behavioral contract.
- Conditional Safety is the absolute difference between two refusal rates. The paper explicitly acknowledges in Section 6 and Limitations that refuse-always versus comply-always can score highly; it is a mode-separation diagnostic, not a safety guarantee.
- Length metric is a ratio, not simultaneous verification of at-most-50 and at-least-200 word constraints. All main ratios remain below the stated target of four.
- TruthfulQA measures instruction-dependent epistemic stance. It is not a proper-scoring probability calibration assessment. Appendix L gives wide confidence intervals around the small positive scores; the 54-times ratio compares a small difference with a near-zero denominator.
- The aggregate 86.7% is a normalized radar score, not fraction of prompts correctly handled. Appendix N and main table report inconsistent AQI deltas; normalization conventions need checking before reconstructing the aggregate.
- AQI is written as $(\mathrm{CHI}+\mathrm{XB})/2$. Standard Xie–Beni is a compactness/separation ratio with smaller values preferred, whereas CHI is usually larger-is-better. No direct addition of raw indices establishes correctness, and the claimed 0–100 normalization requires an explicit definition. Treat AQI as an author-defined proxy pending implementation verification.
- Primary model is Llama-3.1-8B, English only, one training dataset according to Limitations. Main table has DPO ahead on ECLIPTICA and conditional-safety separation, while CITA leads on other proxies. This supports a profile of tradeoffs, not uniform dominance.
- Holding the prompt fixed identifies the effect of instruction text in this test protocol. It does not alone identify the effect of the training algorithm: prealignment, sampling, and hyperparameter selection must also be matched. The study includes Instruct/NoInstruct baselines, which is a useful partial control.

### Part IV — Generalization, domain shift, tangent control, steering, tokens and online optimization

### 1. Towards a Theoretical Understanding to the Generalization of RLHF

Source: [arXiv:2601.16403v1](https://arxiv.org/abs/2601.16403). Main technical locations: Definitions 1–7, Assumptions 1–5, Theorems 1–7, Appendix A Lemma 6, Appendix B Lemma 11 and Proposition 4, Appendices D–H.

#### Mechanism and intuition

The paper studies a known, realizable reward and an exponential-family policy using the *same* feature map. It does not estimate a reward from noisy human comparison labels. The observation sampled in its statistical analysis is a prompt, while the objective integrates over actions with the ground-truth reward available.

With KL coefficient rescaled to one, the setup is

$$
r^*(x,a)=\theta^{*\top}\phi(x,a),\qquad
\pi_\theta(a\mid x)=\frac{\pi_{\mathrm{ref}}(a\mid x)e^{\theta^\top\phi(x,a)}}{Z_\theta(x)},
$$

$$
J_S(\theta)=\frac1n\sum_{i=1}^n
\left\{\mathbb E_{\pi_\theta}[r^*(x_i,a)]-
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\right\}.
$$

The key identity, Lemma 11 / Eq. (44), is

$$
\nabla J_S(\theta)=V_S(\theta)(\theta^*-\theta),\qquad
V_S(\theta)=\frac1n\sum_i\operatorname{Cov}_{\pi_\theta(\cdot\mid x_i)}[\phi(x_i,a)].
$$

**Intuition:** gradient ascent corrects the error only along directions in which sampled prompts allow the policy to change its response probabilities. Errors in the covariance null space cannot be corrected, but are also invisible on the training prompts. Generalization depends on whether they become visible elsewhere. Covariance is the policy's sensitivity, not the empirical covariance of arbitrary input tokens.

Proposition 4 exploits positivity of policy probabilities: every empirical stationary point has exactly the oracle policy on the sampled prompts, hence is a global empirical optimum. This structure is substantially stronger than generic nonconvex neural-network stationarity.

#### Actual assumptions and statements

- Assumption 1: bounded features, $\|\phi\|\le C$.
- Assumption 2: realizable linear *ground-truth* reward, $\|\theta^*\|\le D$; analyzed iterates lie in a radius $R=3D$ region.
- Assumption 3: the population policy covariance is uniformly positive definite, $\mathbb E_x V_x(\theta)\succeq\sigma I$.
- Theorem 1: with sufficiently many prompts, the empirical stationary point uniquely equals $\theta^*$. Its displayed threshold is

$$
n>\frac{128C^4}{\sigma^2}
\left[\log\frac1\rho+d\log\frac{432RC^5}{\sigma^2}\right].
$$

- Assumption 4: for each prompt, action features are linearly independent, with the number of actions at most the ambient dimension. The main text has a footnote adjacent to $d$, which must not be interpreted as an exponent $d^2$. The proof explicitly uses $d\ge|\mathcal A|$.
- Assumption 5: uniformly over prompts, actions, and training sets, raw features admit

$$
\phi(x,a)=V_S(\theta_S^*)b_S(x,a)+r_S(x,a),\qquad
\sup_{x,a}\|r_S(x,a)\|\le\epsilon_n=O(n^{-1}).
$$

- Theorems 2–3 report stationary-point stability and suboptimality of order $\widetilde O(n^{-1/2})$.
- Theorems 4–5 report GA order $\widetilde O(T^{-1/4}+n^{-1/2})$.
- Theorems 6–7 report SGA order $\widetilde O(T^{-1/8}+n^{-1/2})$.
- These rates hide ratios involving the smallest *positive* empirical covariance eigenvalues. They are not unconditional dimension-free constants. The GA/SGA output is the iterate with the smallest full empirical gradient norm, rather than necessarily the final iterate. SGA selection requires full-gradient evaluation, acknowledged in Remark 2.

#### Deep insight and limits

1. The analysis separates parameter identifiability from policy identifiability. That is valuable, but the known-reward assumption removes the hardest preference-learning uncertainty.
2. The $O(1/n)$ *uniform* residual assumption supplies substantial coverage rather than proving it. The simulation's expanding random subspace does not, by itself, establish that every possible unseen feature has residual $O(1/n)$ uniformly over all training sets.
3. Raw features may contain prompt-only coordinates. Such coordinates never change a policy and never belong to its response covariance range. Requiring them to be covered is unnecessarily strong. For example, $\phi(x,a)=(\psi(x,a),c(x))$ has a policy-invariant second coordinate, even if that coordinate's raw residual remains nonzero forever. Contrast features or a quotient by prompt-only shifts are the appropriate identifiable objects.
4. “More optimization improves generalization” follows from the special identity above, exact reward access, and the feature assumptions. It cannot be imported as a general guarantee for neural RLHF with an estimated or hacked reward model.
5. Some printed proof equations need correction before reuse. For an exponential family the correct KL identity is

$$
D_{\mathrm{KL}}(\pi_{\theta_1}\Vert\pi_{\theta_2})=
(\theta_1-\theta_2)^\top\mathbb E_{\pi_{\theta_1}}\phi
-\log Z_{\theta_1}+\log Z_{\theta_2}.
$$

The extracted Eq. (32) has the opposite signs on the log-partition terms. Several displayed theorem/proof differences also reverse the nonnegative suboptimality direction. These are source issues to verify against PDF typography, not a license to quietly reinterpret the paper's theorem as proven for a different setting.

#### Empirical evidence

Only synthetic linear-feature experiments are reported: a rare-basis coverage example ($d=20$, 10,000 Monte Carlo trials) and rank-deficient feature simulations. Appendix H.2 uses exact stationary points, GA with $50n^2$ steps, SGA with $10n^4$ steps, and a 15,000-prompt test approximation. These test the constructed geometry, not noisy preference estimation, neural LLM training, or domain adaptation with changed human utilities.

### 2. An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift

Source: [arXiv:2601.05882v2](https://arxiv.org/abs/2601.05882), dated 2026-08-24. Locations: Sections 3–6, Tables 1–5, Appendix A, D–H.

#### Mechanism and intuition

This is a factorial empirical comparison of the adaptation stage and alignment objective, not a new optimization theorem. It studies DPO, KTO, ORPO, PPO, GRPO, and SFT across source-only, mixed SFT, target SFT, direct alignment, and target pseudo-labeling configurations.

**Intuition:** learning how the target domain expresses good answers can matter more than choosing another preference loss. A strong teacher can supply this missing domain information, but can also transmit one recurring writing template. Better judge scores and less response variety can therefore occur together.

The transfers are Reddit TL;DR to CNN/DailyMail, AskEngineers to AskCulinary, and Cybercrime to Violence/Physical Harm in PKU-SafeRLHF. Backbones are base Llama-3.1-8B and OLMo-3-7B. Target pseudo-labels use Llama-3.3-70B-Instruct; three nonempty teacher candidates per prompt are each declared preferred to the original response. This is a particular asymmetric distillation construction, not an oracle label that independently ranks arbitrary target response pairs.

#### Findings worth retaining

- Adaptation effects are large, heterogeneous, and model dependent. Llama source-only DPO summarization scores 89.87 source versus 58.09 target; target pseudo-labeled SFT scores 95.70 / 83.37. OLMo target pseudo-labeled DPO has 80.16 / 72.26 on summarization.
- A source pseudo-label control partly isolates teacher quality from target relevance: for Llama DPO, target synthetic data yields target 78.50 versus 62.88 for source synthetic data (Table 2).
- Under pseudo-labeling, semantic diversity is 0.07–0.08 for the offline objectives versus 0.22 PPO and 0.25 GRPO (Table 12); syntactic and logical measures do not have identical interpretations.
- On OLMo safety, pseudo-labeled SFT is 99.78 / 95.64, while subsequent DPO is 74.52 / 63.14. Thus stronger supervision does not uniformly survive later preference optimization.
- Standard QA judging masks some pragmatic mismatch; changing to a domain-aware judge changes model rankings and margins.

#### Limits and confounds

1. Source-minus-target win rate is not a pure generalization error: the two domains have different difficulty and human references, as the base-model gaps already show. A small gap can result from source forgetting or weak performance on both domains.
2. The synthetic preferred response is from a strong instruction model, whereas its rejected partner is an original dataset response. This couples domain knowledge, teacher style, length, reference quality, and label construction. The source synthetic control helps, but does not remove these factors or make the teacher's style literally identical across domains.
3. Results use fixed seeds and final checkpoints; the text does not establish repeated-training confidence intervals for the method rankings. Sampling several generations does not substitute for independent training seeds.
4. Diversity is measured on summarization for Llama only. SBERT variation, n-gram variety, logical contradiction, factuality, and pluralistic preference coverage are different quantities.
5. Safety is the fraction not flagged by a moderation service. Universal refusal can score highly. Benign overrefusal/helpfulness is not identified by this statistic.
6. The human judge check is 60 items per domain, three annotators per item, for QA only, excluding ties from reported agreement. It does not validate summarization, factuality, or moderation outcomes across all experimental conditions.
7. The new domain-aware judge explicitly requests a culinary persona and avoidance of engineering jargon. This measures the supplied target norm, not objective proof that every technically detailed response is inappropriate.
8. Online/offline compute, reward models, optimization rates, rollouts, and initialization differ. “Online sampling causes transfer” remains a mechanism hypothesis, not a controlled causal result.

#### Contribution to theory planning

The observations motivate separate terms for missing target response information, reward/label distortion, policy-class projection, and superficial diversity. A theorem using only a scalar covariate density ratio cannot explain a changed conditional preference law. The proposed theory must explicitly distinguish prompt covariate shift from preference-concept shift.

### 3. Tangent Space Fine-Tuning for Directional Preference Alignment in LLMs

Source: [arXiv:2602.01128v1](https://arxiv.org/abs/2602.01128). Locations: Section 3 / Algorithm 1, Section 4, Appendix A Table 1.

#### Mechanism and intuition

The paper trains a frozen-base JVP model rather than only assuming that conventional fine-tuning is approximately linear:

$$
f_{\mathrm{lin}}(x;d)=f(x;\theta_0)+J_{\theta_0}(x)d.
$$

Separate DPO-trained directions $\tau_h,\tau_v$ are combined through $d=\lambda_h\tau_h+\lambda_v\tau_v$. Training uses the last 16 transformer layers and language-model head of Llama-3.2-1B-Instruct. Each axis has 6,000 training and 2,000 evaluation examples from UltraFeedback and HelpSteer2. The comparison includes ordinary task-vector DPO and concatenated-data DPO-Mixed.

**Intuition:** make the network's local response to an update additive by construction. Then turning two parameter knobs cannot introduce the same network-output curvature as merging ordinary fine-tuned checkpoints. This can stabilize control, but it does not establish that the knobs equal the user's desired utility tradeoff.

#### What is and is not linear

Token logits can be linear while sequence log probabilities remain nonlinear:

$$
\log\pi_d(y\mid x)=\sum_t
\left[z_{0,y_t}(x,y_{<t})+J_{y_t}(x,y_{<t})d
-\log\sum_v e^{z_{0,v}(x,y_{<t})+J_v(x,y_{<t})d}\right].
$$

Two different completions generally have different prefixes, so their partition terms do not cancel. Consequently an exact JVP network does not automatically give linear DPO sequence margins. Expected helpfulness, safety, and verbosity are also nonlinear functions of the induced probability distribution. Parameter orthogonality is not orthogonality of behavioral effects.

The full-policy exact composition identity is instead conditional on genuine common-temperature reward realization:

$$
\pi_\alpha^*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)
\prod_k\left(\frac{\pi_k^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right)^{\alpha_k}.
$$

It follows when $\pi_k^*\propto\pi_{\mathrm{ref}}\exp(r_k/\beta)$ and $r_\alpha=\sum_k\alpha_k r_k$. It need not hold for *restricted-class optima* or estimated implicit rewards. This identity is standard; it is not this handoff's proposed novelty.

#### Evidence and limits

- The reward-model generation frontier uses only 100 HelpSteer2 prompts, greedy decoding, and a reward model trained on related source datasets. No broad behavioral guarantee follows.
- Pairwise accuracy averages token log probabilities. That length normalization differs from a sequence-likelihood DPO margin.
- DPO-Mixed gives one concatenated-data balance at each learning rate. It is not a fully tuned preference-conditioned multiobjective baseline or a separate scalarization sweep at every target tradeoff.
- The closest DPA baseline is absent. More than two objectives and larger models are not evaluated.
- The paper acknowledges not recovering all tradeoff regions. CCA decay and near-zero parameter cosines provide descriptive geometry; they do not prove independence, causality, Pareto completeness, or user-weight calibration.
- JVP training costs are about 3.5 versus 2.5 hours per objective in the stated V100 setup. Inference KV-cache support is absent. The text alternates linearized evaluation and parameterized $\theta_0+d$ language, so reproduction must verify whether generation actually retains the JVP model rather than evaluating the original nonlinear network at merged weights.
- There is no theorem package proving utility linearity, composition optimality, or OOD controllability. Treat this as a small empirical method paper.

### 4. YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation

Source: [arXiv:2601.08441v1](https://arxiv.org/abs/2601.08441). Locations: Eqs. (1)–(6), Algorithm 1, Tables 1–5, Appendices A–F.

#### Mechanism and intuition

YaPO moves BiPO's trainable intervention into a frozen sparse-autoencoder latent space. Its activation intervention is

$$
\Phi(h,\lambda,d,v)=
\operatorname{Dec}\!\left(\operatorname{ReLU}(\operatorname{Enc}(h)+d\lambda v)\right)
+h-\operatorname{Dec}(\operatorname{Enc}(h)).
$$

The residual correction makes the zero intervention preserve the original activation when encoder codes are already nonnegative. LLM and SAE weights stay frozen; only $v$ is trained. Random $d\in\{-1,1\}$ trains both steering directions against relative response likelihoods.

**Intuition:** instead of pushing every neuron in a dense direction, use a pretrained dictionary intended to separate concepts, then learn which dictionary coordinates alter preference behavior. Preserve the activation part the SAE reconstructs poorly so that reconstruction error is not mistaken for steering.

#### Deep issues requiring implementation reconciliation

1. Eqs. (1) and (4) print minimization of $\mathbb E\log\sigma(\cdot)$ without a minus sign. That would optimize in the opposite direction. The narrative and decreasing positive loss plots imply intended negative log-likelihood. Verify released code before implementing.
2. “Reference-free” means no separate reference checkpoint, but the objective still divides by the frozen model's unsteered completion likelihood.
3. SAE activations are sparse; the displayed training objective does not enforce an explicit sparsity constraint on the learned 65k/131k-coordinate vector $v$. ReLU enforces nonnegativity, not a guaranteed sparse support. Adding dense $v$ can activate many coordinates. Dictionary coordinates are not automatically causally disentangled.
4. Bidirectionality is not exactly antisymmetry when ReLU clips coordinates at zero. The positive and negative activation changes can differ substantially.
5. For per-instance binary $p_{\mathrm{loc}},p_{\mathrm{non}}$, the Eq. (6) harmonic expression is essentially an indicator that *both* answers are correct. This is not the harmonic mean of the two aggregate accuracies. The actual metric code must settle which quantity Tables 3/9 compute. PNLG is signed; negative values are possible and do not automatically imply better cultural competence.
6. General-task rubrics in Appendix E/F use 4 for more hallucination, harmful compliance, power seeking, or wealth seeking and 1 for less. A higher Table 5/14 mean cannot be described as improved safety without resolving score orientation. The claims of generalization to other tasks are broader than the evidence of reduced undesirable behavior.
7. The headline counts say fifteen cultural contexts, while the country list and Appendix F's statistics contain fourteen groups including Levantine. Table 15 is exactly 22,677 localized plus 22,677 nonlocalized items, despite prose saying 57.7% versus 42.3%. Dataset bookkeeping needs code verification.

#### Empirics and limits

Gemma-2-2B and 9B use available Gemma-Scope SAEs. Layer selection uses localized-to-nonlocalized activation patching on Egypt/Morocco. The cultural dataset's responses are generated with Gemini-2.5-Flash, filtered and manually reviewed; country norms are not a validated universal individual preference distribution. Dialect-specific answer choices can support lexical shortcuts. Separate topic/template splits are important.

YaPO improves many 2B MCQ results, but not every country/task. Portuguese OG favors BiPO. At 9B Arabic MCQ averages are BiPO 29.5 versus YaPO 28.1, and localized OG is SAS 6.15 versus YaPO 5.91. The 9B tuning budgets differ: CAA/SAS are tuned, BiPO/YaPO are fixed to $\lambda=1$. MMLU changes are small, but no statistical equivalence claim follows without intervals. There is no formal generalization or global composition theorem.

### 5. SP2DPO: An LLM-assisted Semantic Per-Pair DPO Generalization

Source: [arXiv:2601.22385v1](https://arxiv.org/abs/2601.22385). Locations: Eqs. (5)–(18), Propositions 1–2, Tables 2–3, Appendix D.2, E, L. The large prompt/engineering appendices were not visually audited.

#### Mechanism and intuition

Three teachers (Qwen-Max, GPT-5-mini, Gemini-2.0-Flash) and three rubric variants assign category, magnitude, and confidence offline. Magnitude times confidence is aggregated by median across prompts and mean across teachers, then mapped to $\beta_i\in[0.03,0.3]$. Category is used for auditing, not directly in the scalar mapping.

$$
L_i(\theta)=\operatorname{softplus}(-\beta_i\Delta_i(\theta)),\qquad
\nabla L_i=-\beta_i\sigma(-\beta_i\Delta_i)\nabla\Delta_i.
$$

**Intuition:** attach an example-specific response curve to each comparison before training. Near zero margin, a larger coefficient supplies a stronger gradient; once a positive margin is large, it also saturates faster. This changes where training effort is concentrated rather than monotonically making every important pair stronger.

#### What the formulas justify

For a genuinely varying margin, changing the coefficient inside the sigmoid generally cannot be replicated by one *fixed* outside weight. But instantaneous gradients can be matched by a policy-dependent weight

$$
w_i(\theta)=\frac{\beta_i\sigma(-\beta_i\Delta_i(\theta))}
{\bar\beta\sigma(-\bar\beta\Delta_i(\theta))}.
$$

The paper's stronger aggregate non-equivalence proof assumes coefficients of all margin gradients must separately agree, which requires conditions beyond three distinct margins at a parameter setting. Constant margins can have zero gradient; dependent gradients can combine. The intended single-pair observation is valid, but the stated aggregate proposition needs additional assumptions. Appendix D.2's “informal” margin-generalization theorem gives neither a concrete bound nor a complete proof that changing boundary mass lowers it. It is not a usable finite-sample guarantee.

Per-pair $\beta_i$ also cannot generally be interpreted as one global KL regularization coefficient. If observed preference logits are $\ell_{ab}$, fitting a potential requires

$$
u_a-u_b=\ell_{ab}/\beta_{ab},\qquad
\ell_{ab}/\beta_{ab}+\ell_{bc}/\beta_{bc}+\ell_{ca}/\beta_{ca}=0
$$

on every triangle with consistent orientation. An arbitrary semantic schedule can break this even when the original logits came from a perfectly consistent shared-temperature reward. Thus this is a heterogeneous classifier geometry, not automatically the optimizer of a coherent scalar reward.

#### Evidence and limits

59,960 successful UltraFeedback annotations train four instruction backbones. The joint robust default does not beat tuned global DPO in any available primary length-controlled result: Qwen 6.82 versus 7.45, Gemma 41.02 versus 41.08, Llama 34.19 versus 34.80; Mistral joint result is absent. Best-by-evaluation configurations improve Gemma by 1.07 and Qwen by 0.10, but trail Mistral by 1.34 and Llama by 0.61. Selecting among many teachers/prompts using the reported evaluation is itself a hyperparameter search. “No per-model beta sweep” does not mean no model-specific selection cost.

The training inner loop has little additional overhead, but approximately nine teacher calls per annotatable pair is substantial offline cost. No rare-safety, target-domain, human-judge, or policy-optimality theorem is established by AE2 alone. Reproduction needs teacher versions, raw responses, failed-call handling, dataset revision, and a separately held-out configuration selection split.

### 6. ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection

Source: [arXiv:2601.09195v3](https://arxiv.org/abs/2601.09195), dated 2026-05-06. Locations: Eqs. (2), (4)–(6), Theorem 1, Table 1–3, Sections 6.1–6.6, Appendix A–B.

#### Mechanism and intuition

ProFit uses the current model's detached probability of a teacher-forced reference token to retain only tokens with $p_t>\tau$:

$$
M_t=\mathbf1\{\operatorname{sg}(p_t)>\tau\},\qquad
L=\mathbb E\left[-\frac1T\sum_t M_t\log p_t\right].
$$

Masked tokens remain in the input context; their direct loss is omitted. Normalization is by total sequence length, not by retained-token count.

**Intuition:** preserve a pretrained model's existing semantic prior while reducing the pressure to imitate an arbitrary rare surface wording. This can help when low-confidence reference expressions are mostly stylistic, but the method has no way to tell an unfamiliar crucial fact from an unfamiliar paraphrase using probability alone.

#### Theorem and its exact reach

For one token, logit gradient is $p-e_{y^*}$. If the logit Jacobian $J=\partial z/\partial\theta$ has full row rank with $\sigma_{\min}(J)\ge\gamma>0$, then

$$
\|\nabla_\theta\ell\|_2\ge\gamma(1-p_{y^*}).
$$

This is a correct conditional spectral lower bound. It does not order the actual gradient norms of two tokens with different Jacobians, prove their gradients conflict, or establish semantic importance. Frozen output heads and restricted trainable parameterizations can violate the surjectivity assumption. A low-probability token can have a large lower bound while a high-probability token has a much larger actual gradient through another Jacobian.

The semantic study uses a Gemini-3-pro labeler and Qwen3-4B probability proxy, reporting means 0.768 core versus 0.485 trivial and $p=10^{-6}$. Distributional significance is not causal necessity; token frequency, answer position, tokenizer segmentation, model familiarity, and annotation selection can explain part of the correlation. No calibrated error rate for “rare but crucial” token rejection is established.

#### Evidence and limits

Training uses 2,000 high-reward InfinityInstruct examples and eight H20 GPUs. Models include Qwen3 0.6B/4B/14B, OLMo2-7B, Llama3.1-8B, and instruction variants. Main average gain over SFT on Qwen3-4B is 10.94 percentage points (52.33 versus 41.39), while Llama gain is only 0.34 (27.04 versus 26.70). Gains are heterogeneous across tasks and scales. Results are generation averages (8 samples, 32 for AIME), not repeated-training seed estimates. General-task Qwen3-4B average slightly favors SFT. Instruct Qwen2.5-3B still loses 0.73 points versus its base.

Thresholding also reduces total loss/gradient scale when retaining fewer tokens. Compare matched effective learning rates, random masks of the same size, and normalization by retained count before attributing all gains to semantic selection. A target token initially below threshold can be permanently excluded from direct learning unless other shared updates raise its probability: this is a real missing-knowledge risk under domain shift. The source does not show that semantic variety is preserved merely because some tokens have no loss.

### 7. Clipping-Free Policy Optimization for LLMs

Source: [arXiv:2601.22801v1](https://arxiv.org/abs/2601.22801). Locations: Eq. (7)–(10), Sections 4–5, Tables 1–2, Appendix A Theorems A.1/A.3/A.5, Appendix B. The method adapts earlier Simple Policy Optimization; the quadratic is not introduced from scratch here.

#### Mechanism and intuition

The clipped term is replaced by

$$
f(r,A)=rA-\frac{|A|}{2\epsilon}(r-1)^2,
\quad
\partial_r f=A-\frac{|A|}{\epsilon}(r-1),
\quad
r^*=1+\epsilon\operatorname{sign}(A).
$$

**Intuition:** when an update overshoots, supply a restoring signal instead of a flat clipped plateau. This can improve repeated-sample optimization, while making early progress less aggressive.

The penalty is convex, but the objective being maximized is **concave** in $r$: $\partial_r^2 f=-|A|/\epsilon$. The paper calls the objective convex in both the main text and Appendix A's definition/proof. That wording is mathematically wrong. The gradient is zero at the optimum and for $A=0$; “nonzero everywhere” must be read as eliminating the clipped plateau. The one-sample ratio optimum is independent of advantage magnitude, contrary to the sentence suggesting larger advantages permit proportionally larger ratio movement.

#### What the theoretical motivation does not guarantee

Pinsker gives $\Omega_{\mathrm{KL}}\subseteq\Omega_{\mathrm{TV}}$ when $\delta_{\mathrm{TV}}\ge\sqrt{\delta_{\mathrm{KL}}/2}$. Optimizing a larger feasible set can improve a stated surrogate bound. Neither fact proves that finite-minibatch CFPO actually enforces a per-state TV constraint or dominates a tuned PPO implementation.

Without advantage weighting, the expected quadratic under the old policy is a chi-squared divergence, not TV itself. With $|A|$ it is an advantage-weighted quadratic. Soft optimization under shared neural parameters does not independently set each sampled ratio to $r^*$, nor control all unseen actions/prefixes. The appendix's performance bound requires care with the absolute value in the advantage supremum. Do not inherit the displayed sign-sensitive quantity as a valid arbitrary-policy theorem without checking the original bound.

At exactly on-policy ratio $r=1$, CFPO and clipped GRPO have the same first gradient. Differences become meaningful with minibatch drift or sample reuse, which the paper usefully studies across TRL and verl.

#### Evidence and limits

Reasoning uses Qwen2.5 1.5B/3B/7B base/instruct, MATH training (7,500 prompts), group size 3, and reuse/mini-batch sweeps totaling 80 trained models across frameworks. CFPO extends stable regimes but does not universally avoid collapse: 1.5B and 7B cold-start settings break down at 16 iterations; GSM8K can favor GRPO. The paper describes incomplete/unintended-language generations as a confound.

RLHF uses Llama3-8B and OpenRLHF rewards/initialization with two rollouts. CFPO length-controlled AE2 is 11.26 versus RLOO 7.25; IFEval is 55.64 versus 47.00, both below the SFT baseline 59.59. Capability average is 0.577 versus RLOO 0.531, still below SFT 0.603. These support improved retention under the given setup, not complete elimination of alignment tax or reward hacking. Shared hyperparameters isolate a drop-in substitution, but an equally tuned conservative baseline could narrow the advantage.

### Part V — Factuality and layer diagnostics

### F-DPO — arXiv:2601.03027v3

Primary source: https://arxiv.org/abs/2601.03027v3

**Mechanism and intuition.** Binary hallucination labels `h=1` identify factual conflicts. Eq.5 swaps chosen/rejected responses whenever the chosen is less factual. Eq.8 then subtracts a fixed margin from the DPO score only when a factual response beats a hallucinated response. It first repairs the direction of supervision, then keeps that direction under stronger optimization pressure. Equal-label pairs retain standard DPO, including pairs where both responses are hallucinated.

Let $m$ be the reference-relative log-probability difference. The loss is $\ell=-\log\sigma(\beta(m-\lambda\Delta h))$, with $\Delta h\in\{0,1\}$ after flipping. Its derivative is $\partial_m\ell=-\beta\sigma(-\beta(m-\lambda\Delta h))$. Thus the margin increases pressure but does not change a given pair's update direction. As $\lambda$ becomes large at fixed $m$, the pressure saturates at $\beta$; the loss/margin magnitude is not itself a new factuality guarantee. Confidently wrong factuality labels can reverse the correct supervision direction.

**What it contributes.** A simple factuality data pipeline and seven-model empirical study, plus factorial flipping/margin ablations and external TruthfulQA multiple-choice tests. Its innovation is predominantly supervision repair and task-specific margin pressure, not a new general preference-identification theorem. GPT-4o-mini supplies dataset factuality judgments and held-out judge scores; the method is verifier-free during policy optimization, not free of external judgment in data construction/evaluation.

**Evidence limits.** Table2 improves over vanilla DPO across the seven models, but Gemma-2-2B F-DPO hallucination=.116 remains worse than its base=.098. Table3 Gemma-2-9B flipped vanilla DPO=.064 versus F-DPO=.068, so the margin is not universally beneficial. TruthfulQA Table6 reports improved MC accuracy but lower BLEU/ROUGE; these measures answer different questions. The high margin default lambda=100 should be interpreted jointly with beta and the empirical log-ratio scale, not as a reward unit. Data balancing, synthetic corruption, label repair, and margin must be separated with compute/data matched comparisons. The paper's attribution of factuality degradation to stylistic preference is plausible but not isolated by Table2 alone. Human agreement r=.8 is reported, with no independent replication here. No correctness theorem establishing arbitrary unseen factuality is supplied.

**Useful research implication.** Treat side-label reliability as an identification problem before introducing a stronger loss. A proposal that merely replaces the safety margin with a factual margin already collides with this paper.

### SPINAL — arXiv:2601.06238v1

Primary source: https://arxiv.org/abs/2601.06238v1

**Mechanism and intuition.** A post-hoc layer diagnostic combines activation-spectrum shape, adjacent-layer logit-lens Fisher–Rao distance, terminal trajectory coherence, and the share of DPO gradient norm in terminal layers. It asks where a checkpoint changed internally. This is a diagnostic, not a new alignment optimizer or an established behavioral certificate. Sections6–7 and appendices explicitly narrow much of the causal interpretation to hypotheses/future validation; many appendix panels are illustrative and Table9 is an unfilled test protocol.

**Valid geometric core.** For predictive distributions $p_\ell,p_{\ell+1}$, $L_\ell=2\arccos\sum_y\sqrt{p_\ell(y)p_{\ell+1}(y)}$ is a Fisher–Rao geodesic distance. It measures movement between distributions, not their entropy: identical uniform distributions and identical point masses both give zero distance. Truncating and renormalizing top-k vocabulary computes distance on the truncated distributions, which need not match full-vocabulary distance. Captured mass and shared support are material.

**Internal formula conflict.** The abstract defines a slope-based exponent, while Sec.3.1 and the reproducibility box define $\sigma_k=Ck^{-1/\alpha}$ and estimate $\alpha=-1/\widehat\beta$ from the log–log slope. Under this latter definition, increasing alpha makes decay *shallower*, opposite to the text's claimed steeper decay/concentration. For a fixed pure-power spectrum it increases effective dimension rather than decreasing it. This must be reconciled with code before interpreting positive delta-alpha as sharpening. Sec.3 uses centered token-mean activations; the computation box instead defaults to last-token states with no extra centering. A stable implementation must choose and disclose one object.

**Exact negative-control issue.** Table9 groups token-order shuffling and permutation of activation-matrix rows as destructive controls. They differ: for a row-permutation matrix $P$, $(PH)^\top(PH)=H^\top H$, so every singular value, spectral slope and effective-rank statistic is exactly unchanged. A common prompt permutation also leaves batch-averaged Fisher–Rao distances unchanged. This proposed row control cannot destroy the measured spectrum. Token shuffling can change model outputs and is a different intervention.

**Inference limit.** Unweighted parameter-gradient norms depend on parameterization; final layers also sit nearest the measurement head. Base versus instruct comparisons confound SFT and subsequent preference training unless checkpoints share a controlled training history. Randomizing terminal layers may destroy language capability generally, so it does not isolate alignment-specific causality. A spectrum/transport summary can be useful for monitoring despite these problems; use direct behavior and matched interventions to establish the interpretation.

**Useful research implication.** Distinguish coordinate-dependent descriptors, genuine probability geometry, and causal behavior. A metric correlated with alignment on a small checkpoint set is insufficient as a policy-quality or calibration theorem.

## 8. Integrity and reproducibility

This report is AI-assisted synthesis and mathematical analysis, including bounded workers using the same model. Those workers are not independent human peer review. Source PDFs, extracted texts, detailed notes and version mappings are retained locally. The checked finite-action values are reproducible with Python 3 standard-library code. No neural-model results or completed open theorems are claimed. The focused related-work screens are not an exhaustive novelty clearance.
