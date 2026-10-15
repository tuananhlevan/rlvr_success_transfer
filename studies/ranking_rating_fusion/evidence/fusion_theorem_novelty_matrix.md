# Proof-level comparison: mixed BT/rating projection and sample-size crossover

Date: 2026-10-14. Scope: the precise hypothesis in the current study, not a search for another direction. This is a shared-context source audit, not blinded external review. Primary methods/theorem passages were inspected; long proofs and experiments were not exhaustively reread. No main report edits or new experiments were made here.

## 1. Exact proposed object

Fix a positive reference, deployment KL coefficient $\tau$, finite-action policy family, pair law and true reward. Write

$$
z_\theta(a,b)=\tau\left(\log\frac{\pi_\theta(a)}{\pi_0(a)}-\log\frac{\pi_\theta(b)}{\pi_0(b)}\right).
$$

Ranking bits have correct BT probability $\sigma(\Delta r)$; ratings have mean gap $\Delta r$, with exact or explicitly noisy observations. The mixed population objective is

$$
L_\lambda(\theta)=L_B(\theta)+\lambda R(\theta),\qquad
R(\theta)=\frac12\mathbb E_q(z_\theta-\Delta r)^2.
$$

The evaluation target is fixed throughout:

$$
J_\tau(\theta)=\mathbb E_{\pi_\theta}r-\tau\mathrm{KL}(\pi_\theta\|\pi_0),\qquad
\mathcal E(\theta)=\sup_{\vartheta\in\Theta}J_\tau(\vartheta)-J_\tau(\theta).
$$

The candidate theorem is **not** the original realizable fusion theorem. It concerns the pseudo-true path $\theta_\lambda$ when the reward cannot be represented by policy log ratios, and whether its statistical gains offset its additional policy-value projection loss.

## 2. Theorem comparison matrix

Entries deliberately separate targets and assumptions. A theorem about the same broad theme is a conceptual collision; an identical observation model, objective, policy restriction and conclusion would be an exact collision.

| Primary source and traceable locator | Method / mathematical object | Assumptions relevant to comparison | Proven target and conclusion | Relation to proposed claim |
|---|---|---|---|---|
| [Rating DPO](https://arxiv.org/html/2602.00603v1), Eq. (13), Assumptions 4.1–4.2, Theorem 4.4, Appendix C.4 | BT cross entropy plus Gaussian gap square loss on policy/reference differences; essentially the proposed loss family. | I.i.d. candidate generation; expressive policy class realizing Gibbs solutions; bounded induced gaps; independent Gaussian rating/ranking channels; exact constrained minimization. | High-probability regularized policy gap with a minimum of exponential ranking-range and polynomial rating-noise terms, divided by sample size and square-rooted, with coverage. | **Exact objective collision.** It does not characterize a restricted misspecified fusion path or prove all-positive-weight negative transfer. Extending the theorem is necessary; relabeling the objective is not a contribution. |
| Same source, Eq. (10), Theorem 4.3 and Appendix C.2 | Rating-dependent offset inside the BT likelihood, not simply the ML fused likelihood. | Oracle rating-error ratio tuning, policy expressivity and boundedness; permits a covering argument for dependent score/rank sources. | Robustness/acceleration bound at effective temperature $\beta\beta_1/(\beta+\beta_1)$. | A changed-temperature RDPO comparison cannot substitute for the fixed-$\tau$ fusion claim. Sparse ratings and oracle trust are already addressed by the paper. |
| [AuxDPO](https://arxiv.org/html/2510.20413v1), Proposition 1, Sections 3–4, Propositions 7–9; supplied ICLR paper | DPO as weighted BT-KL projection onto induced reward manifold; augment reward with policy-gradient-null residuals. | Finite parametric geometry; clean population BT data; correction/recovery uses local linearization and sufficiently large KL coefficient. | Establishes misspecification-induced reversal/value loss and a local auxiliary recovery result up to prescribed approximation error. | **Closest mechanism collision.** Clean surrogate projection can harm value is occupied. Exact mixed BT/square-loss global path classification is not supplied by these propositions. |
| [REBEL v2](https://arxiv.org/html/2404.16767v2), Eq. (5), Claims 1–2, Assumption 1, Theorem 1 | Moving-reference relative-reward regression; current-policy/base pairs; natural-gradient interpretation. | Per-round in-distribution regression error $\epsilon$; comparator coverage; chosen learning rate and bounded quantities. | Some iterate competes in reward value at order $T^{-1/2}+\sqrt{C\epsilon}$; approximation need not vanish. | **Constructive correction collision.** A value-directed gap update is existing. Its displayed theorem is not a fixed-reference fused-likelihood path theorem, and a high residual can make its guarantee weak. |
| [Reference-Sampled Boltzmann Projection](https://arxiv.org/html/2605.02469v1), Props. 1–2, Theorems 3–4 and 9, Section 6, Appendix C | Sampler/weight product defines SFT target; density-ratio weights match Gibbs target; forward fitting versus reverse value. | Reference rollout independence; target coverage; a forward-to-reverse KL certificate for value conversion; exact mirror targets or controlled iterative drift. | Target identity and finite learning/value decomposition including approximation; no global equality of misspecified forward/reverse projections. | **Projection-boundary collision.** Correct rewards/target do not ensure the best restricted value projection. It does not classify the mixed BT/Gaussian path, but excludes exponentiated-score SFT as an obvious new repair. |
| [MetaPO, ACL 2026](https://aclanthology.org/2026.findings-acl.601.pdf), Sections 3.2–3.6, Eqs. (10)–(14), Algorithm 1 | Temporal diagnostics feed learned sample weights; bilevel held-out objective; influence expression $-\nabla L_{\mathrm{val}}^\top H^{-1}\nabla\ell_i$. | Its influence connection assumes ordinary differentiability/invertibility and is described as approximate; no comparable exact global transfer theorem. | Weighting methodology and empirical generalization results; stronger theoretical guarantees are left open. | **Diagnostic/gate collision.** Substituting rating loss and true-value validation into this expression is not itself new theory. Its validation surrogate differs from $J_\tau$, so it is not an exact theorem collision. |
| [Ordering-based Conditions, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/61c00c07e6d27285e4b952e96cc65666-Paper-Conference.pdf), Theorems 1–2; [author arXiv version](https://arxiv.org/html/2504.02130v1) | Finite-bandit policy-gradient methods with linear representation; least-squares reward projection and order preservation. | Update-specific non-domination/order conditions; NPG condition concerns preserving the optimal action in projected rewards. | Sufficient softmax-PG conditions and a necessary/sufficient projected-order condition for the specified NPG; convergence concerns unregularized best-action reward. | **Older geometry collision.** Reward MSE is not a sufficient performance descriptor. Not a fixed-temperature BT/rating-fusion statement. |
| [Fast Rates for Contextual Linear Optimization](https://arxiv.org/html/2011.03030v3), Theorems 2.5, 2.8–2.10, 3.1 and 3.4 | Compare induced empirical decision-risk minimization with estimate-then-optimize over contextual linear decisions. | Bounded decisions/outcomes; class complexity; prediction consistency/star shape for one estimator; additional near-degeneracy/margin and tail conditions for fast rates. | Uniform decision-risk bounds, matched lower comparisons and faster plug-in rates in well-specified settings. | **Rate-comparison collision.** Prediction versus decision methods already have sharp statistical comparisons. Target is linear optimization, not Gibbs value or two-channel fusion. |
| [Decision-Focused Learning with Directional Gradients, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/907a9fb75a408f6c3a2ae1bf84c39e44-Paper-Conference.pdf), Assumptions 1.1, 3.1, 3.6; Theorems 3.4, 3.7–3.8 | Finite-difference surrogates approximate true plug-in decision cost; best-in-class regret under predictor misspecification. | Bounded decisions/outcomes; uniform Lipschitz log conditional density; Rademacher or VC complexity, with polyhedral structure for stronger rates. | Vanishing best-in-class excess regret without predictor realizability, including a polyhedral $n^{-1/2}$ rate. | **Decision-focused recovery collision.** Best-in-class misspecified decision learning is established. Its discontinuous linear argmin target and full vector feedback differ from the present soft policy and mixed observations. |
| [DIAL](https://arxiv.org/html/2609.31215v1), Section 4.2 Theorem 3, Section 5 Theorems 4–5 | Judge-score subspace calibration; weighted human/judge likelihood; adaptive tuning. | Fixed score/subspace dimensions, connected human design, probabilities away from extremes; local mismatch scaling and regularity for stated asymptotics. | Calibration-space approximation versus estimation tradeoff; in a local regime $n\Delta_M\to\delta_M$, anchored first-order risk is $\delta_M+d/2$ versus $(N-1)/2$. | **Crossover-form collision.** Fixed bias can eventually dominate variance reduction; that phenomenon and adaptive trust are existing. Different target/measurement model leaves the exact mixed BT/squared policy projection open. |
| [White 1982, Maximum Likelihood Estimation of Misspecified Models](https://faculty.utrgv.edu/diego.escobari/teaching/Econ8370/Papers/White%281982%29Econometrica-Corey.pdf), Theorems 2.2 and 3.2 | Pseudo-true QMLE and robust covariance. | I.i.d. observations, integrability/compactness/identification for consistency; differentiability, interior solution and nonsingular matrices for normality. | Consistency for KL-minimizing pseudo-truth; asymptotic normality with sandwich covariance under misspecification. | **Statistical scaffold collision.** A pseudo-true limit and $H^{-1}CH^{-1}$ are classical. This theorem alone does not supply the second-order expected target-risk bias coefficient or a sharp fusion crossover. |

### Version and read-scope notes

- RDPO and AuxDPO were also available as supplied local full texts. Concrete source claims are section/equation-located rather than inferred solely from abstracts.
- REBEL, Boltzmann Projection and MetaPO methods/assumptions were reread during this sequence; no claim of exhaustive proof verification is made.
- The NeurIPS 2024 directional-gradient paper's arXiv v2 uses a revised title. The matrix uses the published primary PDF, avoiding accidental theorem numbering from a revised manuscript.
- The ordering paper is a NeurIPS 2023 publication although its author arXiv entry was uploaded later. The publication PDF is the date/venue reference.
- White's original scan has imperfect extracted formulas; the theorem labels and accompanying explanation were inspected. Its covariance result is not being cited as a proof of the new second-order risk expansion.

## 3. Claim-by-claim novelty judgment

| Proposed item | Status after comparison | Minimum change needed |
|---|---|---|
| Clean feedback plus restricted reward projection can hurt policy value | Already established broadly by AuxDPO and older geometry/decision theory. | Do not claim this general observation as new. |
| Every positive mixed BT/square-gap weight hurts in the specified full-support example | No identical statement found in this bounded screen; a useful fusion-specific specialization. | To be central theory, extend beyond one crafted instance and standard convex path calculus. |
| Local rating-transfer derivative using $H_B^{-1}$ | Standard implicit-function/influence calculus; direct MetaPO expression overlap. | Use it as a lemma or measured predictor, not the main theorem. |
| Current-policy rating sampling yields a locally value-aligned direction | Standard policy-gradient/NPG/relative-reward geometry; REBEL overlap. | Need a new attainable guarantee or a sharp regime comparison, not the identity alone. |
| Sandwich covariance including rank/rating correlation | Classical misspecified M-estimation; DIAL already discusses cross-channel covariance/inference issues. | Characterize the particular moments and prove an actual comparison, retaining all correlation terms. |
| Expected excess regret $A_\lambda+V_\lambda/n+o(n^{-1})$ | Standard second-order statistical structure; bias/variance crossover already exists in calibration and decision theory. | Explicit geometric $A_\lambda,V_\lambda$, justified remainders and matching lower/upper comparisons are needed. |
| Validation gate retaining binary-only baseline | Ordinary model selection; learned diagnostics/validation already used in MetaPO. | Not adequate central novelty. |
| Sharp global classification and finite-sample crossover for misspecified mixed BT/squared gap policy fitting | Not supplied by the checked sources; scoped scalar versions are now proved in the companion structural/statistical notes. | This is the narrow credible theorem target; novelty remains conditional. |

## 4. Why the proposed crossover still needs proof, not just a plot

Let $\theta_\lambda$ be the mixed population pseudo-truth and define $A_\lambda=\mathcal E(\theta_\lambda)$. If an estimator has valid second-order expansions

$$
\mathbb E(\widehat\theta_\lambda-\theta_\lambda)=b_\lambda/n+o(n^{-1}),\qquad
\mathbb E[(\widehat\theta_\lambda-\theta_\lambda)(\widehat\theta_\lambda-\theta_\lambda)^\top]
=\Sigma_\lambda/n+o(n^{-1}),
$$

then Taylor expansion gives a coefficient depending on **both** estimator bias and target curvature:

$$
V_\lambda=-\nabla J_\tau(\theta_\lambda)^\top b_\lambda
-\frac12\operatorname{tr}\bigl(\nabla^2J_\tau(\theta_\lambda)\Sigma_\lambda\bigr).
$$

It need not be nonnegative at a surrogate optimum. Asymptotic normality alone does not justify this expected-risk formula: bias, remainder uniform integrability and boundary/separation events require control. Nor does it justify applying the formula at very small $n$.

If $A_\lambda-A_0>0$ and $V_\lambda-V_0<0$, the formal sign change occurs near

$$
n_c=\frac{V_0-V_\lambda}{A_\lambda-A_0}.
$$

This algebra is not a new theorem. A useful theorem must establish a nonempty, practically motivated family where the signs hold, make both coefficients explicit, and bound the remainder tightly enough to bracket an actual crossing. To compare sample budgets, specify whether both channels share the same $n$ examples and whether rating collection is more expensive; otherwise the “crossover” can reflect changed information cost.

For the handcrafted example, the population penalty is already known. The stochastic sweeps being run by the parent can test whether variance savings ever offset it. A failure to find a crossing is an important negative result, not permission to infer one from a generic bias–variance story.

## 5. Focused exact-collision search ledger

Queries used in this pass included:

- “Bradley squared misspecification rating”
- “ranking rating misspecification policy”
- “ML-RDPO theorem misspecified”
- “mixed Bradley-Terry Gaussian policy misspecification”
- “rating gaps policy class misspecification”
- “ranking rating fusion regularized policy theorem”
- “decision focused estimation misspecified models prediction risk policy regret theorem”

The primary exact objective returned was Rating DPO itself. The primary neighboring theorem families were AuxDPO, REBEL, Boltzmann Projection, ordering-based PG, and decision-focused estimation. Searches also returned unrelated recommendation fusion and unverified secondary summaries; those do not establish an exact collision.

**Negative result:** no identical theorem combining this mixed BT/square-gap objective, unrealizable log-ratio class, fixed deployment temperature, all-positive-weight deterioration and characterized finite-sample policy crossover was located in this bounded search. This is not exhaustive clearance; terminology can differ and composite-likelihood/shrinkage theory may contain applicable general results.

## 6. Explicit novelty verdict

**Already occupied:** the objective, the general misspecified projection mechanism, value-aware regression, target-matched weighted likelihood, influence diagnostics, sandwich covariance, ordinary validation protection, and generic bias–variance crossover.

**Potentially distinct but incomplete:** a nonlocal transfer classification and a sharp, explicit sample/projection comparison for **mixed BT and squared cardinal-gap fitting inside a restricted policy log-ratio class**, at the same reference, pair law, temperature and information budget.

The companion notes now prove scoped scalar transfer and local-alternative crossover results. Their ingredients remain largely standard, so these proofs do not establish publication-level novelty. The current results also favor an existing reward-reconstruction baseline. Continue only if the specific theorem boundary and a natural-task application provide a substantive distinction; otherwise treat this as a diagnostic analysis of an existing loss family, without a new-method claim.
