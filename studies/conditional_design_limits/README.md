# Research lead: the statistical cost of shared-policy preference exploration

**Date:** 16 October 2026. **Status:** scoped theoretical lead with an exact geometry result and a derived adaptive lower bound; practical LLM value and full-paper novelty remain unestablished.

## 1. Decision and research question

The replacement candidate is a theoretical study of **which exploration restrictions impose an unavoidable information cost**, rather than a proposed DPO loss or a new generic sampler.

> When all prompts share one bounded policy during a collection round, and auxiliary data use independently optimized within-prompt response distributions, how much more preference feedback is necessary than with coordinated prompt/response queries? Which restrictions are responsible for that difference?

The starting point is an explicit open question in an original corpus paper. [Kim et al., *Coverage Improvement and Fast Convergence of On-policy Preference Learning*, ICML 2026](https://proceedings.mlr.press/v306/kim26bz.html), Remark 4.3, asks whether a fixed-conditional design bound can improve from quadratic to linear in feature dimension. This remains open in the checked final proceedings PDF: Eq. (5), pp. 6–7; the local arXiv v1 calls it Eq. (7). This is not a criticism of an obsolete draft.

Our exact construction answers that particular question negatively. A further information argument shows that, under a specified shared-policy sampling protocol, adaptation need not remove a quadratic discovery cost. These are derived mathematical results, not measurements of language-model failures.

**Recommendation:** retain for a bounded theory sprint. Do not start another RL training campaign. A joint sampler is already an established remedy; a methods proposal claiming to invent that remedy would fail the novelty check. This direction must earn its contribution through sharp limits and an explicit distinction between permitted information protocols.

## 2. Intuition

Imagine several prompt groups sharing a reward model. Each group exposes one feature direction strongly and other directions weakly. A sampler optimized to cover every direction *inside each group* spends most queries on directions that other groups expose much more clearly.

Reallocating only the number of prompts cannot undo this waste: the within-prompt response distribution has already been fixed. Coordinating the response distribution with the prompt can avoid it.

The interesting question is the **necessary information cost under each protocol**, including adaptive collection. The observation that global coordination can help is old and is not our novelty claim.

## 3. Closest work and collision check

| Primary source; checked scope | What is already occupied | Remaining claim to investigate |
|---|---|---|
| [Coverage Improvement, ICML 2026](https://proceedings.mlr.press/v306/kim26bz.html); local §§2–4 and final Remark 4.3/Algorithm 2 | Coverage improvement, two-round hybrid exploration, conditional and global experimental design. | A sharp negative answer to its fixed-conditional design question; a statistical lower bound for its shared-policy query restriction. |
| [Federated Linear Contextual Bandits, 2021](https://arxiv.org/html/2110.14177); §4.3 and relevant Appendix G passages checked in the literature screen | Joint multi-client information design for shared parameters. | No verified exact result matching the centered construction or the bounded shared-softmax discovery theorem below. |
| [FedConPE, 2024](https://arxiv.org/html/2405.02881); §§3.1–3.2 | Explicitly identifies deficiencies of aggregating locally optimal designs and augments missing global directions. | Its linear-reward feedback and extra key-term queries differ from this logistic-feedback protocol. The broad mechanism is nevertheless already known. |
| [Foster et al., 2025](https://arxiv.org/abs/2503.07453); abstract and §4 contract checked | Computational limits of proper exploration, and efficient improper exploration. | The result below is a statistical limit under a shared-before-prompt policy restriction. It is not a replacement for their computational hardness theorem. |

Searches used the source title/identifier, preferential and conditional G-optimal design, shared-policy/proper exploration, and federated joint design. No checked source explicitly supplied the exact results below. **This is a bounded novelty check, not proof of priority.** No new OpenReview reviews or rebuttals were accessed.

Final proceedings PDF verified from [the publisher's linked repository](https://raw.githubusercontent.com/mlresearch/v306/main/assets/kim26bz/kim26bz.pdf), SHA-256 `dafb37d3ce192fa8447768740716469b9625c179b48d63f025f2ba104469ec46`.

### Why this differs from our closed studies

- The synthetic-error study had confounded answer labels and an occupied consistency explanation. Here labels follow a specified, correctly realized Bradley–Terry model.
- The policy-capacity study concerned insufficient feedback for a restricted deployment decision. Here every target policy is representable; the restriction concerns **collection**.
- The old generator-router proposal concerned engineering cost. This candidate asks for statistical lower bounds, not another router benchmark.

## 4. Exact negative answer to the fixed-conditional design question

Let there be $d$ prompts and $2d$ responses per prompt. For $0<\eta\leq1$, define

$$
\phi(i,\pm j)=\pm s_{ij}e_j,
\qquad s_{ii}=1,\quad s_{ij}=\eta\ \ (i\ne j).
$$

Every feature has norm at most one, and every prompt's feature set has full affine dimension. No zero-probability or rank-deficient feature direction is required.

### Conditional optimum

The centered conditional G-optimal distribution at every prompt is uniform over its $2d$ responses. Its mean is zero and its covariance is

$$
V_i=\frac1d\operatorname{diag}(s_{i1}^2,\ldots,s_{id}^2).
$$

To justify uniqueness, use the augmented moment matrix of $(1,\phi)$: its determinant equals the determinant of the centered covariance. Uniform sampling attains centered maximum leverage $d$ and is D-optimal by the equivalence criterion. Strict concavity of log determinant makes the augmented optimal matrix unique. Its mean and axis masses then force equal signed response probabilities. A different choice among conditional optima cannot evade the construction.

For any prompt allocation $\mu$, the aggregate covariance has diagonal entries

$$
V(\mu\times\pi^{cg})_{jj}
=\frac{\eta^2+(1-\eta^2)\mu_j}{d}.
$$

The unit vector $e_j$ is an available centered response at prompt $j$. Smaller responses along that axis cannot have greater leverage. Consequently,

$$
\boxed{
\inf_{\mu\in\Delta_d}\max_{i,a}
\phi(i,a)^\top V(\mu\times\pi^{cg})^{-1}\phi(i,a)
=\frac{d^2}{1+(d-1)\eta^2}.
}
$$

**Proof of optimality:** some $\mu_j\leq1/d$, giving the displayed lower bound. Uniform $\mu$ attains it on every axis. For $\eta=1/d$, the value is $\Theta(d^2)$, ruling out a universal $O(d)$ improvement for the fixed-conditional problem as written.

By contrast, drawing prompt $i$ uniformly and only responses $\pm i$ gives covariance $I/d$ and maximum leverage $d$. This coordinated design is a comparison baseline, **not a claimed new algorithm**.

## 5. A scoped adaptive discovery lower bound

This extends the geometry into a statistical question. The proof below has been checked within the session, including a separate worker check; it has not received external peer review.

### Model and permitted queries

Take $d\geq16$, $R=\log d$, $\eta=1/d$, and a uniform reference over responses. Policies have the form

$$
\pi_\theta(a\mid x)\propto
\exp(\theta^\top\phi(x,a)),\qquad \|\theta\|_2\leq R.
$$

The unknown target is one of $d$ representable policies with parameters $\theta_J^*=Re_J$. A comparison of two responses at the same prompt returns an independent label with probability

$$
P_J(a\succ b\mid x)
=\sigma\!\left(\frac{(\theta_J^*)^\top[\phi(x,a)-\phi(x,b)]}{R}\right).
$$

Before each prompt is drawn, a learner may adaptively mix:

1. A globally shared policy $\pi_{\theta_t}$, with prompts uniform and $\|\theta_t\|_2\leq R$.
2. Any prompt marginal $\mu_t$ paired with the fixed conditional design $\pi^{cg}$.

Mixtures of globally shared policies are allowed. Pair sampling is from the resulting conditional response distribution at the same prompt. The output also satisfies $\|\widehat\theta\|_2\leq R$. There is no free reward oracle or additional target-dependent side information.

**The ordering of decisions is essential:** choosing a separate parameter vector after observing each prompt is outside this protocol.

### Information bound

Use the null target $\theta^*=0$ as a change-of-measure reference, with transcript law $P_0^n$. A globally shared query parameter obeys

$$
\sum_i\cosh(\theta_{t,i})
\leq d+\cosh R-1\leq\frac32d.
$$

The first inequality follows from
$\cosh u-1\leq(\cosh R-1)u^2/R^2$ for $|u|\leq R$.
Each prompt's response normalizer is at least $2d$. Thus the probability of drawing that prompt's strongly exposed axis is at most $3/(2d)$ under the shared-policy component. Under $\mu_t\times\pi^{cg}$ it is exactly $1/d$, regardless of $\mu_t$.

For either component, $\mathbb E\|\phi\|_2^2\leq5/(2d)$. For a particular query, write $z_J=\phi(x,a)_J-\phi(x,b)_J$. The label divergence is

$$
\operatorname{kl}(1/2,\sigma(z_J))
=\log\cosh(z_J/2)\leq z_J^2/8.
$$

Summing across alternatives, using the two action marginals and the adaptive KL chain rule, gives

$$
\sum_{J=1}^d D_{\mathrm{KL}}(P_0^n\|P_J^n)
\leq\frac{5n}{4d}.
$$

This bound permits arbitrary adaptive update/fitting rules within the query protocol. It is not a claim about a particular optimizer getting stuck.

### Required information for successful estimation

The target covariance $V(\pi_J^*)$ is diagonal, and its $J$th diagonal entry is at least $1/(9d)$. At prompt $J$, set $Z=d+d^{-1}+2d-2\leq3d$. The variance on this axis equals

$$
\frac{(d+d^{-1})(2d-2)+4}{Z^2}\geq\frac19;
$$

averaging prompts supplies the factor $1/d$.

Therefore target-weighted squared error less than $R^2/(36d)$ implies $\widehat\theta_J>R/2$. Let $A_J$ denote this event. The output norm constraint implies $\sum_JP_0(A_J)\leq4$. For at least half the indices, $P_0(A_J)\leq8/d\leq1/2$.

If estimation succeeds with probability at least $0.9$ for every alternative, then $P_J(A_J)\geq0.9$. Binary data processing yields

$$
\frac d2\operatorname{kl}(1/2,0.9)
\leq\sum_J D_{\mathrm{KL}}(P_0^n\|P_J^n)
\leq\frac{5n}{4d}.
$$

Hence

$$
\boxed{n\geq\frac25\operatorname{kl}(1/2,0.9)d^2
>0.204d^2.}
$$

For the **same finite known-amplitude family**, coordinated queries comparing $+e_i$ and $-e_i$ distinguish a Bernoulli mean $\sigma(2)$ at the unknown prompt $J$ from $1/2$ elsewhere. Hoeffding's inequality and a union bound give identification using $O(d\log(d/\delta))$ labels. This is not an upper bound for estimating arbitrary vectors in the parameter ball.

### What this theorem does not imply

- It does not refute the source's convergence theorem or improve its general small-error sample bound.
- Its accuracy threshold shrinks with dimension. One important prompt out of $d$ drives the construction; constant population-wide LLM improvement is not established.
- It does not cover every proper exploration policy. Prompt-conditioned parameters, some temperature transformations and unrestricted coordinated queries leave the restricted class and can remove the obstruction.
- It does not establish a universal cost for full LLM finetuning, prompt engineering, or active learning.

## 6. Numerical checks actually completed

[geometry_check.py](geometry_check.py) constructs the finite feature matrices directly, calculates covariance/leverage, and independently optimizes the prompt marginal using a linear program. The LP is valid because covariance is diagonal and the feature union contains every unit axis. The canonical output is [results.json](results.json).

| Dimension | Best prompt allocation with fixed conditional design | Coordinated design | Ratio |
|---|---:|---:|---:|
| 2 | 3.200 | 2 | 1.600 |
| 4 | 13.474 | 4 | 3.368 |
| 8 | 57.690 | 8 | 7.211 |
| 16 | 241.830 | 16 | 15.114 |
| 32 | 993.911 | 32 | 31.060 |

These are **maximum feature-leverage values**, not accuracy percentages, training-speed gains or measured human-label savings. They check the exact geometry; the proof supplies the asymptotic conclusion. No GPU, model training, new rollouts or human annotation were used. Numerical checking is not an independent proof of the adaptive theorem.

```bash
python studies/conditional_design_limits/geometry_check.py
```

Dependencies: Python, NumPy, SciPy. The recorded execution used NumPy 2.4.0 and SciPy 1.16.3.

## 7. Work required before treating this as a full research project

1. **Proof and priority audit.** Verify the exact centered-design question and adaptive query lower bound against broader experimental-design and exploration literature. Obtain external mathematical review. A previously published implication would close the corresponding novelty claim.
2. **Matching statistical characterization.** Prove upper/lower bounds in the same protocol, distinguish discovery from final precision, and characterize regimes where the quadratic penalty disappears. A single worst-case family alone may support only a short note.
3. **Test ordinary escapes explicitly.** Include prompt-conditioned sampling, temperature changes, mixed policies, pairwise active design and source-relevant exploration methods. Treat their success as an explained change of protocol, never hide them to advertise a new method.
4. **Establish practical relevance separately.** Reuse a real response bank to ask whether conditional feature designs actually waste globally informative directions. Evaluate held-out selection or policy outcomes, including generation costs and feature approximation. Frozen candidate-bank replay cannot measure new generation throughput or unseen-response discovery.
5. **Commitment gate.** A theoretical paper needs a substantial characterization beyond the exact open-question counterexample; an NLP methods paper additionally needs a consequential residual after ordinary escapes. Neither package is complete here. If the only useful takeaway is a standard coordinated sampler, close the methods direction.

## 8. Other branches rejected in this search

- Cross-image DPO partition cancellation and shared-context distillation: direct remedies already exist in [SymMPO](https://arxiv.org/abs/2506.11712) and [IC-VCO](https://arxiv.org/abs/2605.31312).
- MixDPO heterogeneity as a measurement target: scale confounding is classical; see [Hess and Train, 2017](https://eml.berkeley.edu/~train/papers/scale.pdf). An immediate normalization fix is insufficient to create a new research program.
- Generated-test selection under false premises: [Verification Trap](https://arxiv.org/html/2610.05170) already tests substantive auditor remedies; its inserted input guarantees also require care in defining the authoritative task contract. Do not recycle this as a new auditor/router method.
- Generic proper-versus-improper exploration: already directly studied by Foster et al. above.

## 9. Self-review

This lead has a firmer theoretical starting point than the retired synthetic-error proposal: an explicit unresolved mathematical question, an exact answer for that question, and a statistical argument applying to every estimator under an explicit collection protocol. Its **broad intuition and the available coordinated fixes are already known**. Its practical impact, broader priority and sufficiency for a full paper remain open. Retaining a scoped theory lead is not a claim that a top-conference-ready research program has been found.
