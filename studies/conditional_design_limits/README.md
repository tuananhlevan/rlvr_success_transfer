# Research lead: the statistical cost of shared-policy preference exploration

**Date:** 16 October 2026. **Status:** bounded theory sprint completed. Exact geometry, matching constant-confidence resource rates on the sparse family, and an exact KL-constrained design curve are derived. The earlier interpretation of a universal quadratic *label* requirement is corrected. A full methods project is not justified.

## 1. Decision and research question

The replacement candidate is a theoretical study of **which exploration restrictions impose an unavoidable information cost**, rather than a proposed DPO loss or a new generic sampler.

> When all prompts share one bounded policy during a collection round, and auxiliary data use independently optimized within-prompt response distributions, how much more preference feedback is necessary than with coordinated prompt/response queries? Which restrictions are responsible for that difference?

The starting point is an explicit open question in an original corpus paper. [Kim et al., *Coverage Improvement and Fast Convergence of On-policy Preference Learning*, ICML 2026](https://proceedings.mlr.press/v306/kim26bz.html), Remark 4.3, asks whether a fixed-conditional design bound can improve from quadratic to linear in feature dimension. This remains open in the checked final proceedings PDF: Eq. (5), pp. 6–7; the local arXiv v1 calls it Eq. (7). This is not a criticism of an obsolete draft.

Our exact construction answers that particular question negatively. A further information argument shows that, under a specified shared-policy sampling protocol, adaptation need not remove a quadratic discovery cost. These are derived mathematical results, not measurements of language-model failures.

**Sprint decision:** retain the mathematical results as a scoped theoretical note. Do not start an RL training campaign. Screening before labeling is an ordinary repair of the label-cost bottleneck, and changing the policy constraint or conditioning on the prompt can also remove the generation bottleneck. A full paper still needs broader novelty and a consequential use case beyond this constructed family. See Sections 6–10 for completed work and remaining gaps.

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
| [Foster et al., 2025](https://arxiv.org/abs/2503.07453); §§3.1/4 and Appendix A.1 checked | Computational limits of proper exploration, and efficient improper exploration. | Already separates generation from feedback, including screening before an oracle query. Our narrow construction does not establish a new general separation principle. |

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

## 5. Audited lower bound: generated pairs and labels are different resources

### Model, loss and fresh-pair query contract

Take $d\geq16$, $R=\log d$, $\eta=1/d$, uniform response reference, and

$$
\pi_\theta(a\mid x)\propto\exp(\theta^\top\phi(x,a)),
\qquad \|\theta\|_2\leq R.
$$

The unknown target belongs to the **finite, known-amplitude** family $\theta_J^*=Re_J$, $J\in[d]$. A same-prompt pair receives an independent label with probability

$$
P_J(a\succ b\mid x)=\sigma\!\left(
\frac{(\theta_J^*)^\top[\phi(x,a)-\phi(x,b)]}{R}\right).
$$

Before a raw prompt/pair draw, the learner can adaptively mix:

1. Shared policies with $\|\theta_t\|_2\leq R$, selected **before** a uniform prompt is drawn.
2. Any prompt marginal $\mu_t$ with the fixed within-prompt distribution $\pi^{cg}$.

The resulting conditional response distribution supplies both action marginals at the same prompt. Mixtures of shared policies are allowed. After observing prompt, responses and their known features, the learner **may decide whether to request the label**. Each raw pair can receive at most one label in this theorem. Denote by $G$ the hard budget of generated raw pairs, and by $L$ the hard budget of requested labels. Unlabeled rejected pairs count toward $G$ but not $L$. Labels are the only target-dependent external information; the feature map and response collection kernel are known. The output obeys $\|\widehat\theta\|_2\leq R$. Section 7 separately allows cached-pair recombination and repeated labels; the quadratic $G$ bound must not be extended to that protocol.

This explicit screening permission is a correction to the earlier discussion, which treated a drawn pair and a queried label as one resource. The original bound remains valid if every pair must be labeled, but that convention cannot establish a universal human-feedback requirement.

### Adaptive information bound

Use the null target $\theta^*=0$ as a change-of-measure reference. For every allowed shared parameter,

$$
\sum_i\cosh(\theta_{t,i})\leq d+\cosh R-1\leq\tfrac32d.
$$

Indeed $\cosh u-1\leq(\cosh R-1)u^2/R^2$ for $|u|\leq R$. Each response normalizer is at least $2d$. Thus the average probability of the prompt's strong axis is at most $3/(2d)$ under a shared policy and exactly $1/d$ under $\mu_t\times\pi^{cg}$. This implies $\mathbb E_0\|\phi\|_2^2\leq5/(2d)$.

For $z_J=\phi(x,a)_J-\phi(x,b)_J$,

$$
\operatorname{kl}(1/2,\sigma(z_J))
=\log\cosh(z_J/2)\leq z_J^2/8.
$$

An unqueried label contributes zero information. A queried pair contributes at most $\|\phi(x,a)-\phi(x,b)\|_2^2/8$ after summing alternatives. The adaptive KL chain rule therefore gives

$$
\boxed{\sum_{J=1}^d D_{\rm KL}(P_0\|P_J)
\leq\min\!\left\{\frac{5G}{4d},\frac L2\right\}.}
$$

The first bound uses raw-pair second moments, even when the learner screens them. The second uses $\|\phi(x,a)-\phi(x,b)\|_2\leq2$. All choices of prompts, policies and label requests can depend on previously observed information under this contract.

### Required information

The target covariance is diagonal and $V(\pi_J^*)_{JJ}\geq1/(9d)$. At prompt $J$, the normalizer is $Z=d+d^{-1}+2d-2\leq3d$, and the axis variance equals

$$
\frac{(d+d^{-1})(2d-2)+4}{Z^2}\geq\frac19.
$$

Thus target-weighted squared error below $R^2/(36d)$ implies $A_J=\{\widehat\theta_J>R/2\}$. The output norm ensures $\sum_JP_0(A_J)\leq4$. At least half the indices have $P_0(A_J)\leq8/d\leq1/2$. Uniform success probability $0.9$ implies $P_J(A_J)\geq0.9$ and, by binary data processing,

$$
\sum_JD_{\rm KL}(P_0\|P_J)
\geq\frac d2\operatorname{kl}(1/2,0.9).
$$

Writing $k=\operatorname{kl}(1/2,0.9)=0.5108256\ldots$, we obtain

$$
\boxed{G\geq\tfrac25kd^2>0.204d^2,
\qquad L\geq kd>0.510d.}
$$

If every raw pair must be labeled, $L=G$ and the quadratic label bound applies. **With screening, the valid lower bound on labels is linear.**

### Connection to policy quality

This need not remain a covariance-error statement. If $\widehat\theta_J\leq R/2$, then at prompt $J$,

$$
\pi_J^*(+J\mid J)\geq\tfrac13,
\qquad\pi_{\widehat\theta}(+J\mid J)\leq\frac1{2\sqrt d}\leq\tfrac18.
$$

Data processing to this binary event and Pinsker's inequality imply, in **either KL direction**, average prompt KL at least $25/(288d)$. Hence the same resource lower bounds apply to achieving a smaller KL error with probability $0.9$ for every target.

For reward $r_J(x,a)=(\theta_J^*)^\top\phi(x,a)/R$ and KL coefficient $\tau=1/R$, define $J_\tau(\pi)=\mathbb E_\pi r_J-\tau\mathbb E_xD_{\rm KL}(\pi\|\pi_0)$. The variational identity gives

$$
J_\tau(\pi_J^*)-J_\tau(\pi)
=\tau\,\mathbb E_xD_{\rm KL}(\pi\|\pi_J^*).
$$

Consequently regret below $25/(288Rd)$ also forces the identification event used above. **The thresholds still shrink with dimension**; this does not demonstrate a constant population-wide LLM benefit.

## 6. Matching upper bounds and a simple repair

### Exact reduction to identifying one biased coin

Choose prompt $i$ using the allowed auxiliary prompt marginal, and draw two uniform conditional responses. Retain a pair only when **exactly one** response lies on its strong axis $\pm e_i$. Its acceptance probability is

$$
\rho_d=2\frac1d\left(1-\frac1d\right)=\frac{2(d-1)}{d^2}.
$$

Let $Y$ indicate preference for the strong-axis response. If that response is positive, set $B=Y$; if negative, set $B=1-Y$. Direct averaging over response signs gives

$$
B\sim\begin{cases}
\operatorname{Bernoulli}(\sigma(1)),&i=J,\\
\operatorname{Bernoulli}(1/2),&i\ne J.
\end{cases}
$$

For $i\ne J$, the strong-axis sign is independent of the other response and cancels its nuisance effect exactly. This holds for any $\eta$, not only $1/d$. The script verifies it by enumerating ordered response pairs, rather than assuming the reduced coin model.

### Removing the previous logarithmic gap

A known-gap knockout tournament identifies the unique biased coin using $K=O(d\log(1/\delta))$ retained observations. For powers of two, round $r$ has $d/2^{r-1}$ active coins. Compare consecutive pairs using

$$
m_r=\left\lceil\frac{2}{\Delta^2}\log\frac{2^{r+1}}\delta\right\rceil
$$

observations per coin, where $\Delta=\sigma(1)-1/2$. The true coin loses its round with probability at most $\delta/2^r$ by Hoeffding's inequality. Only its own path needs a union bound. Moreover,

$$
K\leq d\left[\frac4{\Delta^2}\log\frac1\delta
+\frac{12\log2}{\Delta^2}+2\right].
$$

Thus the earlier $\log d$ factor was unnecessary. General $d$ can be padded to the next power of two with internally simulated, known fair dummy coins.

For screened acquisition, query only retained labels. Then

$$
L=K=O(d\log(1/\delta)),\qquad
\mathbb E G=K/\rho_d=O(d^2\log(1/\delta)).
$$

A hard cap $G_{\max}=\lceil(2K+8\log(2/\delta))/\rho_d\rceil$, with tournament error budget $\delta/2$, bounds generation overflow by $\delta/2$ using a binomial lower-tail bound. Hence **at fixed confidence**, the generation and label rates match their respective lower bounds: $G=\Theta(d^2)$ and $L=\Theta(d)$. We do not claim matching dependence on arbitrarily small $\delta$.

If all generated pairs are obligatorily labeled, the same algorithm has $L=G=\Theta(d^2)$ at fixed confidence. Screening changes label cost without changing its generator.

For direct coordinated pairs $(+e_i,-e_i)$, the biased mean becomes $\sigma(2)$. The same tournament needs $G=L=O(d\log(1/\delta))$, thus $\Theta(d)$ at fixed confidence. These are rates for the finite known-amplitude sparse family, **not** arbitrary parameter vectors or final fine-precision DPO learning.

### Prior-art implication

Screening is not a proposed invention. [Foster et al. (2025), §§3.1 and 4](https://arxiv.org/pdf/2503.07453), already distinguish generated candidates from oracle queries and explicitly select whether to request feedback. This source now receives a deeper method-level check than in the initial screen. Our result cannot be advertised as discovering that generation and human-feedback efficiency differ.

## 7. Which restrictions actually cause the gap?

### Cache reuse changes the generation–label tradeoff

Now retain the same raw generation restrictions but permit arbitrary recombination of responses **within their original prompt**, with independent repeated noisy labels. Reusing an answer at a different prompt is not covered by this cache protocol.

A simple counterexample to an unconditional generation lower bound is to collect all $2d$ signed responses at prompt 1. Uniform collection takes $O(d\log(d/\delta))$ raw pairs with high probability by the coupon-collector bound. Querying cached antipodal pairs gives one coin per coordinate: its biased mean is $\sigma(2)$ if $J=1$, $\sigma(2\eta)$ if $J\ne1$, and $1/2$ at non-target coordinates. A tournament with known minimum gap $\sigma(2\eta)-1/2=\Theta(\eta)$ uses $O(d\eta^{-2}\log(1/\delta))$ independent labels. For $\eta=1/d$, this gives

$$
G=O(d\log(d/\delta)),\qquad L=O(d^3\log(1/\delta)).
$$

It is expensive in labels, but it invalidates any universal claim that quadratically many *new generations* are necessary when repeated cached labels are unrestricted.

**A lower tradeoff still holds.** Let $S_J$ mean that a strong-axis response at prompt $J$ has ever been generated. Under the null, $\sum_JP_0(S_J)\leq3G/d$, since a raw pair contains at most $3/d$ strong-axis responses on average. Stop the transcript for alternative $J$ when $S_J$ first occurs. Before that time, every cached response's $J$th coordinate has magnitude at most $\eta$. Summing KL across these separately stopped experiments gives

$$
\sum_J D_{\rm KL}(P_0^{\rm stop,J}\|P_J^{\rm stop,J})
\leq\frac{\eta^2 L}{2}.
$$

To see the summed bound, at each label query project its response difference onto coordinates whose strong response has not appeared: both projected responses have norm at most $\eta$, so their squared difference is at most $4\eta^2$.

Set $B_J=S_J\cup(A_J\cap S_J^c)$, observable in the stopped experiment. Uniform estimation success gives $P_J(B_J)\geq0.9$, while $\sum_JP_0(B_J)\leq3G/d+4$. Pinsker and Cauchy–Schwarz therefore yield

$$
0.9d\leq\frac{3G}{d}+4+\frac\eta2\sqrt{dL}.
$$

For $\eta=1/d$,

$$
\boxed{\frac{3G}{d^2}+\frac12\sqrt{\frac L{d^3}}
\geq0.9-\frac4d.}
$$

Thus a subquadratic generation budget forces a cubic label budget asymptotically under this within-prompt cache protocol. The fresh-screening and weak-cache constructions match these two polynomial endpoint scales. This is not a complete Pareto frontier with sharp constants/logarithms, nor a statement about cross-prompt answer reuse or correlated repeated ratings. The always-valid linear label lower bound from the output-identification argument also remains necessary.

### Prompt-conditioned bounded policies

After observing prompt $i$, choose a sign-symmetric mixture of $\pi_{Re_i}$ and $\pi_{-Re_i}$. Each component still has norm $R$. The strong-axis mass is

$$
p_{\rm main}=\frac{\cosh R}{\cosh R+d-1}\geq\frac13.
$$

Two independent responses are opposite strong-axis responses with probability $p_{\rm main}^2/2\geq1/18$. Filtering to those pairs yields the $\sigma(2)$ coin. Generation and labels are both linear in $d$ at fixed confidence. The only protocol change is permitting the parameter choice to depend on the current prompt. This is an ordinary escape, not a new method claim.

### Temperature and the norm budget

For a flat shared vector $\theta_i=R/\sqrt d$, temperature $T=1/\sqrt d$ makes the effective logit on every strong axis equal to $R$. Its strong-axis mass approaches $1/3$ as $d$ grows. Both responses still use a shared policy, so **sharing alone does not force the obstruction**. The effective parameter norm has increased from $R$ to $R\sqrt d$.

That scale is also necessary for constant average strong-axis mass within this feature family. If its average is at least a fixed $c>0$, at least $cd/2$ prompts have strong-axis mass at least $c/2$. Since all other axes contribute at least $d-1$ to the cosh normalizer, those coordinates satisfy

$$
|\theta_i|\geq\operatorname{arcosh}\!\left(\frac{c(d-1)}{2-c}\right).
$$

Their Euclidean norm is therefore $\Omega(\sqrt d\log d)$. Mixtures cannot exceed the largest component's average mass. This necessity/sufficiency concerns **coverage mass**, not a general minimax learning rate for all temperatures.

Extremely low temperature can instead produce two identical responses and destroy pairwise information. The numerical controls include both a single cooled policy and a sign-symmetric mixture. Replacing the Euclidean ball by an $\ell_\infty$ ball also admits the dense exploratory vector at radius $R$. A theorem advertised as a consequence of policy sharing alone would be incorrect.

### Exact KL-budget design frontier

A parameter norm need not represent a real pipeline constraint. We therefore also solve a distinct, operationally interpretable design problem: keep prompts uniform, allow arbitrary response laws $q_i$, and impose average KL to the uniform response reference at most $\kappa$.

Use the original, zero-centered feature vectors in the leverage criterion and the average within-prompt covariance $V(q)$. Define $p_\kappa\in[1/d,1]$ as the largest $p$ with

$$
\operatorname{kl}(p,1/d)\leq\kappa,
$$

with $p_\kappa=1$ when $\kappa\geq\log d$. Then

$$
\boxed{\inf_{q:\,\mathbb E_xD_{\rm KL}(q_x\|\pi_0)\leq\kappa}
\max_{i,a}\phi(i,a)^\top V(q)^{-1}\phi(i,a)
=\frac{d}{\eta^2+(1-\eta^2)p_\kappa}.}
$$

**Lower bound:** if $\bar p$ is average strong-axis mass, binary data processing and Jensen give $\operatorname{kl}(\bar p,1/d)\leq\kappa$, hence $\bar p\leq p_\kappa$. Also $\operatorname{tr}V(q)\leq\eta^2+(1-\eta^2)\bar p$. Since every unit axis is available,

$$
\max_j e_j^\top V(q)^{-1}e_j
\geq\frac1d\operatorname{tr}V(q)^{-1}
\geq\frac d{\operatorname{tr}V(q)}.
$$

**Attainment:** assign total mass $p_\kappa$ equally to $\pm e_i$ at prompt $i$, and distribute the remaining mass equally over its other signed axes. Its mean is zero, average covariance is isotropic, and KL equals $\operatorname{kl}(p_\kappa,1/d)$. This attains the bound.

At $\kappa=0$ it recovers the earlier quadratic expression. For fixed positive $\kappa$ and $\eta=1/d$, $p_\kappa\sim\kappa/\log d$ and leverage is $\Theta(d\log d/\kappa)$. Linear leverage requires $\kappa=\Theta(\log d)$. This is an **exact design-criterion frontier**, not a new general learning-rate theorem, and its broader priority remains unchecked.

## 8. Completed numerical sprint

The canonical [runner](geometry_check.py) and [results](results.json) contain:

- Original matrix construction and independent LP optimization of prompt allocation.
- Complete response-pair enumeration validating the retained-coin reduction at three overlap levels.
- Known-gap tournament simulations at $d=16,32,64,128$, 1,000 trials per dimension and five accounting/protocol controls.
- Temperature controls with exact pairwise label information under the null.
- KL-frontier root solving and explicit covariance/KL checks of its attaining distributions.

### Resource accounting at $d=64$

| Acquisition protocol | Mean generated pairs $G$ | Mean requested labels $L$ |
|---|---:|---:|
| Fixed conditional sampler, label every pair | 772,009.8 | 772,009.8 |
| Same sampler, screen before labeling | 772,009.8 | 23,752 |
| Direct coordinated pairs | 8,758 | 8,758 |
| Prompt-conditioned bounded policies, then screen | 154,342.8 | 8,758 |
| Within-prompt cache, recombination and repeated noisy labels | 347.5 | 20,685,088 |

All five controls identified the target in 1,000/1,000 simulated trials; the Wilson 95% interval for each simulated success probability is approximately [0.9962, 1]. The Hoeffding schedule is deliberately conservative. **These success fractions are not LLM accuracies.**

The first two rows use exactly the same retained observations and therefore the same decisions. Their label reduction is a change in what is queried/billed, not an independent empirical discovery. The direct and prompt-conditioned controls likewise share their retained coin observations. Different coin gaps account for their different tournament sample constants. The table does not isolate acceptance probability at a common retained-label budget across all methods. The cache control trades many more independent labels for far fewer generated pairs; it does not dominate screening.

Raw rejections are counted by negative-binomial sampling, an exact distributional shortcut for this mathematical model; cache collection uses geometric coupon waiting times and labeled observations use binomial counts. We do not materialize millions of responses or oracle calls. Consequently the run time, about one second locally, is not a claim about real rollout speed. The simulation uses uncapped stopping counts; the hard-cap theorem is proved separately above.

### Temperature control at $d=64$

The scalar below is $\sum_J\mathbb E_0\operatorname{kl}(1/2,P_J(Y=1\mid x,a,b))$, computed by enumerating the action pairs; it measures information across sparse alternatives per raw labeled pair.

| Temperature | Single shared policy | Sign-symmetric shared mixture |
|---|---:|---:|
| 1 | 0.004300 | 0.004316 |
| 0.125 | 0.053694 | 0.078218 |
| 0.0625 | 0.007013 | 0.211047 |
| 0.03125 | 0.000002 | 0.216889 |

Cooling moderately increases information in this construction; cooling a single policy too aggressively destroys it. All cooled samplers have effective parameters outside the original radius-$R$ Euclidean ball. This is an explicit limit of the earlier lower bound.

### Reproduction and integrity

```bash
python studies/conditional_design_limits/geometry_check.py
```

The command overwrites the single canonical result file. Dependencies are Python, NumPy and SciPy; versions, seeds and trial counts are recorded in JSON. No GPU, LLM training, human annotations or external datasets were used. Enumeration checks and Monte Carlo support implementation consistency; the written arguments establish the mathematical claims. All extracted mathematical expressions were also compiled with LaTeX in a temporary directory; this checks syntax, not proof validity or a particular Markdown renderer. The current sprint was performed inline in the same model context, without external peer review.

## 9. What is still missing?

| Gap | Current status | Why it matters / concrete next requirement |
|---|---|---|
| External proof and priority review | Missing | The exact counterexample appears to answer the checked source question, but absence in a bounded search does not establish publication novelty. The KL-frontier and strengthened resource accounting need their own close-literature audit. |
| Full parameter classes and unknown reward strength | Missing | Matching rates above solve a finite sparse known-amplitude family. They do not characterize arbitrary linear rewards, neural policies or fine-precision estimation. A substantial theory paper needs a result beyond a single construction and elementary coin reduction. |
| Complete cache/resource frontier | Partial | A stopped-transcript tradeoff and two endpoint constructions are available. Intermediate regimes, sharp logarithms and cross-prompt recontextualization are not characterized; repeated labels require the specified independent-noise oracle. |
| Realistic collection constraints | Missing | Establish why real collection must use the norm-constrained, shared-before-prompt protocol, or develop a verified result under an operational constraint. Known-feature screening and temperature changes already repair parts of this fixture. |
| Consequential deployment failure after ordinary repairs | Missing; methods commitment gate fails | No real response bank or LLM experiment establishes a residual after selective labeling, prompt-conditioned acquisition, current active design and appropriate temperature. Without this, a new alignment method is unjustified. |
| Population-scale effect | Missing | Our utility threshold shrinks as $1/(d\log d)$ and one important prompt drives the construction. A constant-average benefit, realistic tail-risk metric or another explicitly consequential target must be demonstrated rather than inferred. |
| Practical acquisition cost | Missing | The mathematical model reveals features freely. Real generation, embedding extraction, controller overhead and rejected-candidate storage need to be counted. Frozen-bank replay cannot measure unseen candidate discovery. |

**No further optimizer or loss proposal follows from this sprint.** A bounded next step could be external theory review or an already available candidate-bank replay with the listed ordinary controls. Another long RL run is not warranted by these results.

## 10. Sprint conclusion

Completed: sharp fixed-conditional geometry; corrected adaptive generation/label lower bounds; a cache-aware stopped-transcript tradeoff; matching constant-confidence upper bounds on the same sparse family; connection to KL/regularized reward; a sharp coverage-radius scale; an exact KL-budget design frontier; and reproducible numerical controls.

The sprint also weakens the earlier project pitch: quadratic feedback cost is not fundamental when screening is permitted, shared policies alone are not the cause, and standard acquisition changes can remove the constructed obstruction. The retained output is **a scoped theoretical note and a clear account of missing evidence**, not a validated new LLM method or a publication-ready paper.
