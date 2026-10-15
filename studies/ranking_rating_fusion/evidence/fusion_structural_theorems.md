# T1: structural results for the ranking–rating fusion path

Date: 2026-10-14. Status: proved finite-action population results and bounded executable checks. This is shared-context mathematical development, not independent external review, novelty clearance, or a neural training theorem. This is the structural theory supporting the canonical proposal and research report.

## 1. What T1 now establishes

1. An exact global fusion-path description for scalar exponential policies with any positive reference, fixed deployment temperature, and connected fixed comparison designs. The designs for ranking and rating may differ.
2. Under a transparent pair-slope/feature-sum condition, an exact classification of all-weight harm, all-weight benefit, monotone improvement, and overshoot with a unique return below the binary baseline.
3. Quantitative nonlocal value-loss bounds in the harmful regime and an exact finite-weight map, rather than only an infinitesimal sign.
4. A boundary of the earlier local remedy: frozen current-policy rating sampling can initially help, overshoot, and eventually harm on the same original example.
5. A multidimensional exact residual-moment path formula, a centered pair-distribution mismatch certificate, an exact residual-centered value Hessian, and a finite-weight remainder bound with explicit finite-family constants.

These results genuinely extend the single numerical example. Their proof ingredients are convexity, covariance ordering, implicit differentiation, and standard perturbation inequalities. **They are not yet a sufficiently novel theoretical paper.** Their strongest practical implication is the distinction between initial sampling alignment and nonlocal fusion safety. A statistical/cost theorem and a proof-level comparison against existing projection/regression work remain necessary.

## 2. Model and conventions

There are finitely many actions $i=1,\ldots,n$, features $f_i\in\mathbb R$, nonconstant across actions, true rewards $r_i\in\mathbb R$, reference probabilities $\rho_i>0$, and a fixed temperature $\tau>0$:

$$
\pi_\theta(i)=\frac{\rho_i e^{\theta f_i}}{Z(\theta)},\qquad
J(\theta)=\sum_i\pi_\theta(i)r_i-\tau\mathrm{KL}(\pi_\theta\Vert\rho).
$$

For each unordered edge $e=(i,j)$ write $d_e=f_i-f_j$, $g_e=r_i-r_j$, and let $b_e,c_e\ge0$ be fixed ranking/rating weights. Each channel's positive-weight graph is connected. Zero-feature-difference edges contribute no parameter derivative. Connectedness and nonconstant features ensure at least one informative edge in each channel.

The population objectives are

$$
B(\theta)=\sum_e b_e\operatorname{CE}\big(\sigma(g_e),\sigma(\tau\theta d_e)\big),\qquad
R(\theta)=\frac12\sum_e c_e(\tau\theta d_e-g_e)^2,
$$

$$
\theta_\lambda=\arg\min_\theta[B(\theta)+\lambda R(\theta)],\qquad\lambda\ge0.
$$

Weights may be probability masses or positive finite measures after removing zero-contribution self-pairs. Their normalization fixes the units of $\lambda$. Do not compare a numerical crossing weight across different normalization conventions without conversion. Swapping an edge reverses both $d_e$ and $g_e$ and leaves both population losses unchanged. The gaps are signed, not absolute values following a noisy winner label.

All statements use exact population BT probabilities and exact gaps. They do not assert a finite-sample guarantee, nor invoke a positive-variance Gaussian data model for deterministic gaps. Sampling weights remain fixed along each fusion path. Recomputing them adaptively defines a different procedure.

## 3. Theorem 1: the scalar fusion path is exactly an interval

Define

$$
A=R''=\tau^2\sum_e c_ed_e^2>0,\qquad
\theta_R=\frac{\sum_e c_ed_eg_e}{\tau\sum_e c_ed_e^2},\qquad
\theta_B=\arg\min B.
$$

**Theorem.** $B$ is strictly convex and coercive, $R$ is strictly convex, and all minimizers are unique. If $\theta_B=\theta_R$, the path is constant. Otherwise, for every finite $\lambda>0$, $\theta_\lambda$ lies strictly between $\theta_B$ and $\theta_R$. It moves strictly monotonically toward $\theta_R$, converges to that endpoint as $\lambda\to\infty$, and visits every interior point exactly once. At an interior point $t$, its weight is

$$
\lambda(t)=-\frac{B'(t)}{A(t-\theta_R)}>0.
$$

Its derivative is

$$
\theta_\lambda'=-\frac{A(\theta_\lambda-\theta_R)}{B''(\theta_\lambda)+\lambda A}.
$$

**Proof.**

$$
B''(\theta)=\tau^2\sum_e b_ed_e^2\sigma(\tau\theta d_e)[1-\sigma(\tau\theta d_e)]>0.
$$

Each true probability is strictly between zero and one. Any informative edge's CE grows linearly as its logit tends to either infinity, giving coercivity. The rating Hessian is $A>0$. Suppose $\theta_R>\theta_B$. At $\theta_B$, the fused derivative equals $\lambda A(\theta_B-\theta_R)<0$; at $\theta_R$ it equals $B'(\theta_R)>0$. Strict convexity puts its root between them. Implicit differentiation proves a positive derivative. The left-moving case is identical with signs reversed. Boundedness and the root equation imply $\theta_\lambda\to\theta_R$; otherwise $\lambda A(\theta_\lambda-\theta_R)$ could not stay bounded. The displayed inverse map gives the unique weight for every interior point. If the endpoints coincide, both derivatives vanish there and uniqueness makes the path constant. $\square$

**Additional conclusions.** On a nonconstant path, rating error strictly decreases while binary CE strictly increases. Neither fact determines the sign of deployment transfer.

If $s=\operatorname{sign}(\theta_R-\theta_B)$, $D=|\theta_R-\theta_B|$, and $h_\lambda=s(\theta_\lambda-\theta_B)$, define $\underline b,\overline b$ as the minimum/maximum of $B''$ on the endpoint interval. They are finite with $\underline b>0$. The fundamental theorem of calculus gives

$$
\frac{\lambda AD}{\overline b+\lambda A}\le h_\lambda\le
\frac{\lambda AD}{\underline b+\lambda A},
$$

because $B'(\theta_B+sh)=s\bar b(h)h$ for some average curvature in that interval. Also

$$
R(\theta_\lambda)-R(\theta_R)=\frac A2(D-h_\lambda)^2.
$$

These bounds quantify how quickly the fusion path moves and rating projection error shrinks at a finite weight.

## 4. Theorem 2: global value geometry from pair slopes

Consider **all** action pairs with $f_i\ne f_j$, not just sampled comparison edges. Define

$$
u_{ij}=f_i+f_j,\qquad a_{ij}=\frac{r_i-r_j}{f_i-f_j},\qquad
w_{ij}(\theta)=\rho_i\rho_j(f_i-f_j)^2e^{\theta u_{ij}}.
$$

Let $\bar w$ normalize these weights. Then

$$
V(\theta)=\operatorname{Var}_{\pi_\theta}(f)>0,\quad
k(\theta)=\frac{\operatorname{Cov}_{\pi_\theta}(f,r)}{V(\theta)}
=\sum_{ij}\bar w_{ij}(\theta)a_{ij},
$$

$$
J'(\theta)=V(\theta)[k(\theta)-\tau\theta],\qquad
k'(\theta)=\operatorname{Cov}_{\bar w_\theta}(a,u).
$$

**Slope–sum assumption:** whenever $u_e<u_{e'}$, $a_e\ge a_{e'}$. Equal feature sums impose no ordering restriction.

**Theorem.** Under this assumption, $k'\le0$ for every $\theta$. Consequently $F(\theta)=k(\theta)-\tau\theta$ is strictly decreasing, has one root $\theta_*$, and $J$ is strictly increasing before that root and strictly decreasing after it. This root is the unique global value maximum, for every positive reference and temperature.

**Proof.** Pairwise expansions give

$$
\operatorname{Cov}_{p}(f,r)=\sum_{i<j}p_ip_j(f_i-f_j)(r_i-r_j),\quad
\operatorname{Var}_{p}(f)=\sum_{i<j}p_ip_j(f_i-f_j)^2.
$$

The common partition factor cancels in their ratio. Differentiating normalized exponential weights yields the covariance formula. Equivalently,

$$
k'(\theta)=\frac12\sum_{e,e'}\bar w_e\bar w_{e'}(a_e-a_{e'})(u_e-u_{e'})\le0.
$$

Thus $F'\le-\tau<0$. The weighted average $k$ is bounded between the finite minimum and maximum slopes, so $F$ has positive/negative signs at the two infinite ends and exactly one root. Multiplication by $V>0$ preserves its signs, proving the global maximum. $\square$

**Important limits.** This is a sufficient structural assumption, not a necessary condition for unimodality. Global concavity of $J$ is not asserted. The conclusion also follows from the weaker condition $\sup_\theta k'(\theta)<\tau$. A bounded-range covariance bound, $k'\le(\max a-\min a)(\max u-\min u)/4$, can certify that weaker condition when it is below $\tau$, but may be extremely loose.

**A broad explicit reward class.** If $r_i=\alpha f_i-\eta f_i^2+c$ with $\eta\ge0$, then $a_{ij}=\alpha-\eta u_{ij}$ and

$$
k'(\theta)=-\eta\operatorname{Var}_{\bar w_\theta}(u)\le0.
$$

The old three-action reward is exactly in this class: $\alpha=268/7$, $\eta=240/7$. The theorem therefore includes arbitrary action counts/features/references and nonlinear concave utility in one representable feature; it is not confined to those three decimal values.

## 5. Theorem 3: necessary and sufficient transfer classification

Without any unimodality assumption, Theorem 1 already gives the exact necessary/sufficient criterion

$$
J(\theta_\lambda)-J(\theta_B)=\int_{\theta_B}^{\theta_\lambda}V(t)[k(t)-\tau t],dt.
$$

All finite weights are noninferior precisely when this integral is nonnegative at every interior point of the endpoint interval. This criterion is exact but largely a restatement of utility; it is not by itself a substantive characterization.

Under Theorem 2, the classification becomes an endpoint/location test. Let the endpoints be distinct, $s=\operatorname{sign}(\theta_R-\theta_B)$, $D=|\theta_R-\theta_B|$, and $a=s(\theta_*-\theta_B)$.

| Structural condition | Exact conclusion |
|---|---|
| $a\le0$ | Every finite $\lambda>0$ harms value; value is strictly decreasing along the path. This condition is necessary and sufficient for all-positive-weight strict harm. |
| $a\ge D>0$ | Value strictly increases at every finite weight. The rating endpoint is before, or exactly at, the value optimum. |
| $0<a<D$ and $J(\theta_R)\ge J(\theta_B)$ | The path increases then decreases, but every finite positive weight still improves over binary. Endpoint equality gives a return to baseline only in the infinite-weight limit. |
| $0<a<D$ and $J(\theta_R)<J(\theta_B)$ | There is one finite maximizing weight $\lambda_*$ and one later baseline-return weight $\lambda_c$. Value improves for $0<\lambda<\lambda_c$, equals binary at $\lambda_c$, and harms for $\lambda>\lambda_c$. |

**Equivalent endpoint criterion:** all finite weights are noninferior if and only if $J(\theta_R)\ge J(\theta_B)$. If the endpoints differ, this condition implies strict improvement at all finite positive weights. If the endpoints coincide, all weights are neutral.

**Proof.** The path visits the open endpoint interval in order. A strictly unimodal function strictly decreases when moving away from its maximizer. When moving toward it, it initially increases; if the endpoint crosses it, the function thereafter strictly decreases. A crossing below the starting height occurs exactly once if and only if the terminal height is lower. If the terminal height is at least the starting height, every interior height is strictly above the starting height. Conversely, a lower terminal height implies a sufficiently large finite weight is harmful by continuity. These arguments prove all entries, including necessity of $a\le0$ for uniform strict harm: when $a>0$, small weights strictly improve. $\square$

In the overshoot case,

$$
\lambda_*=-\frac{B'(\theta_*)}{A(\theta_*-\theta_R)}.
$$

The baseline crossing is the unique point $t_c$ beyond $\theta_*$ toward $\theta_R$ satisfying $J(t_c)=J(\theta_B)$, with $\lambda_c=-B'(t_c)/[A(t_c-\theta_R)]$. These weights require reward/value information; the theorem does not make them available from ranking observations alone.

### Quantitative all-weight harm

On the compact hull of $\theta_B,\theta_R,\theta_*$ let $v_{\min},v_{\max}$ bound $V$, and let

$$
0<\gamma\le\tau-k'(t)\le\Gamma.
$$

Then

$$
\frac{\gamma v_{\min}}2(t-\theta_*)^2
\le J(\theta_*)-J(t)\le
\frac{\Gamma v_{\max}}2(t-\theta_*)^2.
$$

This follows by integrating $J'=VF$ and bounding $|F(t)|$ between $\gamma|t-\theta_*|$ and $\Gamma|t-\theta_*|$. In the all-weight harmful regime, with $b=|\theta_B-\theta_*|$,

$$
J(\theta_B)-J(\theta_\lambda)
\ge\gamma v_{\min}(b h_\lambda+h_\lambda^2/2).
$$

Insert the lower bound on $h_\lambda$ from Theorem 1 to obtain an explicit finite-$\lambda$ loss bound. It is linear in small $\lambda$ when $b>0$, quadratic when the binary endpoint is already value-optimal, and approaches a nonzero endpoint loss. These conclusions do not require globally strong concavity.

The distance/regret bounds also give sufficient, though not necessary, safe-overshoot criteria. For example, $\Gamma v_{\max}|\theta_R-\theta_*|^2\le\gamma v_{\min}|\theta_B-\theta_*|^2$ guarantees $J(\theta_R)\ge J(\theta_B)$.

## 6. A nonlocal boundary: frozen on-policy ratings overshoot

Freeze the ordered rating-pair law at independent actions from $\pi_B=\pi_{\theta_B}$. Self-pairs contribute zero. Pair covariance identities yield

$$
\theta_R^{\mathrm{on}}=\frac{\operatorname{Cov}_{\pi_B}(f,r)}{\tau\operatorname{Var}_{\pi_B}(f)}=
\frac{k(\theta_B)}\tau.
$$

At $\lambda=0$, its rating gradient is $-2\tau J'(\theta_B)$, so

$$
\frac{dJ(\theta_\lambda)}{d\lambda}\bigg|_0
=\frac{2\tau[J'(\theta_B)]^2}{B''(\theta_B)}\ge0.
$$

Under slope–sum ordering, if $\theta_B>\theta_*$ then $k(\theta_B)\le k(\theta_*)=\tau\theta_*$, hence $\theta_R^{\mathrm{on}}\le\theta_*$. The reverse inequality holds when $\theta_B<\theta_*$. If $k$ is strictly decreasing between them, the rating endpoint lies strictly past the value optimum. This explains why local correction can become nonlocal overshoot. It does not imply that every overshoot harms; Theorem 3 decides that through terminal value.

For the original instance, using the binary distinct-unordered edge average and frozen **unconditioned ordered** on-policy rating pairs:

| Quantity | Checked value |
|---|---:|
| Binary endpoint | 2.2462544799 |
| Value optimum | 1.6414085771 |
| Frozen on-policy rating endpoint | -0.6560934657 |
| Binary value | 5.3719371624 |
| Frozen on-policy rating-only value | 3.8629277425 |
| Finite value-maximizing weight | 0.1352952129 |
| Return-to-binary weight | 0.3780697863 |
| Return-to-binary parameter | 1.1162944588 |

Thus $0<\lambda<\lambda_c$ helps, and every $\lambda>\lambda_c$ harms. The endpoint criterion makes the large-weight failure exact; the decimal weights are numerical roots, not certified exact constants. A refreshed, value-directed infinitesimal procedure is a different algorithm and is not ruled out.

## 7. Multidimensional extension: exact residual moments along the entire path

Now $f_i\in\mathbb R^d$ affinely span the identifiable coordinates, and each connected positive-weight edge graph supplies feature differences spanning $\mathbb R^d$. Connectedness plus affine span suffices. The policy is $\pi_\theta(i)\propto\rho_i e^{\theta^Tf_i}$, and the same fixed-temperature objectives use $z_e=\tau\theta^T\Delta f_e$.

$B$ is strictly convex and coercive; $A=R''=\tau^2E_{q_R}[\Delta f\Delta f^T]\succ0$. The path is smooth, exists uniquely for all $\lambda\ge0$, and converges to the rating projection. It need not be a line segment in parameter space.

At any point of the path define

$$
e_\theta(i)=r_i-\tau\theta^Tf_i,\qquad
v_\theta=\operatorname{Cov}_{\pi_\theta}(f,e_\theta)=\nabla J(\theta),
$$

$$
m_\theta=E_{q_R}[\Delta f\,\Delta e_\theta],\qquad
H_\lambda=B''(\theta_\lambda)+\lambda A.
$$

**Theorem 4 (exact path identity).**

$$
\theta_\lambda'=\tau H_\lambda^{-1}m_{\theta_\lambda},\qquad
\frac{dJ(\theta_\lambda)}{d\lambda}
=\tau v_{\theta_\lambda}^TH_\lambda^{-1}m_{\theta_\lambda}.
$$

**Proof.** $\nabla R=-\tau m$. Differentiate $\nabla B(\theta_\lambda)+\lambda\nabla R(\theta_\lambda)=0$ and use invertibility of $H_\lambda$. Chain rule proves the second identity. $\square$

This formula is exact at finite weights, not merely at zero. Its integral is the exact value change. The mathematics is still implicit differentiation; the content is the explicit residual moment under different pair and deployment measures.

### Theorem 5: a finite-path moment certificate

Let $p_\theta=\pi_\theta\otimes\pi_\theta$ be the independent ordered pair law, and $D_\theta=m_\theta-2v_\theta$. For a positive-definite $H$, use $\|u\|_{H^{-1}}^2=u^TH^{-1}u$. Then

$$
\frac{dJ(\theta_\lambda)}{d\lambda}
\ge\tau\|v_\theta\|_{H_\lambda^{-1}}
\left(2\|v_\theta\|_{H_\lambda^{-1}}-\|D_\theta\|_{H_\lambda^{-1}}\right).
$$

If $\|D_\theta\|_{H_\lambda^{-1}}\le(2-\epsilon)\|v_\theta\|_{H_\lambda^{-1}}$ throughout $0\le\lambda\le\Lambda$ for $\epsilon>0$, integration gives

$$
J(\theta_\Lambda)-J(\theta_B)
\ge\tau\epsilon\int_0^\Lambda\|v_{\theta_\lambda}\|_{H_\lambda^{-1}}^2\,d\lambda\ge0.
$$

For a normalized ordered rating design $q_R$ (self-pair masses may be included), let $h_\theta(i,j)=\Delta f\Delta e_\theta$, and

$$
W_\theta=E_{p_\theta}\|h_\theta-E_{p_\theta}h_\theta\|_{H_\lambda^{-1}}^2.
$$

Then the centered mismatch bound is

$$
\|D_\theta\|_{H_\lambda^{-1}}
\le\sqrt{\chi^2(q_R\Vert p_\theta)W_\theta}.
$$

**Proof.** Independent pairs satisfy $E_{p_\theta}h_\theta=2\operatorname{Cov}_{\pi_\theta}(f,e_\theta)=2v_\theta$. Substitute $m=2v+D$ in Theorem 4 and apply metric Cauchy–Schwarz. Also

$$
D=E_{p_\theta}\left[(q_R/p_\theta-1)(h_\theta-E_{p_\theta}h_\theta)\right],
$$

since the density difference integrates to zero. Hilbert-space Cauchy–Schwarz gives the centered bound. All target pair probabilities are positive in this finite model. $\square$

**Limits.** This is a deterministic certificate requiring residual and distribution information, not an estimator or a noise guarantee. Its divergence is $\chi^2(q_R\Vert p_\theta)$; importance-sampling variance for estimating target moments from source samples generally involves the reverse direction $\chi^2(p_\theta\Vert q_R)$ and can be infinite if source edges are missing. Connected graphs identify exact rewards through paths, but do not make direct importance ratios exist on missing edges.

For a fixed design, a uniform positive margin $\epsilon$ out to infinite weight generally cannot hold unless the rating endpoint is value-stationary: at that endpoint $m\to0$ while $D\to-2v$. If $v_R\ne0$, the metric ratio tends to two. Finite-path certification is therefore the natural scope; pointwise positivity should not be converted into arbitrary-depth safety.

## 8. Exact value curvature and a finite-weight remainder

### Theorem 6: residual-centered value Hessian

Let $\tilde f=f-E_\pi f$, $\tilde e=e_\theta-E_\pi e_\theta$, and $V_\theta=E_\pi[\tilde f\tilde f^T]$. Then

$$
\nabla^2J(\theta)=E_{\pi_\theta}[\tilde f\tilde f^T\tilde e]-\tau V_\theta.
$$

**Proof.** Differentiate $\nabla J=\operatorname{Cov}_\pi(f,r)-\tau V_\theta\theta$. Derivatives of exponential-family covariance are third central moments. The contraction of the derivative of $V$ with $\theta$ combines with the reward third moment into the centered $e_\theta$ term; the direct derivative of $V\theta$ supplies $-\tau V$. $\square$

In particular, if $\max_i\tilde e_i\le\tau-\delta$ on a convex region, then $\nabla^2J\preceq-\delta V$ there. With a uniform covariance lower bound, this certifies strong concavity on that region. It is a sufficient condition, often too strong for large residuals. At a realizable optimum $e$ is constant and the Hessian is exactly $-\tau V$.

### Theorem 7: finite-weight value-sign certificate

Let $\theta_B$ be the binary optimum, $H_0=B''(\theta_B)$, $g=\nabla R(\theta_B)$, and $v=\nabla J(\theta_B)$. Write $h=\theta_\lambda-\theta_B$ and define the ridge-linearized displacement

$$
h_{\mathrm{lin}}=-\lambda(H_0+\lambda A)^{-1}g.
$$

Suppose on a convex region containing $\theta_B$ and the path segment that $B''\succeq mI$, $A\succeq aI$ with $m,a>0$, $B''$ is operator-Lipschitz with constant $M$, and $\|\nabla^2J\|\le L_J$. Then

$$
\|h\|\le\frac{\lambda\|g\|}{m+\lambda a},\qquad
\|h-h_{\mathrm{lin}}\|\le\frac{M\lambda^2\|g\|^2}{2(m+\lambda a)^3},
$$

$$
\left|J(\theta_\lambda)-J(\theta_B)-v^Th_{\mathrm{lin}}\right|
\le E_\lambda:=\frac{\|v\|M\lambda^2\|g\|^2}{2(m+\lambda a)^3}
+\frac{L_J\lambda^2\|g\|^2}{2(m+\lambda a)^2}.
$$

Thus $v^Th_{\mathrm{lin}}>E_\lambda$ certifies positive transfer at that finite weight; $v^Th_{\mathrm{lin}}<-E_\lambda$ certifies negative transfer. Otherwise the bound is inconclusive, not evidence of safety.

**Proof.** Define $\bar H=\int_0^1B''(\theta_B+th)dt$. The exact optimality equation is

$$
(\bar H+\lambda A)h=-\lambda g.
$$

Its inverse bound gives the displacement inequality. Subtract the linearized equation to obtain

$$
h-h_{\mathrm{lin}}=-(H_0+\lambda A)^{-1}(\bar H-H_0)h.
$$

Lipschitz continuity gives $\|\bar H-H_0\|\le M\|h\|/2$. Apply the two inverse/displacement bounds. Finally Taylor's integral remainder for $J$ is bounded by $L_J\|h\|^2/2$, and the linear-term error by $\|v\|\|h-h_{\mathrm{lin}}\|$. $\square$

This is a quantitative finite-weight result rather than an unspecified $O(\lambda^2)$ term. Its ingredients are standard perturbation analysis; no novelty claim is attached to the lemma itself.

### Explicit constants exist in the finite family

The region assumption is not vacuous. Monotonicity of $\nabla B$, the exact optimality equation, and $A\succeq aI$ give the global path bound $\|\theta_\lambda-\theta_B\|\le\|g\|/a$ for all weights. Use the ball of this radius around $\theta_B$.

Let $D_f=\max_{ij}\|f_i-f_j\|$, $C_R=\max_i r_i-\min_i r_i$, and $T=\|\theta_B\|+\|g\|/a$. On that ball valid constants are

$$
M\le\frac{\tau^3}{4}\sum_e b_e\|\Delta f_e\|^3,\qquad
L_J\le\frac{D_f^2}{4}(\tau+C_R+\tau TD_f).
$$

For $Z_{\max}=\max_e\tau(|\theta_B^T\Delta f_e|+\|g\|\|\Delta f_e\|/a)$ and $G_B=\sum_e b_e\Delta f_e\Delta f_e^T\succ0$, one may take

$$
m=\tau^2\sigma(Z_{\max})[1-\sigma(Z_{\max})]\lambda_{\min}(G_B)>0.
$$

The first bound uses $|\sigma''|\le1/4$. The second uses the exact residual Hessian and the bounded-range variance inequality $\|V\|\le D_f^2/4$, with $\operatorname{osc}(e)\le C_R+\tau TD_f$. These constants may become extremely conservative as rewards, path radius, or saturation grow. Numerical usefulness is not guaranteed by existence.

## 9. Why the scalar endpoint rule does not automatically extend

In several dimensions the fusion path is generally curved. Even strict convexity of both losses and strong concavity of value do not make endpoint noninferiority sufficient.

For illustration only, take $B(\theta)=\|\theta\|^2/2$, $R(\theta)=(\theta-(1,1))^T\operatorname{diag}(1,100)(\theta-(1,1))/2$, and $J(\theta)=-\|\theta-(1,0)\|^2$. Both endpoints have value $-1$, but the fusion path is

$$
\theta_\lambda=\left(\frac\lambda{1+\lambda},\frac{100\lambda}{1+100\lambda}\right).
$$

At $\lambda=0.1$, value is below $-1$; for sufficiently small weights it improves, and at weight 100 it improves again. This is an explicit convex-loss/concave-value counterexample to an extension based on convexity alone. **It is not claimed to be an instance of the coupled logistic/Gibbs preference model.** Constructing analogous multiple-crossing examples within that coupled model, or identifying additional structural restrictions that rule them out, remains open here.

## 10. Executable checks and their scope

Reproducible standard-library script: [fusion_structural_checks.py](../verification/fusion_structural_checks.py). Output: [fusion_structural_checks_results.json](../verification/fusion_structural_checks_results.json).

The script checks five prespecified scalar cases, seven finite weights per case, the inverse path map, and peak/crossing weights where appropriate:

- Original uniformly weighted all-negative instance.
- A rating design with endpoint 1.8, giving monotone positive transfer.
- A design with endpoint 1.3, giving safe overshoot.
- Frozen on-policy pairs, giving unsafe overshoot and the crossing reported above.
- A nonuniform reference $(0.2,0.3,0.5)$ and temperature 1.7, still giving all-weight harm by the structural criterion.

The positive scalar designs are intentionally constructed through mixtures of fixed edge laws to demonstrate theorem branches, not sampled empirical instances or evidence of prevalence. The structural proofs, not seven-weight grids, supply the all-weight conclusions. Previously saved realizable controls remain applicable and were not redundantly expanded.

The two-dimensional finite example uses square features, uniform reference, and rewards $(0,0.3,0.4,0.9)$. It checks five fusion weights, exact residual-moment identities, the centered $\chi^2$ mismatch bound, finite-difference verification of the residual value Hessian, and the finite-weight remainder with the explicit ball constants. Its certificate proves positive transfer numerically at weights $10^{-4},10^{-3},10^{-2},0.1$; at weight 1 the bound is inconclusive despite the directly computed positive value. These are numerical inequality checks with tolerances, not a symbolic certificate of those example constants.

No claim about unseen neural models, finite-sample training, random-instance prevalence, or adaptive sampling is made. The generic quadratic multidimensional illustration is kept separate in the JSON.

## 11. Substantiveness assessment and remaining proof obligations

**Beyond the old toy:** a global arbitrary-reference/temperature/design scalar theorem, exact safe/unsafe overshoot classification, finite-weight quantitative harm, a broad concave-feature-utility class, and a concrete nonlocal failure of frozen on-policy repair. The multidimensional results expose residual moments, curved paths, and an explicit usable remainder rather than hiding finite-step error.

**Still elementary or standard:** interval tracing for two scalar convex losses, covariance ordering, the endpoint classification of a unimodal function, metric Cauchy–Schwarz, and Hessian-Lipschitz perturbation bounds. The concave quadratic reward class is meaningful but limited. The multidimensional certificates require true residual/value information; no statistical learner is derived. A paper consisting only of these results would face a strong “routine specialization of surrogate mismatch” objection.

**What would make the next package substantive:**

1. An explicit observation/cost contract in which finite/noisy cardinal supervision and policy capacity interact, with a characterized statistical-versus-projection boundary beyond a named approximation constant.
2. A sharper multidimensional structural criterion that can be estimated from affordable observed feedback and controls a genuinely nonlocal optimizer path, with matching counterexamples or lower bounds under that contract.
3. A useful bound connecting feedback design to value transfer and sampling variance, rather than assuming an oracle residual certificate; compare with standard importance correction, reward reconstruction/direct optimization, REBEL, AuxDPO, and validation selection.
4. A controlled natural-task test showing these distinctions predict meaningful behavior, including beneficial-rating regimes and converged fixed-temperature controls.

Exact gaps on a connected finite graph still identify reward up to a constant. T1 does not establish that new comparisons are necessary or that a novel optimizer is needed. The source-grounded empirical mechanism and theorem novelty remain separate open questions.

**Completion boundary:** Theorems 1–7 and their stated corollaries are proved in this note under their explicit assumptions. Numerical examples/checks have been run and saved. A full multidimensional transfer classification, feedback-efficient estimator, sharp statistical tradeoff, coupled-model multiple-crossing construction, and publication-level novelty remain unfinished.
