# T2: statistical benefit and projection harm in scalar ranking–gap fusion

Prepared 14 October 2026. This is a bounded, shared-context derivation and numerical check for [the capacity-fusion proposal](../proposal.md), not an independent external review. No language-model training was performed. The estimating-equation expansion is standard statistical machinery; the explicit fifth-order projection calculation and crossover below are derived here. Novelty against the full literature is **not established**.

## 1. Observation contract and target

One finite prompt, a fixed full-support reference, bounded scalar features and rewards, and a compact interval $\Theta=[-M,M]$. For a randomly drawn fixed-orientation pair $e=(i,j)$ define

$$
a_e=\tau(f_i-f_j),\qquad g_e=r_i-r_j,\qquad p_e=\sigma(g_e).
$$

Draw iid pairs from a specified distribution $q$. Conditional on a pair, $Z\sim\operatorname{Bernoulli}(p_e)$ and $G=g_e+\xi$, with $\xi\sim N(0,v_e)$. Unless stated otherwise the channels are conditionally independent. Rewards, Gaussian means, BT margins and the deployment temperature have the same fixed units. Let

$$
\ell_\lambda(\theta;e,Z,G)=\operatorname{CE}(Z,\sigma(a_e\theta))
+\frac\lambda2(a_e\theta-G)^2,
\qquad \widehat\theta_\lambda=\arg\min_{\Theta}P_n\ell_\lambda.
$$

The population minimizer $\theta_\lambda$ is assumed unique and interior. A compact interval prevents the finite-sample binary estimate from diverging on a separable sample; ignoring that event is not a valid unqualified finite-$n$ argument. Deployment is

$$
\pi_\theta(i)=\frac{\pi_0(i)e^{\theta f_i}}{\sum_j\pi_0(j)e^{\theta f_j}},\qquad
J_\tau(\theta)=\mathbb E_{\pi_\theta}r-\tau\operatorname{KL}(\pi_\theta\|\pi_0),
\qquad R(\theta)=\max_{u\in\Theta}J_\tau(u)-J_\tau(\theta).
$$

This analyzes an existing estimator, not the optimal attainable value from those observations. In particular, connected noiseless gaps reconstruct every reward up to an additive constant. The same data then permit direct restricted value optimization. That baseline is acknowledged in §8.

## 2. Explicit scalar bias, covariance and deployment-risk coefficient

At the population optimizer write $s_e=\sigma(a_e\theta_\lambda)$, $h_e=s_e(1-s_e)$, and $t_e=h_e(1-2s_e)$. The sample estimating function and its derivatives are

$$
\psi=a_e[(s_e-Z)+\lambda(a_e\theta_\lambda-G)],\qquad
\psi'=a_e^2(h_e+\lambda),\qquad
\psi''=a_e^3t_e.
$$

Define the conditional mean residual, sample curvature, and population moments

$$
m_e=a_e[(s_e-p_e)+\lambda(a_e\theta_\lambda-g_e)],\qquad k_e=a_e^2(h_e+\lambda),
$$

$$
H=\mathbb E_q k_e,\qquad T=\mathbb E_q(a_e^3t_e),\qquad
C=\mathbb E_q\left[m_e^2+a_e^2\{p_e(1-p_e)+\lambda^2v_e\}\right],\qquad
D=\mathbb E_q(m_ek_e).
$$

$\mathbb E_qm_e=0$ by population stationarity, so $C=\mathbb E\psi^2=\operatorname{Var}\psi$ and $D=\operatorname{Cov}(\psi,\psi')$. The $m_e^2$ term is design variance from misspecification; omitting it changes the experiment. With correlated channels, replace the braces by

$$
p_e(1-p_e)+\lambda^2v_e+2\lambda\operatorname{Cov}(Z,\xi\mid e).
$$

The plus sign follows because both stochastic terms appear with a minus sign in $\psi$. The conditional mean-zero assumption still makes $D$ unchanged. An arbitrary dependent channel requires its own joint tail assumption for the probability bound below.

Let $A=P_n\psi$, $U=P_n\psi'-H$, and $\delta=\widehat\theta_\lambda-\theta_\lambda$. On the interior-root event, Taylor expansion of the empirical score gives

$$
\delta=-\frac A H+\frac{AU}{H^2}-\frac{TA^2}{2H^3}+\text{remainder}.
$$

Because $\mathbb E A=0$, $\mathbb E(AU)=D/n$, and $\mathbb E A^2=C/n$,

$$
\boxed{b_\lambda=\frac D{H^2}-\frac{TC}{2H^3}},\qquad
\mathbb E\delta=\frac{b_\lambda}{n}+O(n^{-3/2}),\qquad
\mathbb E\delta^2=\frac C{nH^2}+O(n^{-3/2}).
$$

Consequently, with $A_\lambda=R(\theta_\lambda)$,

$$
\boxed{\mathbb ER(\widehat\theta_\lambda)
=A_\lambda+\frac{V_\lambda}{n}+O(n^{-3/2}),\qquad
V_\lambda=-J_\tau'(\theta_\lambda)b_\lambda
-\frac12J_\tau''(\theta_\lambda)\frac C{H^2}.}
$$

All constants depend on the specified compact region, design, noise bound and uniform interior margin. §3 gives a computable uniform remainder. $V_\lambda$ is not necessarily nonnegative: the pseudo-true loss optimizer need not maximize deployment value. Neither the estimator bias nor the sign of the deployment Hessian can be discarded.

For the scalar exponential family, the derivatives needed for exact computation are

$$
J_\tau'=\operatorname{Cov}_{\pi_\theta}(f,r)-\tau\theta\operatorname{Var}_{\pi_\theta}f,
$$

$$
J_\tau''=\mathbb E_{\pi_\theta}[(f-\bar f)^2(r-\bar r)]
-\tau\operatorname{Var}_{\pi_\theta}f
-\tau\theta\mathbb E_{\pi_\theta}[(f-\bar f)^3].
$$

These avoid finite-difference cancellation. In a stratified design with fixed pair counts, curvature is deterministic at a fixed parameter: the random-design $D$ term and $m_e^2$ variance cannot simply be copied from the iid-pair formula. The simulations below use multinomial counts and therefore match this formula.

## 3. Uniform remainder: an actual assumption and bound, not an $o(1/n)$ placeholder

Here is one sufficient finite-action setting. Assume $0<a_{\min}\le |a_e|\le a_{\max}$ on the sampled support; $v_e\le v_{\max}$; $|g_e|\le g_{\max}$; and $\theta_\lambda$ has distance at least $m>0$ from the boundary, uniformly over the family being studied. Fix $\lambda$ or a bounded interval of weights. Define

$$
h_* = \sigma(a_{\max}M)[1-\sigma(a_{\max}M)],\quad
h_0=a_{\min}^2(h_*+\lambda),\quad
K_2=a_{\max}^2(1/4+\lambda),\quad
K_3=a_{\max}^3/4,\quad K_4=3a_{\max}^4/8.
$$

Every empirical objective is $h_0$-strongly convex. $K_3$ and $K_4$ safely bound $|\psi''|$ and $|\psi'''|$; the squared-gap term contributes neither derivative. Let

$$
B_\psi=a_{\max}[1+\lambda(a_{\max}M+g_{\max})],\quad
\sigma_A^2=B_\psi^2+\lambda^2a_{\max}^2v_{\max},\quad
\sigma_U=K_2/2,\quad \sigma_W=K_3.
$$

$A$ is sub-Gaussian with proxy $\sigma_A^2/n$. The bounded part of the score lies in $[-B_\psi,B_\psi]$ and has mean zero; conditionally independent Gaussian gap noise supplies the second proxy term. Hoeffding gives sub-Gaussian proxies $\sigma_U^2/n$ and $\sigma_W^2/n$ for $U$ and $W=P_n\psi''-T$. In particular, the loose but valid moment bounds $\mathbb EA^4\le16\sigma_A^4/n^2$ and corresponding bounds for $U,W$ suffice.

Strong convexity and the constrained first-order inequality give **even on the boundary**

$$
|\delta|\le |A|/h_0.
$$

Thus the non-interior event has probability at most

$$
p_n=\min\{1,2\exp[-nh_0^2m^2/(2\sigma_A^2)]\}.
$$

For an interior empirical root, writing $e_1=\delta+A/H$, a linear-score Taylor bound gives

$$
|e_1|\le\frac{|U||A|}{h_0^2}+\frac{K_3A^2}{2h_0^3}.
$$

Subtracting the displayed second-order root approximation yields

$$
\left|\delta+\frac AH-\frac{AU}{H^2}+\frac{TA^2}{2H^3}\right|
\le \frac{U^2|A|}{h_0^3}
+\frac{3K_3|U|A^2}{2h_0^4}
+\frac{K_3^2|A|^3}{2h_0^5}
+\frac{|W|A^2}{2h_0^3}
+\frac{K_4|A|^3}{6h_0^4}.
$$

The expectation is bounded by $K_b n^{-3/2}$, where an explicit safe choice is

$$
K_b=\frac{4\sigma_U^2\sigma_A}{h_0^3}
+\frac{6K_3\sigma_U\sigma_A^2}{h_0^4}
+\frac{4K_3^2\sigma_A^3}{h_0^5}
+\frac{2\sigma_W\sigma_A^2}{h_0^3}
+\frac{4K_4\sigma_A^3}{3h_0^4}.
$$

On the non-interior event, $|\delta|\le2M$, $|U|\le K_2$, and Cauchy–Schwarz control the discarded polynomials. A corresponding bias remainder is

$$
B_b(n)=2Mp_n+
\left(\frac1{h_0}+\frac{K_2}{h_0^2}\right)\frac{\sigma_A}{\sqrt n}\sqrt{p_n}
+\frac{2K_3\sigma_A^2}{h_0^3n}\sqrt{p_n}.
$$

The second-moment remainder is at most $K_s n^{-3/2}+B_s(n)$ with

$$
K_s=\frac{8\sigma_U\sigma_A^2}{h_0^3}+\frac{8K_3\sigma_A^3}{h_0^4},\qquad
B_s(n)=4M^2p_n+\frac{4\sigma_A^2}{h_0^2n}\sqrt{p_n}.
$$

If $L_3=\sup_{\Theta}|R'''|$, the fully specified risk remainder obeys

$$
\left|\mathbb ER(\widehat\theta_\lambda)-A_\lambda-\frac{V_\lambda}{n}\right|
\le \frac{|R'|K_b+|R''|K_s/2+4L_3\sigma_A^3/(3h_0^3)}{n^{3/2}}
+|R'|B_b(n)+\frac{|R''|}{2}B_s(n),
$$

where the two derivatives outside the supremum are evaluated at $\theta_\lambda$. This bound is conservative, especially when $h_0$ is small; it is sufficient for the uniform local-alternative theorem. It does **not** certify that the large-sample fixed-$\epsilon$ threshold is exact at $n=8$ or $32$.

**Audit of the derivation.** The covariance term uses iid observations; score stationarity holds only on the interior event; Gaussian labels are unbounded despite bounded features/rewards; the boundary contribution is exponential rather than silently zero; the loss has fourth derivatives on the compact set; deployment has a uniformly bounded third derivative there. These assumptions are all satisfied uniformly for sufficiently small $\epsilon$ in §5, with a fixed interval such as $[-2,2]$.

## 4. Correct high-probability finite-sample bound

Under §3's assumptions, with probability at least $1-\delta$,

$$
\boxed{|\widehat\theta_\lambda-\theta_\lambda|
\le r_n:=\frac{\sigma_A}{h_0}\sqrt{\frac{2\log(2/\delta)}n}.}
$$

This follows directly from sub-Gaussian concentration of the score at its population root and strong monotonicity, including constrained boundary solutions. If $r_n<m$, the empirical optimum is interior. A sharper score proxy can replace the displayed safe one when actual conditional moments are known.

If self-pairs or zero feature gaps occur, a uniform $a_{\min}>0$ is unavailable. Let $\mu_2=\mathbb E_q a_e^2>0$. With probability at least $1-\delta$, provided the denominator is positive,

$$
|\widehat\theta_\lambda-\theta_\lambda|
\le\frac{\sigma_A\sqrt{2\log(4/\delta)/n}}
{(h_*+\lambda)\left[\mu_2-a_{\max}^2\sqrt{\log(2/\delta)/(2n)}\right]}.
$$

This uses separate $\delta/2$ events for the score and empirical second feature moment. It is inappropriate to assume a positive empirical eigenvalue without either a deterministic bound or such an event.

For $L_2=\sup_{\Theta}|J_\tau''|$, either parameter bound implies

$$
|R(\widehat\theta_\lambda)-A_\lambda|
\le |J_\tau'(\theta_\lambda)|r_n+\frac{L_2}{2}r_n^2.
$$

The leading high-probability value term is ordinarily $n^{-1/2}$ at a misspecified surrogate optimum, even though the **expected signed** expansion is $1/n$. Claiming a quadratic value guarantee without deployment stationarity would be wrong. Union bounds give a conservative simultaneous comparison of binary and fused estimates; paired analysis can be sharper.

## 5. Explicit provable crossover in the rescaled three-action family

Use uniform reference, uniform sampling over the three distinct unordered edges, $\tau=1$, and

$$
f=(0,7/10,1),\qquad r=\epsilon(0,1,2/5),\qquad \epsilon>0.
$$

Keep the compact interval fixed. For all sufficiently small $\epsilon$, both population loss optima are unique interior roots near zero. The observations have accurate mean gaps and conditionally independent Gaussian noise of constant variance $v\ge0$; $v=0$ is the exact-gap case. Let $d=f_i-f_j$, $g_0=(r_i-r_j)/\epsilon$ and

$$
K=\mathbb E d^2=\frac{79}{150},\qquad
k=\frac{\mathbb E(dg_0)}K=\frac{46}{79}.
$$

Since $\sigma(x)=1/2+x/4-x^3/48+O(x^5)$, the odd scalar score and the implicit function theorem give

$$
\theta_0=\epsilon k+\epsilon^3b_3+O(\epsilon^5),\qquad
\theta_\lambda=\epsilon k+\frac{\epsilon^3b_3}{1+4\lambda}+O(\epsilon^5),
$$

$$
b_3=\frac{k^3\mathbb E d^4-\mathbb E(dg_0^3)}{12K}
=-\frac{3726}{156025}.
$$

The rating fit moves toward its own linear projection $\epsilon k$, but that is not generally the deployment-value projection. At the uniform policy, let $\mu=\mathbb E f$, $\bar r_0=\mathbb E r_0$, and $r_0=(0,1,2/5)$. Expanding the deployment gradient,

$$
J'(\theta_0)=c_2\epsilon^2+O(\epsilon^3),\qquad
c_2=k\mathbb E[(f-\mu)^2(r_0-\bar r_0)]
-k^2\mathbb E[(f-\mu)^3]
=-\frac{2898}{156025}.
$$

Hence the population regret difference has a fully computed leading structural term:

$$
\boxed{A_\lambda-A_0=P_\lambda\epsilon^5+O(\epsilon^6),\qquad
P_\lambda=\frac{10797948}{24343800625}\frac{4\lambda}{1+4\lambda}>0.}
$$

**Proof of the fifth order:** the optimizer displacement is $-4\lambda b_3\epsilon^3/(1+4\lambda)+O(\epsilon^5)$, while the deployment gradient at the binary optimum is $c_2\epsilon^2+O(\epsilon^3)$. The Taylor product $-J'\Delta\theta$ supplies $P_\lambda\epsilon^5$; the quadratic displacement term is $O(\epsilon^6)$. This distinguishes the residual geometry from an unexplained approximation-error symbol.

At $\epsilon=0$, $\theta_\lambda=0$, $m_e=T=D=b_\lambda=J'=0$, and

$$
H_\lambda=K(1/4+\lambda),\quad C_\lambda=K(1/4+\lambda^2v),\quad
\operatorname{Var}_{\mathrm{uniform}}f=\frac{79}{450},
$$

$$
V_0\longrightarrow\frac23,\qquad
V_\lambda\longrightarrow\frac16\frac{1/4+\lambda^2v}{(1/4+\lambda)^2}.
$$

Consequently a variance advantage exists exactly when

$$
\boxed{v<4+2/\lambda.}
$$

This threshold concerns the fixed penalty weight $\lambda$. The correctly specified Gaussian likelihood weight would be $\lambda=1/v$ when $v>0$; arbitrary weights represent different uses of the same channel. Exact-gap $v=0$ permits a finite trust weight but does not mean a literal positive-variance Gaussian MLE.

### Crossover theorem, with a meaningful quantifier

Fix $\lambda>0$ and $v<4+2/\lambda$. Put

$$
W_{\lambda,v}=\frac23-\frac16\frac{1/4+\lambda^2v}{(1/4+\lambda)^2}>0,\qquad
t_* = W_{\lambda,v}/P_\lambda.
$$

For **any** positive sample sequence satisfying $n(\epsilon)\epsilon^5\to t\in(0,\infty)$,

$$
\boxed{\lim_{\epsilon\downarrow0}\epsilon^{-5}
\left[\mathbb ER(\widehat\theta_\lambda)-\mathbb ER(\widehat\theta_0)\right]
=P_\lambda-\frac{W_{\lambda,v}}t.}
$$

Thus for every fixed $t<t_*$ the fused estimator has strictly lower expected deployment regret for all sufficiently small $\epsilon$; for every fixed $t>t_*$ it has strictly higher regret for all sufficiently small $\epsilon$. This proves an actual sample-budget sign change in a specified family. It does not assert a unique monotone crossing in $n$ or settle $t=t_*$.

**Uniform proof:** smooth bounded population roots converge to zero, giving a common interior margin and strong-convexity bound. §3 supplies a uniform $O(n^{-3/2})$ expected-risk remainder plus an exponentially small boundary remainder. The projection remainder divided by $\epsilon^5$ is $O(\epsilon)$; the covariance/bias coefficient is continuous at zero; and $n^{-3/2}/\epsilon^5=O(\epsilon^{5/2})$ when $n\epsilon^5\to t$. Every discarded term therefore vanishes. The bias term was included before taking this limit; its disappearance at the leading local limit does not justify dropping it at a fixed nonzero reward scale.

For exact gaps and $\lambda=0.1$,

$$
P_{0.1}=\frac{3085128}{24343800625}=0.00012673156700238956,\quad
V_0=\frac23,\quad V_{0.1}=\frac{50}{147},\quad
t_*=\frac{48687601250}{18896409}=2576.5531032906833.
$$

The large constant is honest: the structural penalty is fifth order and small in this family. Near-realizable behavior can therefore require many observations to reveal it. This is a comparative-estimator theorem, not a claim that one should prefer a deliberately restricted reward fit to a direct-value baseline.

### General matched-design moment law: arbitrary finite action count and temperature

The same theorem is not restricted to three actions. Fix any finite action set, full-support reference $\pi_0$, distinct scalar features $f_i$, a reward direction $h_i$, and $r_i=\epsilon h_i$. Draw independent reference actions and condition on a distinct pair. Equivalent fixed orientation uses

$$
q(i,j)=\frac{2\pi_0(i)\pi_0(j)}{1-\sum_l\pi_0(l)^2},\qquad i<j.
$$

Put $d=f_i-f_j$, $g_h=h_i-h_j$, $F_0=\operatorname{Var}_{\pi_0}f>0$, and

$$
\mu_2=\mathbb E_qd^2,\quad \mu_4=\mathbb E_qd^4,\quad
\mu_{fh^3}=\mathbb E_q(dg_h^3),\quad
k=\frac{\mathbb E_q(dg_h)}{\mu_2}
=\frac{\operatorname{Cov}_{\pi_0}(f,h)}{F_0}.
$$

The last equality follows by expanding independent-pair moments; conditioning multiplies numerator and denominator by the same constant. Define the explicit structural coefficients

$$
b_3=\frac{\mu_4k^3-\mu_{fh^3}}{12\mu_2},\qquad
c_2=k\kappa_{\pi_0}(f,f,h)-k^2\kappa_{\pi_0}(f,f,f),
$$

where the two third cumulants are $\mathbb E[(f-\bar f)^2(h-\bar h)]$ and $\mathbb E[(f-\bar f)^3]$. For any fixed $\tau>0$ and constant Gaussian variance $v$,

$$
\theta_0=\frac{\epsilon k+\epsilon^3b_3}{\tau}+O(\epsilon^5),\qquad
\theta_\lambda=\frac{\epsilon k+\epsilon^3b_3/(1+4\lambda)}{\tau}+O(\epsilon^5),
$$

$$
J_\tau'(\theta_0)=\frac{c_2\epsilon^2}{\tau}+O(\epsilon^3),\qquad
A_\lambda-A_0=\underbrace{\frac{c_2b_3}{\tau^2}\frac{4\lambda}{1+4\lambda}}_{P_{\lambda,\tau}}\epsilon^5+O(\epsilon^6),
$$

$$
V_0\to\frac{2F_0}{\tau\mu_2},\qquad
V_\lambda\to\frac{F_0}{2\tau\mu_2}\frac{1/4+\lambda^2v}{(1/4+\lambda)^2}.
$$

**Proof:** substitute $u=\tau\theta$ into the scalar score. Its odd Taylor expansion is identical to the previous calculation with $K$ replaced by $\mu_2$. In the deployment gradient, $\epsilon\operatorname{Cov}_{\pi_0}(f,h)-\epsilon kF_0=0$ by matched design. The next term from the exponential-family perturbation $\theta=\epsilon k/\tau+O(\epsilon^3)$ is $c_2\epsilon^2/\tau$. Multiplying by the optimizer displacement gives the fifth-order coefficient. At $\epsilon=0$, $H=\tau^2\mu_2(1/4+\lambda)$, $C=\tau^2\mu_2(1/4+\lambda^2v)$ and $J_\tau''=-\tau F_0$, yielding the displayed statistical constants. Compact finite support supplies §3's uniform remainder. This establishes the local-alternative limit with $P_{\lambda,\tau}$ and $W=V_0-V_\lambda$ in the same way.

For $c_2b_3>0$ and $v<4+2/\lambda$, the explicit threshold is $t_*=W/P_{\lambda,\tau}$ and the same benefit-then-harm theorem holds. The three-action coefficients above demonstrate a nonempty class. Strict inequalities and continuous finite moments imply an open neighborhood of nearby features, rewards and full-support references with the same signs: no exact rational tuning is needed to retain the regime. This does not prove prevalence under a distribution of real task geometries.

The moment law also separates four leading regimes: $P>0,W>0$ gives small-budget benefit/large-budget harm; $P<0,W>0$ gives benefit on both sides of this local scale; $P>0,W<0$ gives harm on both sides; $P<0,W<0$ gives small-budget harm/large-budget benefit. If either coefficient vanishes, the theorem does not infer a crossover and higher-order terms must be analyzed.

**Design matching is substantive.** For an arbitrary fixed pair law, retain $k=\mathbb E_q(dg_h)/\mu_2$. If $c_1=\operatorname{Cov}_{\pi_0}(f,h)-kF_0\ne0$, then $J_\tau'(\theta_0)=c_1\epsilon+O(\epsilon^2)$ and instead

$$
A_\lambda-A_0=\frac{c_1b_3}{\tau}\frac{4\lambda}{1+4\lambda}\epsilon^4+O(\epsilon^5).
$$

When this coefficient is positive and the statistical advantage is positive, the corresponding local crossover uses $n\epsilon^4\to t$. Thus the fifth-order law encodes a cancellation caused by pair/deployment moment matching. It cannot be claimed for an arbitrary scored corpus without checking that cancellation. Equal-feature pairs can be removed as uninformative score terms and the informative sample count declared; alternatively the empirical-design event in §4 must be included in the uniform proof.

## 6. Numerical verification, frozen before simulation

[Verification script](../verification/fusion_local_alternatives.py) independently computes the rational coefficients and analytic $H,T,C,D,J',J''$. Its [configuration](../verification/fusion_local_alternatives_config.json) was written before random draws: $\epsilon=0.2,0.4,0.8$; $n=\operatorname{round}(c\,t_*/\epsilon^5)$ for $c=0.25,0.5,1,2,4$; 4,096 paired draws per cell; seed 2026101407; exact gaps; $\lambda=0.1$; $\Theta=[-2,2]$. Each paired draw shares its multinomial pair counts and BT bits across estimators. Sufficient statistics allow millions of observations per draw without generating individual rows. Every prescribed cell was reported; no tuning or removal of cells occurred.

The sign convention is $\Delta R=R_{\mathrm{fused}}-R_{\mathrm{binary}}$: **negative favors fusion**. Intervals below are approximate 95% Monte Carlo intervals for the paired mean, not confidence intervals over real prompts or empirical support for conference acceptance.

| $\epsilon$ | Factor $c$ | $n$ | Mean $\Delta R$ | 95% Monte Carlo interval |
|---:|---:|---:|---:|---:|
| .2 | .25 | 2,012,932 | $-1.322\times10^{-7}$ | $[-1.461,-1.182]\times10^{-7}$ |
| .2 | .5 | 4,025,864 | $-4.616\times10^{-8}$ | $[-5.526,-3.705]\times10^{-8}$ |
| .2 | 1 | 8,051,728 | $-2.137\times10^{-9}$ | $[-8.204,3.929]\times10^{-9}$ |
| .2 | 2 | 16,103,457 | $1.853\times10^{-8}$ | $[1.433,2.273]\times10^{-8}$ |
| .2 | 4 | 32,206,914 | $2.681\times10^{-8}$ | $[2.382,2.980]\times10^{-8}$ |
| .4 | .25 | 62,904 | $-4.347\times10^{-6}$ | $[-4.697,-3.996]\times10^{-6}$ |
| .4 | .5 | 125,808 | $-1.466\times10^{-6}$ | $[-1.674,-1.258]\times10^{-6}$ |
| .4 | 1 | 251,617 | $-1.220\times10^{-7}$ | $[-2.549,0.110]\times10^{-7}$ |
| .4 | 2 | 503,233 | $4.805\times10^{-7}$ | $[3.893,5.716]\times10^{-7}$ |
| .4 | 4 | 1,006,466 | $8.723\times10^{-7}$ | $[8.104,9.341]\times10^{-7}$ |
| .8 | .25 | 1,966 | $-1.417\times10^{-4}$ | $[-1.506,-1.329]\times10^{-4}$ |
| .8 | .5 | 3,932 | $-5.722\times10^{-5}$ | $[-6.241,-5.203]\times10^{-5}$ |
| .8 | 1 | 7,863 | $-1.039\times10^{-5}$ | $[-1.339,-0.739]\times10^{-5}$ |
| .8 | 2 | 15,726 | $1.193\times10^{-5}$ | $[1.011,1.376]\times10^{-5}$ |
| .8 | 4 | 31,452 | $2.304\times10^{-5}$ | $[2.183,2.426]\times10^{-5}$ |

No estimator reached the compact boundary in these 61,440 paired draws. All three scales show benefit below the leading threshold and harm above it, with the center increasingly close to zero as $\epsilon$ shrinks. At $\epsilon=.8$ the center remains beneficial: the leading local threshold is an asymptotic prediction, not an exact finite-$\epsilon$ formula. The fixed-$\epsilon$ expansion using actual population coefficients tracks the observed mean more closely; see all predictions in the [cell CSV](../verification/fusion_local_alternatives_results.csv); raw paired draws are regenerable from the seed and configuration, and independent shrinking-scale checks are stored in [the canonical moment-law result](../verification/fusion_moment_law_checks_results.json).

The verified population ratio $(A_{.1}-A_0)/\epsilon^5$ is .000102694, .000115568, .000121457, .000124183, .000125481 at $\epsilon=.8,.4,.2,.1,.05$, tending toward the exact coefficient .000126732. Very small-$\epsilon$ double-precision value differences eventually lose accuracy; the exact rational coefficients provide the algebraic verification independently of that subtraction.

## 7. Fixed reward scale three: useful experiment, different guarantee

The root Sprint 1 experiment uses the same family at $\epsilon=3$: $r=(0,3,1.2)$. Its population binary and $\lambda=.1$ parameters are 1.34897816196 and 1.49426322641. From its [coefficient CSV](../verification/fusion_sprint1/asymptotic_coefficients.csv),

$$
A_{.1}-A_0=0.010850426778832523,\qquad
V_{.1}-V_0=0.5028010988994722-0.9725888170727148
=-0.4697877181732426.
$$

The fixed-scale large-$n$ expansion therefore predicts a zero near $n=43.2967$. This is an approximate finite-scale diagnostic, not a proof that the true transition is at that integer. Root's recorded experiment finds $J_{\mathrm{fused}}-J_{\mathrm{binary}}$ positive at $n=8$ and $32$, negative at $128$ and 16,384, consistent with a finite-budget benefit that disappears. Those observations are distinct from the local-alternative theorem and the 15-cell verification above.

At this scale the estimator-bias contribution matters: binary $b=1.81979644801$, $J'=-.06217343039$; fused $b=.60961263744$, $J'=-.08664974182$. Omitting $-J'b$ removes about .113143 and .052823 from the respective $V$ coefficients and changes the threshold. A variance-only account would consequently mispredict the comparison. The root CSV uses numerical deployment derivatives; this note and the additional script supply analytic formulas for independent checking.

## 8. Baselines, limits and research contribution boundary

**The exact-gap reconstruction baseline is decisive.** Once the observed comparison graph is connected, exact gaps determine $r$ up to a constant. Maximizing the known $J_\tau$ over the same scalar policy class yields zero restricted regret without additional labels. In the three-edge design, a disconnected observed graph requires all draws to use one edge, an event of probability $3(1/3)^n$ for $n\ge1$. Thus the exact-gap large-sample fusion penalty is avoidable with the same information. This theorem cannot support an information-theoretic claim that accurate feedback itself is harmful or new target-policy labels are necessary.

For noisy gaps, separate saturated reward reconstruction and direct restricted optimization are required controls. For independent Gaussian gaps, the unrestricted finite-action reward model can use graph least squares; reward/policy capacity should not be conflated. REBEL-style policy-relevant regression and the local current-policy gap identity are also meaningful comparison points, but are not analyzed by this frozen iid-pair theorem.

**What is proved here:** the scalar iid M-estimator's explicit $D$ and third-derivative bias terms; a sufficient uniform expected-risk remainder with boundary control; a correct sub-Gaussian finite-sample parameter/value guarantee; an exact fifth-order structural penalty in a specified family; and an actual local-alternative sample crossover with an explicit noise threshold and exact coefficients. The numerical check confirms the algebra and the predicted signs without establishing novelty or external relevance.

**What remains needed for a strong paper:** characterize broader feature/reward classes where the first nonzero structural order and sign are predictable; show whether such geometry appears naturally in scored language-task selectors; compare exact and noisy reward reconstruction under an honest common observation/cost contract; extend the comparative theory beyond a tuned scalar example; and clear collisions with existing misspecified likelihood, surrogate-risk, multi-fidelity and preference/rating work. Generic sandwich covariance, ordinary Taylor bias and the fact that learners can ignore an extra channel are not new contributions.

## 9. Primary-source anchoring and read depth

- [Direct Preference Optimization with Rating Information: Practical Algorithms and Provable Gains, arXiv:2602.00603](https://arxiv.org/html/2602.00603v1): ML-RDPO Eq. (13) motivates the summed Bernoulli–Gaussian likelihood and the mapping $\lambda=1/V$ under matching normalization. The local source's conditional independence/realizability assumptions do not imply our misspecified deployment comparison. The source-method mapping is documented in [the theorem comparison](fusion_theorem_novelty_matrix.md); this derivation does not claim a new full-paper read.
- [Coverage Improvement and Fast Convergence of On-policy Preference Learning, arXiv:2601.08421](https://arxiv.org/abs/2601.08421): local full text was read in the earlier coverage task; main preference theorem assumptions impose true BT reward and representable Boltzmann target. Its coverage-rate theorem is not a misspecified-fusion bias theorem; coverage improvement and projection harm can coexist.
- [Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO, arXiv:2505.19770](https://arxiv.org/html/2505.19770v1), §3.1 Remark 2 and Appendix C: exact preference modeling alone does not remove policy misspecification; this is a collision with any broad claim that cardinal feedback reveals a previously unknown reward/policy-capacity distinction. Relevant primary sections were read in the earlier novelty task.
- [REBEL, arXiv:2404.16767](https://arxiv.org/html/2404.16767v2), Algorithm 1 and Theorem 1: relative reward-gap regression with moving policy references and an explicit approximation/generalization contract; a necessary existing competitor for a proposed correction, not the same fixed-reference statistical estimator. These sections were read in the earlier coverage task.

No new source search was needed to derive the algebra. Literature novelty remains a separate proof-level task; the source evidence here does not establish conference readiness.
