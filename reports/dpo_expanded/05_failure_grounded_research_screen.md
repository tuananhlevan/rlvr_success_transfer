# Research direction reassessment: failures, mechanisms, and existing repairs

**Date:** 15 October 2026. **Decision:** no new method proposal has passed this screen. Several consequential problems are real; the evidence below does not yet establish a sufficiently original solution or a failure that survives the strongest relevant remedies.

**Later research update:** this screen remains the record of rejected alternatives. The subsequent [policy-capacity study](../../studies/policy_capacity_feedback/research_report.md) develops a continuous-class information characterization and reproducible finite-model checks. Its Section 11 records the later novelty and practical-repair audit; a full-project recommendation is withheld.

This is the canonical result of the renewed analysis after rejecting generator routing. The [previous screen](04_source_grounded_problem_screen.md) records that rejected proposal. The [original corpus analysis](01_expanded_literature_analysis.md) remains the detailed analysis of the spreadsheet papers.

## 1. What this analysis tested

The question was whether the literature supports a research direction with all four properties:

1. A consequential failure documented in an actual method, experiment, engineering setup, or public discussion.
2. An explanation of the failure that predicts something beyond the original observation.
3. A substantive remaining gap after relevant existing solutions are considered.
4. A feasible experiment capable of disproving the explanation, followed by theory whose assumptions match that experiment.

The search expanded beyond DPO variants into multi-teacher distillation, self-training, reasoning quality, personalized alignment, and tool-using agents. This is a targeted narrative review, not an exhaustive systematic review. It does not establish that no publishable problem exists in these areas.

**Evidence boundaries.** Primary manuscripts and their relevant appendices were examined through web access. Reading depth varies and is stated below. Version-specific conclusions use the linked version. The central URLVR, TEMPO, reasoning-quality, and BLINDSPOT records were also checked against their arXiv metadata. The personalized-alignment manuscript was first posted in December 2025 and revised in January 2026. This pass did not newly access OpenReview reviews, rebuttals, or metareviews, and made no OpenReview API requests; the [earlier discussion audit](03_openreview_discussion_audit.md) retains its existing coverage. Bounded literature checks by workers in this session helped identify collisions; they are not independent external peer review.

**No new model experiment was run.** All numerical results below belong to the cited papers. The mathematical calculations in Section 4 explain mechanisms; they are neither new empirical evidence nor claimed novel theorems. No acceptance probability is assigned.

## 2. Main conclusion

The most consequential failures found concern **lost capabilities, misleading training rewards, and unsafe executed actions**. These have clearer practical consequences than saving a router's inference cost. However, that establishes the importance of a problem area, not the originality of our proposed contribution.

| Problem area | Consequence | What prevents promotion now? |
|---|---|---|
| Cross-domain teacher transfer | Improving one skill can damage another useful capability. | Recent preservation, recovery, and update-aware methods already target the mechanism we would otherwise propose. |
| Intrinsic self-training | More optimization can reduce mathematical accuracy while proxy reward rises. | The failure is well documented, but externally grounded verification and critic recalibration already provide successful alternatives. A remaining operational regime must be demonstrated. |
| Long-horizon agent safety | An agent can execute an operation after its authorization has become invalid. | A trusted execution-time check is a strong practical solution for explicit authorization predicates. A new alignment loss has not been shown necessary. |
| Personalized alignment | A better preference classifier may produce no better user-facing generation. | The source already evaluates this mismatch and finds strong simple prompting/retrieval baselines. |
| Reasoning quality under outcome RL | Accuracy can improve while written reasoning becomes less useful for verification. | The source itself improves its reasoning metrics with expert-trace SFT and auxiliary rewards. |

An existing remedy being relevant is **not proof that the entire problem is solved**. Conversely, an untested combination of settings is **not evidence that a new method is needed**. The current result is insufficient evidence for promoting a proposal, rather than a declaration that these research areas are closed.

## 3. Paper-level evidence and intuition

### 3.1 Multi-teacher distillation: a real capability problem in a crowded solution space

| Paper and reading scope | Intuition | Specific finding or contribution | Consequence for a new proposal |
|---|---|---|---|
| [Every Coin Has Two Sides](https://arxiv.org/html/2608.16647v2), v2; §§4.1–4.2 | Training on one subject can change the general reasoning habits used for other subjects. Prompt labels do not isolate the affected parameters. | Teacher transfer can help or hurt untrained domains; teacher mixtures and sequential training produce capability tradeoffs. | This is evidence of real negative transfer. It does not establish that a new teacher router, loss weight, or gradient projection is needed. |
| [Composing What Each Teacher Learned / Δ-MOPD](https://arxiv.org/html/2610.10460v1), v1; §5.4, Appendix D.1, limitations | Transfer what changed during teacher specialization, rather than copying the teacher's entire inherited distribution. | Teacher-relative shifts reduce a reported phase-order gap from 10.50 to 6.42 points. Phase switches also change prompt domain and carry optimizer state. | A residual remains in this setup, but attributing it specifically to incompatible teacher composition would require separating curriculum and optimizer effects. |
| [CaMOPD](https://arxiv.org/html/2605.27115v2), v2; §§3.2–4 | Learn a new specialist capability while periodically recovering capabilities displaced by that learning. | Alternates recovery/preservation and uses capability-gap selection, including settings without original teacher training data. | Generic recovery, replay-style preservation, and gap-based teacher selection are occupied contributions. |
| [TrustMOPD](https://arxiv.org/html/2609.23697v3), v3; §§3–4, Appendix A.5/F | A teacher's change from its precursor is a proxy for what its specialist training contributed. | Uses RL displacement to guide trust. Its theoretical interpretation assumes suitable KL-regularized optimal teachers; displacement is not a general certificate of truth. | Heterogeneous provenance is a limitation, but the limitation alone does not demonstrate a consequential failure. The proposed setting must also include precursor-free baselines. |
| [Slow-Fast MOPD](https://arxiv.org/abs/2610.02324), v1; abstract-level collision check | Maintain a slowly moving capability anchor, and suppress update components that drive the student away from it. | Proposes an EMA slow model and selectively removes conflicting teacher-log-probability update components. | Directly collides with a new “project updates to preserve general capability” proposal. Full comparative evaluation was not audited here. |
| [USA: Update-aware SAM](https://arxiv.org/abs/2609.34225), abstract-level collision check | The shape of the optimization update can matter more than which specialist endpoint is selected. | Reports negative transfer for evaluated merging operators and uses update-aware sharpness optimization to improve cross-domain OPD. | Optimizer geometry is already an explicit solution direction. The reported gains are not independently reproduced here. |

**Assessment.** This is an important capability-preservation problem. We do not currently have a distinct mechanism or documented residual failure that distinguishes a new approach from these methods. Domain routing alone would also miss the source's central observation: training effects cross domain boundaries.

### 3.2 Self-training: reward can become detached from competence

| Paper and reading scope | Intuition | Specific finding or contribution | Consequence for a new proposal |
|---|---|---|---|
| [How Far Can Unsupervised RLVR Scale LLM Training?](https://arxiv.org/html/2603.08660v1), §§3–7, Appendix B.2–B.3 | Repeatedly rewarding confidence or agreement can reinforce an initially wrong preference. | Five intrinsic-reward methods exhibit degradation; combined stabilizing hyperparameters still collapse around 1,000 steps in the reported extended run. Small-dataset adaptation and external self-verification behave differently. | Strong evidence against “just tune the optimizer.” It is not evidence that all self-improvement is impossible, or that external verification fails. |
| [Dual Consensus](https://arxiv.org/html/2603.16223v1), §2, Appendix D.2 | A wrong answer can be popular but less robust to a deliberate model intervention. | Uses an unlearning intervention and a combined consensus signal to address spurious majorities; the theory assumes a relevant robustness distinction between true and wrong modes. | A new agreement-only weighting scheme must engage with this intervention baseline and its assumptions. |
| [TEMPO](https://arxiv.org/html/2604.19295v1), §§3–4.5, §6 | Periodically reconnect the critic to labeled evidence so actor-generated confidence does not become the sole judge of progress. | Alternates actor refinement and labeled-source critic recalibration; reports sustained gains. Limitations include source-label dependence and absent formal alternating-optimization convergence guarantees. | A generic periodically calibrated critic is already proposed. OOD dependence is acknowledged, but this review did not establish a realistic shift where TEMPO fails and a new method succeeds. |
| [Rethinking Groups in Critic-Free RLVR](https://arxiv.org/html/2606.17250v1), §§2–3, §5.1 | Penalizing an unsuccessful trajectory can also suppress useful tokens it shares with successful ones. | Its single-rollout failure is stabilized in the reported setup by reducing the negative-update coefficient to 0.25; group baselines and token filtering provide further remedies. | Reject another negative-gradient damping or token-filter proposal as the main research contribution. |

**Assessment.** Intrinsic self-training collapse is the strongest directly evidenced failure in this sweep. Its importance survives the baseline check; our proposed novelty does not yet do so. Any future study must state which external evidence is available. Labeled recalibration, executable verification, and purely intrinsic feedback are different information regimes.

### 3.3 Tool agents: safety is about the action and current state

| Paper and reading scope | Intuition | Specific finding or contribution | Consequence for a new proposal |
|---|---|---|---|
| [BLINDSPOT](https://arxiv.org/html/2609.16305v1), §6.2, Tables 6–7, Appendix B/D | A refusal early in a conversation does not guarantee that a later tool action remains safe. Authorization may change while the conversation continues. | Evaluates adaptive long-horizon interactions and executed state changes. Controlled sharing requires approval valid at execution time; matched safe cases help expose over-refusal. | A refusal classifier or trajectory loss alone does not address stale authorization. Trusted runtime checks and utility-matched controls are mandatory baselines. |
| [FATE: On-Policy Self-Evolution via Failure Trajectories](https://arxiv.org/html/2605.11882v1), §§3.2–3.3, Appendix K | Repair the failed interaction, execute the repair, and verify what actually changed. | Uses executable environment resets and repair verification, with filters for utility, refusal behavior, and trajectory validity. | Failure-trajectory repair and execution-grounded filtering already exist. A new preference objective needs a residual that these mechanisms miss. |
| [AgentSpec](https://arxiv.org/abs/2503.18666), primary abstract-level baseline check | Put enforceable rules between the model's proposed action and the environment. | Runtime enforcement is a direct competing approach to behavioral safety training. | For explicit, observable authorization rules, implementing enforcement can resolve the issue without developing a new alignment algorithm. |

**Assessment.** The remaining hard cases may involve ambiguous policy semantics or unavailable state. But “the agent has partial information” is not enough to establish a new research contribution: conservative reauthorization and belief-state control are already strong approaches. This screen did not establish a benchmark failure surviving them at matched legitimate-task utility.

### 3.4 Other consequential failures that do not yet yield a new method

| Paper and reading scope | Intuition | Evidence and existing response | Decision |
|---|---|---|---|
| [The Reward Model Selection Crisis in Personalized Alignment](https://arxiv.org/abs/2512.23067), v2; abstract and relevant experimental PDF passages | Ranking stored examples correctly is different from steering the answers a generator can actually produce. | Reports weak association between reward-model accuracy and reward-guided generation; simple ICL/RAG is strong and can outperform reward-guided methods. | Reject “improve personalized reward accuracy” without demonstrating user-facing gains against ICL/RAG under the same user information. |
| [Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning](https://arxiv.org/html/2604.22074v1), §§3–7.1, Appendix H | Correct answers do not force a model to write reasoning that it uses or that another model can reconstruct. | CIR declines on 19/40 tasks and SR on 17/40. Expert-trace SFT and auxiliary CIR/SR rewards improve the measured properties; SR-minus checks question-paraphrase leakage. | Reject generic faithfulness rewards. Improved metrics are not a general certificate of faithful computation, but that distinction alone is not a new paper. |
| [Why Multi-Step Tool-Use RL Collapses](https://arxiv.org/html/2606.26027v1), §§4.2–5.4 and limitations | The model can retain useful task knowledge while losing the syntax needed to invoke its tools. | Corrective supervision helps; format/content shifts remain difficult. A grammar-constrained decoding comparison is missing from the examined evidence. | Do not infer a new alignment mechanism until constrained decoding and ordinary corrective SFT are compared. |
| [Objective-Level Hacking](https://arxiv.org/html/2602.01103v1), §§4.3–4.4, §§6–7 | A token objective can amplify a mismatch between training and inference in MoE policies. | The paper's sequence-level GSPO baseline avoids the demonstrated abnormal discrepancy growth. | Reject an incremental clipping or token-weighting repair without a residual under sequence-level correction. |

## 4. Deeper mechanism analysis and mathematical limits

These are explanatory calculations, using simplified assumptions stated explicitly. They help identify what a proposed theory would need to explain. They do not validate a research direction by constructing a failure.

### 4.1 Why domain routing does not guarantee capability preservation

Let a student have parameters $\theta$, teacher-specific training loss $L_i$, and held-out capability loss $L_j$. One gradient step on teacher $i$ gives, by Taylor expansion,

$$
L_j(\theta-\eta\nabla L_i(\theta))-L_j(\theta)
=-\eta\langle\nabla L_j,\nabla L_i\rangle+O(\eta^2).
$$

The expansion assumes twice differentiable losses and a locally bounded Hessian. Selecting prompts from domain $i$ does not force its gradient to be orthogonal to the gradients of other capabilities. Negative inner products can therefore increase another capability's loss.

For a mixture of teacher updates, simultaneous first-order preservation would require

$$
\left\langle\nabla L_j,\sum_i w_i\nabla L_i\right\rangle\geq 0
\quad\text{for every protected capability }j,
\qquad w_i\geq 0.
$$

This identifies two different issues: an optimizer can choose a poor update even when a useful common direction exists; alternatively, local gradients may leave no nonzero direction that improves every objective. The first suggests optimization and preservation baselines. The second is a local tradeoff, not proof of a global capacity limit.

**What would make theory substantive:** relate observable training quantities to eventual capability loss under an explicitly restricted model class, and explain a residual that current preservation methods exhibit. The inner-product expansion itself is standard and supplies neither novelty nor a method advantage.

### 4.2 Why increasing self-reward does not certify increasing accuracy

Consider an idealized categorical policy $p_t(a)$ with a fixed, detached reward $r(a)$. Under continuous-time KL mirror ascent, its replicator dynamics are

$$
\dot p_t(a)=p_t(a)\bigl(r(a)-\mathbb E_{p_t}[r]\bigr).
$$

Let $c(a)\in\{0,1\}$ indicate correctness, unavailable to the training algorithm. Then

$$
\frac{d}{dt}\mathbb E_{p_t}[c]
=\operatorname{Cov}_{p_t}(c,r),
\qquad
\frac{d}{dt}\mathbb E_{p_t}[r]
=\operatorname{Var}_{p_t}(r)\geq 0.
$$

These identities follow by substituting the policy dynamics into each expectation. Reward can rise while accuracy falls whenever the covariance is negative. They do not describe all neural RLVR updates: shared parameters, moving rewards, finite samples, and KL anchoring change the dynamics. In particular, a moving reward adds its own time-derivative term.

**Research implication:** measuring confidence more precisely does not establish the sign of its covariance with correctness. A useful proposal must specify where corrective information comes from and why it remains reliable as the policy changes. “Add a verifier” and “periodically recalibrate a critic” already have direct precedents in this screen.

**Theory pitfall:** a guarantee that assumes this covariance is always positive assumes away the central failure. Conversely, an impossibility statement over arbitrary hidden correctness functions is a generic identifiability result, not evidence of a useful new algorithm.

### 4.3 When agent authorization is an enforcement problem

Let $s_t$ be trusted current environment state and $A(s_t,a)$ an executable predicate indicating whether action $a$ is authorized. A runtime mechanism can restrict execution to

$$
\mathcal A_{\mathrm{safe}}(s_t)=\{a:A(s_t,a)=1\}.
$$

If the predicate is correct, reads current state atomically with execution, and mediates every relevant action, the restricted policy executes no action violating that predicate. This follows directly from membership in the permitted set. It does not guarantee all forms of safety or eliminate harms outside the predicate.

If only history $h_t$ is available, a belief-state formulation instead permits actions satisfying

$$
\Pr\bigl(A(s_t,a)=0\mid h_t\bigr)\leq\delta.
$$

Now belief calibration, unavailable evidence, and the option to obtain fresh authorization matter. Those are substantive issues, but this is standard constrained decision-making structure. A new training loss does not solve them merely by rephrasing the constraint.

**Necessary empirical distinction:** compare full-state enforcement, enforcement with agent-visible information, and conservative fresh-authorization requests. A residual that disappears with a deployable check is an integration issue for that setting; a residual that persists must be measured alongside legitimate-task completion.

### 4.4 Why reward ranking accuracy can miss generation quality

For user $u$, let $\mu_u$ be the distribution of response pairs used to evaluate a reward model. Pairwise accuracy measures a property such as

$$
\Pr_{(y^+,y^-)\sim\mu_u}
\bigl[\widehat r_u(y^+)>\widehat r_u(y^-)\bigr].
$$

Deployment instead evaluates

$$
V_u(\widehat r_u)=
\mathbb E_{y\sim\pi_{\widehat r_u}(\cdot\mid x,u)}[r_u^*(x,y)],
$$

where $\pi_{\widehat r_u}$ is the actual decoding or policy-adaptation procedure. The first quantity alone places no distribution-free bound on the second: it need not constrain behavior on responses reached during generation, and it omits how the decoder uses reward magnitudes.

This explains why an improved classifier can fail to improve user-facing behavior. It does not establish a new solution. A proposal must beat prompting/retrieval given the same user evidence, and cannot manufacture an advantage by withholding preference history from those baselines.

## 5. Remaining questions and the evidence each owes

These are **conditional investigation questions**, not endorsed new paper proposals. No method is named or invented before establishing its necessity.

| Question | Why it matters | Decisive next check | What would end the direction? |
|---|---|---|---|
| Does self-training still lose competence under a realistic shift after strong external grounding? | A scalable learning pipeline must retain a meaningful connection between its reward and task success. | Reproduce extended training with a current calibrated-critic method and executable verification where available; distinguish source-label, target-label, and verifier access. Measure held-out accuracy, reward calibration, diversity, and peak-to-final regression. | Existing calibration or verification sustains competence under the actual information regime; or the proposed regime simply removes all information that could identify correctness. |
| Does multi-teacher preservation leave a repeatable capability loss that is not explained by curriculum, optimizer state, or available anchors? | Skill integration should preserve useful abilities rather than merely improve a mean score. | Start from a released conflicting teacher pair. Compare interleaving, reset versus retained optimizer state, recovery, and a current preservation method. Report each capability and its tradeoff frontier. | Existing recovery/preservation fixes the loss; the apparent residual is only a curriculum artifact; or a new method merely selects a different point on the same frontier. |
| Do long-horizon unsafe actions persist under a deployable authorization monitor at comparable legitimate utility? | Harm is determined by what tools execute, including after stale approvals. | Evaluate matched safe/unsafe workflows with execution-time checks and fresh-authorization baselines. Separate missing state, ambiguous policy, incomplete mediation, and model noncompliance. | An ordinary monitor or fresh check resolves the demonstrated mechanism; blanket refusal creates the apparent safety gain; or the “hard case” requires withholding state available in deployment. |

The first question has the strongest documented training failure in this sweep. It still has a large novelty debt. The second has a dense recent solution landscape. The third has high practical importance but a strong possibility of being solved by enforcement engineering in the measured setting.

### Experiment requirements if a question passes its next check

Before deriving a new objective, the evidence should include:

- The original method and a strongest relevant repair, with comparable data, model capacity, information access, and tuning effort.
- A natural failure regime anchored to an actual deployment or source setup, not a synthetic construction designed to defeat one loss.
- At least one repeatability check across seeds or independently sampled cases, with uncertainty on the consequential outcome.
- A mechanism intervention: change the hypothesized cause while keeping competing explanations controlled.
- Ablations for information access, optimizer/curriculum changes, and the strongest simple repair.

The intended result is a decision. Success means a material failure persists under those controls; it does not mean a preferred new algorithm wins. If it does persist, theory should target that measured mechanism. If it disappears, close the candidate.

## 6. Self-review

**Does this identify real problems?** Yes. The primary evidence includes extended-training degradation, cross-domain capability interference, and state-dependent unsafe actions. These have direct quality or safety consequences.

**Does it show that their practical solutions are trivial?** Only in specific settings. Explicit authorization predicates can be handled by enforcement; one examined negative-update collapse has a simple damping repair. Multi-teacher preservation and externally grounded self-training can require substantial engineering. Existing work on them limits our novelty claim without proving universal resolution.

**Does it justify a new theoretical paper now?** No. The elementary calculations above explain why certain failures are possible. They neither establish an unaddressed mechanism nor prove an advantage over existing successful methods. Missing convergence guarantees in a limitations section are insufficient motivation by themselves.

**Did this produce initial experimental support?** No new empirical support. Published experiments substantiate the problem areas; our own residual-failure tests remain unperformed. A toy example would not fill this gap. This workspace has no usable pretrained model weights or GPU for the required training comparisons, so reporting benchmark results here would be unsupported.

**Recommendation.** Retire the rejected routing proposal and do not replace it with another weighting, damping, projection, or verifier-calibration variant on the present evidence. Preserve the three conditional questions above with their explicit evidence debts. The next defensible proposal should begin with a failure that remains after the relevant current repair is applied, or with a clearly new theoretical characterization of a practically observed unresolved regime.

## 7. Deliverable status

**15 October continuation:** the [success-transfer brief](../../studies/rlvr_success_transfer/research_report.md) records the subsequent candidate screen, an RLVR research question, current prior-art checks, analytical controls and explicit remaining experiment gates. It does not turn the conditional questions here into endorsed methods.

| Item | Status |
|---|---|
| Broader literature screen centered on consequential failures | Completed within the stated targeted scope |
| Paper-level intuition and specific method/appendix anchors | Included above; abstract-only checks explicitly labeled |
| Mathematical explanation of four mechanisms | Included, with assumptions and limits |
| Novel method established | No |
| New model-training experiments | Not run |
| Paper-ready proposal or theoretical guarantee | Not established |
| New OpenReview discussion coverage | Not obtained in this pass |

This result replaces optimism unsupported by evidence with a concrete account of what is known, what is already addressed, and what must be demonstrated before committing to another research direction.
