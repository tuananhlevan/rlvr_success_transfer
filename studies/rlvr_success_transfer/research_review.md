# Research investment review: learning from successful RLVR rollouts

Date: 16 October 2026. Scope: review of the current research question, local evidence, selected primary literature and the value of additional experiments. This is an advisory review; it does not stop the remote Colab process or alter the training protocol.

## Decision

**The current evidence does not justify committing to a multi-day methods/theory paper campaign.** The question merits a bounded feasibility investigation. Completing the 120-step reproduction may establish a baseline phenomenon, but it does not by itself supply an original research contribution.

The earlier [research report](research_report.md#research-decision-after-the-pilot) already withheld a new-method recommendation after the pilot. I should have kept that scientific gate explicit while helping make the reproduction executable. Time spent on setup, migration and launcher repairs provides no additional evidence for the research hypothesis.

## What is being evaluated

The strongest version of the question is: when a reasoning model already generates correct solutions, which measurable property of the learning signal prevents improvement on independently held-out problems, and can an intervention repair that failure beyond strong existing methods?

This could matter for choosing whether to spend compute on additional rollouts, a different update, or richer supervision. Its value depends on a consequential residual failure and an actionable explanation. Neither has been established locally.

## Evidence assessment

| Criterion | Evidence | Assessment |
| --- | --- | --- |
| A real motivating problem | Chen et al. report weak learning despite successful rollouts; a September reanalysis retains a slow-learning effect while challenging cohort construction and gradient evidence. | Worth investigating, with narrower claims than an intrinsic representation barrier. |
| A locally demonstrated hard failure | The pilot used eight mixed-reward prompts, a shorter response cap and a different recipe; it was not the published persistent-failure cohort. | Missing. The pilot cannot establish or refute the published effect. |
| Failure survives ordinary repairs | Student-success SFT improved selected-prompt success from 43.75% to 81.25%. | Contrary evidence for this pilot's need for a novel repair. Unequal training budgets preclude a controlled algorithm ranking. |
| Meaningful transfer | Student-success SFT changed success on 16 held-out MATH problems from 20.31% to 24.22%; the descriptive prompt-bootstrap interval for the change is approximately -5.5 to +14.8 points. | Inconclusive. Same-problem prefix changes are weaker evidence than generalization to new problem families. |
| A distinctive intervention | No new repair has been validated. GAR already rewards gradient alignment with expert traces; SRL already supplies expert-action supervision. | Significant competition; a gradient score or expert-guided variant needs a specific distinction and matched comparisons. |
| A new theoretical result | Score identities, Fisher geometry, the local Taylor argument and tangent optimization are foundations. The report explicitly labels stronger finite-sample/finite-update results as targets. | No established theory contribution. |
| A practical diagnostic | Some pilot linear forecasts have the wrong sign or imply gains outside the probability range when extrapolated to the endpoint. | Uncalibrated. This does not disprove a conditional local theorem, but the forecasts cannot currently guide expensive interventions. |

Local numbers come from [the canonical pilot result](gpu_pilot_results.json), not a new training run. This is one training seed, tiny evaluation panels, different supervision/token budgets and exploratory controls. Confidence intervals are descriptive and do not quantify training-seed variability.

## Current literature changes the value of the experiment

1. **The phenomenon is already published.** Repeating weak learning despite occasional correct answers is occupied by [Chen, He and Zhao, ICML 2026](https://proceedings.mlr.press/v306/chen26aq.html). A reproduction needs a further scientific purpose.
2. **Measurement corrections are also occupied.** [Chakma et al., September 2026 preprint](https://arxiv.org/html/2609.40115), particularly sections 3.2–3.6, retain slow learning but identify difficulty-label instability and sample-count effects on gradient comparisons. Repeating those corrections is insufficient novelty. Their remaining mechanism question is legitimate; it is not evidence that our proposed answer is correct.
3. **Related repairs are concrete competitors.** [GAR, September 2026 preprint](https://arxiv.org/html/2609.03342v1), sections 2–3, uses expert-anchor gradient rewards. [SRL, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7ab7fc1278add78fd6eae2da7a14c79b-Abstract-Conference.html), supplies expert-action supervision. These sources establish overlap and relevant baselines; they do not demonstrate that either solves the exact proposed fixed cohort. The SRL comparison here is grounded in its official abstract, not a full new method audit.

This is a targeted update of the existing prior-art screen, not an exhaustive proof of novelty or a new OpenReview review/rebuttal audit. Preprint claims remain author-reported findings.

## What the ongoing run can establish

The driver trains a baseline, performs periodic validation, and ultimately evaluates operational slow-learning candidates. It does not currently run matched repair forks, test a new intervention, evaluate a predeclared held-out family mechanism, or validate the proposed gradient diagnostic.

The user reports that Colab has reached step-3 sampling. The archived repository results record the earlier step-1 export; no new Colab validation curve or final result was available for this review. Successful execution and progress into later steps do not establish the scientific premise.

A short run cannot certify persistent failure. Conversely, spending enough compute to establish persistent failure still leaves the intervention, causal explanation, baseline comparisons and theory to be demonstrated. This distinction makes the full reproduction a costly entry condition for the project rather than a decisive test of the proposed contribution.

## Why the current cost is difficult to justify

The last archived sampling batch generated 7,093,028 response tokens across 8,192 responses. A full campaign additionally needs repeated optimization, independent evaluations, multiple seeds, repair baselines and transfer experiments. Current A100 throughput has not been measured well enough to forecast that campaign's cost. The old 5060 Ti timing is not an A100 completion estimate.

The critical issue is the value of the next experiment: completing a baseline alone cannot distinguish our prospective mechanism from existing optimizer, supervision and measurement explanations. Neither an interesting problem nor a functioning implementation supplies that missing discrimination.

## Minimum evidence before expanding compute

1. **Use existing compatible states first.** Retain the completed local checkpoint and raw responses. Check whether released assets can support the required comparison before reproducing additional full training histories. Audit task IDs and provenance; do not treat the pilot's selected prompts as the published hard cohort.
2. **Specify a small falsification experiment.** Freeze the proposed claim, prompt panel, independent evaluation sample, strongest feasible standard repair, primary transfer outcome and an additional GPU-hour cap before observing intervention outcomes. A cap limits investment; it does not make a short horizon a persistent-failure test.
3. **Test ordinary repairs before designing a new one.** At the same starting checkpoint compare current RL training with student-success SFT and the strongest relevant existing alternative. Record generated/trained tokens, expert-information access, actual policy change and wall time. Add GAR/SRL comparisons when the proposed claim uses their information or mechanism.
4. **Demand independent transfer.** Improvements must extend to verified, separately constructed problem variants/families, with broad held-out performance checked. Fitting trained wording or changing answer formatting is insufficient for the intended mechanism claim.
5. **Test the mechanism out of sample.** A diagnostic must predict the sign or ranking of realized transfer gains on interventions and examples not used to fit it. A low gradient cosine or a same-sample Fisher fit cannot certify a representation barrier.
6. **Expand only after positive evidence.** A consequential residual surviving strong standard repairs, plus a plausible differentiated intervention or falsifiable mechanism prediction, would justify a larger replication. Only then scope substantive theory and cross-model/seed experiments. An exact contribution must still survive a refreshed prior-art check.

Stop the methods direction if ordinary repairs close the relevant gap economically, if apparent gains vanish on independent transfer evaluation, or if the proposed contribution reduces to established measurement or optimization facts. A narrowly framed mechanism study could still be valuable, but its causal result must be demonstrated rather than assumed.

## Recommendation for the running job

Preserve its latest verified checkpoint and generation chunks. **I recommend pausing at a safe saved boundary and reviewing a bounded feasibility test before authorizing additional multi-session runs.** Avoid discarding a nearly completed optimizer update solely for this review. I have not issued any command to stop the remote process.

For the intended top-conference methods/theory paper, the current answer is **not enough evidence to justify the full compute commitment**. The retained possibility is a finite-budget search for a residual failure with a distinctive, useful explanation. A strong result there could change this decision; engineering progress alone cannot.

## Review limits

This is an AI-assisted, inline review by the same assistant involved in the study; it is not an independent panel decision. It draws on canonical local pilot/results/code and a fresh check of the four primary sources linked above. No new GPU experiment, live Colab inspection, acceptance-probability estimate or global novelty certification is claimed.
