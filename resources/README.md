# Spreadsheet paper resources

Verified on 2026-10-14: all 27 listed entries are present (26 distinct papers; AW-DPO is listed twice).

Sixteen arXiv PDFs were downloaded automatically through the connected Chrome browser. Eleven existing PDFs were reused. The JSON manifest records source rows, versions, page counts, sizes, and SHA-256 hashes. This verifier checks local files; it does not require OpenReview API access.

SafeDPO source discrepancy: the spreadsheet links the 2025 submission inside its ICLR 2026 tab. The local manuscript is preserved; the analysis distinguishes submission rounds.

| Sheet / row | Paper | Local PDF |
|---|---|---|
| ICLR 2026 / 2 | Property-Driven Protein Inverse Folding with Multi-Objective Preference Alignment | [4249_Property_Driven_Protein_I.pdf](4249_Property_Driven_Protein_I.pdf) |
| ICLR 2026 / 3 | Why DPO is a Misspecified Estimator and How to Fix It | [24413_Why_DPO_is_a_Misspecifie.pdf](24413_Why_DPO_is_a_Misspecifie.pdf) |
| ICLR 2026 / 4 | Token-Importance Guided Direct Preference Optimization | [6576_Token_Importance_Guided_D.pdf](6576_Token_Importance_Guided_D.pdf) |
| ICLR 2026 / 5 | Mitigating Mismatch within Reference-based Preference Optimization | [362_Mitigating_Mismatch_within.pdf](362_Mitigating_Mismatch_within.pdf) |
| ICLR 2026 / 6 | DuPO: Enabling Reliable Self-Verification via Dual Preference Optimization | [6741_DuPO_Enabling_Reliable_Se.pdf](6741_DuPO_Enabling_Reliable_Se.pdf) |
| ICLR 2026 / 7 | Importance Sampling for Multi-Negative Multimodal Direct Preference Optimization | [24535_Importance_Sampling_for_.pdf](24535_Importance_Sampling_for_.pdf) |
| ICLR 2026 / 8 | SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety | [5041_SafeDPO_A_Simple_Approach.pdf](5041_SafeDPO_A_Simple_Approach.pdf) |
| ICLR 2026 / 9 | Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs | [11934_Uni_DPO_A_Unified_Paradi.pdf](11934_Uni_DPO_A_Unified_Paradi.pdf) |
| ICLR 2026 / 10 | P^2 -DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization | [8851_P_2_DPO_Grounding_Halluci.pdf](8851_P_2_DPO_Grounding_Halluci.pdf) |
| ICLR 2026 / 11 | Alignment-Weighted DPO: A principled reasoning approach to improve safety alignment | [15113_Alignment_Weighted_DPO_A.pdf](15113_Alignment_Weighted_DPO_A.pdf) |
| ICLR 2026 / 12 | Is On-Policy Data always the Best Choice for Direct Preference Optimization-Based LM Alignment? | [24230_Is_On_Policy_Data_always.pdf](24230_Is_On_Policy_Data_always.pdf) |
| ARXIV 2026 / 2 | Coverage Improvement and Fast Convergence of On-policy Preference Learning | [2601.08421v1.pdf](2601.08421v1.pdf) |
| ARXIV 2026 / 3 | Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning | [2601.03027v3.pdf](2601.03027v3.pdf) |
| ARXIV 2026 / 4 | MixDPO: Modeling Preference Strength for Pluralistic Alignment | [2601.06180v1.pdf](2601.06180v1.pdf) |
| ARXIV 2026 / 5 | From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models | [2601.06108v1.pdf](2601.06108v1.pdf) |
| ARXIV 2026 / 6 | An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift | [2601.05882v2.pdf](2601.05882v2.pdf) |
| ARXIV 2026 / 7 | SPINAL - Scaling-law and Preference Integration in Neural Alignment Layers | [2601.06238v1.pdf](2601.06238v1.pdf) |
| ARXIV 2026 / 8 | Direct Preference Optimization with Rating Information: Practical Algorithms and Provable Gains | [2602.00603v1.pdf](2602.00603v1.pdf) |
| ARXIV 2026 / 9 | Tangent Space Fine-Tuning for Directional Preference Alignment in Large Language Models | [2602.01128v1.pdf](2602.01128v1.pdf) |
| ARXIV 2026 / 10 | Alignment-Weighted DPO: A principled reasoning approach to improve safety alignment | [2602.21346v1.pdf](2602.21346v1.pdf) |
| ARXIV 2026 / 11 | YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation | [2601.08441v1.pdf](2601.08441v1.pdf) |
| ARXIV 2026 / 12 | SP^2DPO: An LLM-assisted Semantic Per-Pair DPO Generalization | [2601.22385v1.pdf](2601.22385v1.pdf) |
| ARXIV 2026 / 13 | ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection | [2601.09195v3.pdf](2601.09195v3.pdf) |
| ARXIV 2026 / 14 | CATTO: Balancing Preferences and Confidence in Language Models | [2601.23096v2.pdf](2601.23096v2.pdf) |
| ARXIV 2026 / 15 | ECLIPTICA - A Framework for Switchable LLM Alignment via CITA - Contrastive Instruction-Tuned Alignment | [2601.06157v1.pdf](2601.06157v1.pdf) |
| ARXIV 2026 / 16 | Clipping-Free Policy Optimization for Large Language Models | [2601.22801v1.pdf](2601.22801v1.pdf) |
| ARXIV 2026 / 17 | Towards a Theoretical Understanding to the Generalization of RLHF | [2601.16403v1.pdf](2601.16403v1.pdf) |

Verification: `python3 reports/dpo_expanded/verify_corpus.py` from the workspace.

Research outputs: [expanded report directory](../reports/dpo_expanded/).

## Additional experiment source data

[Released RLVR reward-statistics CSV](rlvr_success_transfer/prompt_reward_stats.csv), downloaded directly on 15 October 2026 from the [authors' repository](https://github.com/yulinchen99/unlearnability-rlvr/tree/main/example_data). SHA-256: `af1033f772ba3e353856d9d65ade0e4fc453443952d87dde078f3f083f0ce4ad`. This is auxiliary source data, not another PDF or our own experimental result. Its prompt identifiers have not been reconciled with our separately sampled MATH pilot; no original-cohort reproduction is inferred from it.
