# DPO literature analysis

## Documents

1. [Literature analysis](01_expanded_literature_analysis.md): 26 distinct papers, intuition, contributions, formulas and limitations.
2. [Research programs](02_theoretical_research_proposals.md): earlier alternative programs and their theoretical/novelty constraints.
3. [OpenReview discussion audit](03_openreview_discussion_audit.md): reviews, rebuttals and metareviews; ProtAlign coverage remains incomplete.
4. [Source-grounded problem screen](04_source_grounded_problem_screen.md): exact method/limitation anchors, existing remedies and the rejected generator-routing proposal.
5. [Renewed failure-grounded analysis](05_failure_grounded_research_screen.md): broader 2026 literature, intuition, failure mechanisms, current repairs, mathematical explanations, and explicit evidence still required for a new direction.

**Latest GPU experiment:** [learning from successful RLVR rollouts](../../studies/rlvr_success_transfer/README.md). Five full-parameter Qwen2.5-0.5B pilot arms completed on the RTX 5060 Ti 16 GB. Ordinary SFT learned the sampled cases; the published persistent-failure cohort remains unreproduced. The report includes uncertainty, formatting/truncation checks, gradient forecasts and a policy-KL diagnostic. These results do not justify endorsing a new method.

**Earlier theoretical result:** [policy capacity and preference feedback](../../studies/policy_capacity_feedback/README.md). This study derives a continuous-class statistical characterization and reproducible finite-model checks. Its focused audit finds substantial prior coverage and a trivial prompt-feature repair of the illustrative contextual failure. The theorem is retained, but a full-project recommendation is withheld pending a stronger novelty or practical-necessity case.

The [preceding research screen](05_failure_grounded_research_screen.md) documents the rejected alternatives. The [informative-delay study](../../studies/informative_delays/README.md) retains a scoped theoretical finding; that direction is deprioritized because practical repairs exist and a substantial remaining gap is unestablished. The ranking–rating fusion and verifier-audit studies also failed their baseline gates.

## Sources and verification

[Source-paper inventory](../../resources/README.md), [PDF manifest](../../resources/download_manifest.json), and [spreadsheet input inventory](evidence/spreadsheet_paper_inventory.json).

[Analytical checks](verification/analytic_checks.py) produce [one canonical result](verification/analytic_checks_results.json) and the phase figure used in the research-program report.

```bash
python3 reports/dpo_expanded/verify_corpus.py
python3 reports/dpo_expanded/verification/analytic_checks.py
python3 scripts/validate_reports.py
```

The shared validator checks Markdown math and local links, retaining only [the validation result](../validation_results.json). Original papers are preserved as PDFs; temporary text extracts, build products and earlier report versions are removed.
