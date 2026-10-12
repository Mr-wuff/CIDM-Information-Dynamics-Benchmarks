# Manuscript evidence synchronization audit

All five manuscript-aligned experiment families are now present on the repository's main branch, including canonical source notebooks and per-run CSVs.

| Study | Canonical complete records | Record table |
|---|---:|---|
| Controlled architecture comparison | 180 | `experiments/pde_benchmarks/controlled_comparison/results/seedwise_predictions.csv` |
| Corrected eight-task multiphysics benchmark | 120 | `experiments/pde_benchmarks/multiphysics_benchmark/results/seedwise_predictions_publication.csv` |
| Structural component ablation | 110 | `experiments/pde_benchmarks/component_ablation/results/seedwise_ablation.csv` |
| Long-horizon forecasting and input-noise diagnostics | 48 | `experiments/pde_benchmarks/long_horizon_evaluation/results/seedwise_training_and_evaluation.csv` |
| Latent-state consistency and causal self-reencoding | 18 | `experiments/pde_benchmarks/state_consistency/results/seedwise_training_and_evaluation.csv` |

The complete public-facing inventory is **476 completed independent experimental records**, not 476 additional model families. The 18 state-consistency training instances each have repeated horizon/mode evaluations. Superseded diffusion–sorption records, unsuccessful preliminary runs, mechanism interaction/rescue studies, and scaling studies not reported in the manuscript are not part of this release.

Original upstream PDEBench input arrays and trained model checkpoints are not redistributed. Notebooks have their archived execution output removed and may require environmental configuration. Run `python scripts/verify_paper_results.py` to validate all five archived record counts and uniqueness constraints. The check does not independently rerun model training.

The accompanying local release archive contains SHA-256 checksums and source selection provenance.
