# CIDM: Information Dynamics Benchmarks

Reproducibility materials for the **Closed Information Dynamics Model (CIDM)** study, covering the benchmark and diagnostic experiments explicitly reported in the associated manuscript.

## Manuscript experiments

| Study | Experimental protocol | Source |
| --- | --- | --- |
| Controlled architecture comparison | 3 PDE tasks × 6 model families × 10 seeds; 4,000 updates | `experiments/pde_benchmarks/controlled_comparison/` |
| Multiphysics generalization | 8 PDE tasks × 5 model families × 3 seeds; 2,500 updates | `experiments/pde_benchmarks/multiphysics_benchmark/` |
| Component ablation | Skew, flux, hidden, spectral encoder, spectral operator and boundary reservoir over 5 tasks | `experiments/pde_benchmarks/component_ablation/` |
| Long-horizon evaluation | 6/12/24/48-step forecasts over 4 tasks × 4 models × 3 seeds; 24-step input-noise diagnostics | `experiments/pde_benchmarks/long_horizon_evaluation/` |
| Latent-state consistency | Pure latent and 1/6/12-step causal self-reencoding over 3 tasks × 2 CIDM variants × 3 seeds | `experiments/pde_benchmarks/state_consistency/` |

Results contain original per-seed measurements, aggregate tables, and manuscript consistency audits. Notebook outputs were stripped without modifying the computational cells. Public file names describe the scientific task, not the internal sequence of experimental revisions.

**Data contract correction.** The initial multiphysics experiment used an invalid diffusion–sorption dataset contract. Its reported rows were replaced exclusively by the validated follow-up experiment. The canonical `seedwise_predictions_publication.csv` comprises 105 original valid rows plus 15 corrected diffusion–sorption rows. The initial invalid rows are not included in the published result tables.

Original PDEBench training data are **not redistributed** here. See `data/README.md` for the upstream dataset and applicable license. Model checkpoints can be supplied as separate versioned release artifacts where license and storage limits permit; no unverified checkpoint is represented as a final published model.

Training notebooks retain some original experiment-specific variable names and cache labels because rewriting them would alter code paths and compromise exact historical reproducibility. These internal implementation identifiers are not part of the publication naming scheme.

## Reproducibility checks

```bash
python scripts/verify_paper_results.py
pytest -q
```

The benchmark suite needs PyTorch, the upstream PDEBench data and compatible optional architecture dependencies. A successful architecture smoke test does not by itself certify bitwise reproduction of every GPU run.

See `docs/experiment_inventory.md` for exclusions, `docs/reproducibility_status.md` for scope and `docs/security_and_data_policy.md` for publishing safeguards.