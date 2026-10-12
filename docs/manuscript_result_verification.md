# Manuscript result verification

The uploaded experimental source packages were checked against the manuscript. Canonical studies and independently completed experiment records:

| Study | Records | Correct source |
|---|---:|---|
| Controlled model comparison | 180 | Publication-convergence run; 3 PDE tasks, 6 model families, 10 seeds |
| Extended multiphysics benchmark | 120 | Eight-task base run plus corrected Diffusion–Sorption evaluation (105 unchanged + 15 replaced) |
| Structural component ablation | 110 | Main single-component ablation run |
| Long-horizon and input-noise study | 48 | Four PDE tasks, four models, three seeds |
| Latent-state consistency | 18 | Three tasks, two CIDM models, three seeds |

A preliminary three-task comparison with only 90 runs, the superseded uncorrected Diffusion–Sorption results, mechanism-rescue interaction trials, and scaling/saturation experiments are excluded because they are not part of the reported manuscript experiments. This file records the selection and should be accompanied by the corresponding canonical results and notebook sources.

The reported 18 state-consistency runs each contain multiple rollout evaluation modes, which should not be counted as independent training runs.

All measurements are historical archived results. Selection and naming standardization do not constitute new training or independent replication.
