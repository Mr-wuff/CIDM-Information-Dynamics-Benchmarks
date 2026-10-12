# Latent-state consistency and causal self-reencoding

This section reproduces the manuscript experiment involving 3 PDE tasks, 2 CIDM variants, and 3 random seeds (18 trained model instances). Four causal rollout modes—pure latent, reencode every 12 steps, every 6 steps, and every step—were evaluated using the same weights.

The full run-level table contains 18 rows. Additional CSV tables contain horizon-specific and mode-specific measurements; these are not independent additional training runs. The notebook stores the original experiment code without training outputs.

The 48-step manuscript RMSE data should be interpreted as the fixed setup's results, not evidence that any synchronization frequency is universally optimal.
