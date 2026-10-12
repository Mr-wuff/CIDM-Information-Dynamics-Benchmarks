# Experiment naming conventions

Use scientific intent rather than development chronology in file names. Recommended naming pattern:

`<dataset-or-system>_<experimental-variable-or-method>_<evaluation-purpose>.ipynb`

Examples:

- `lorenz96_port_hamiltonian_chaotic_dynamics.ipynb`
- `era5_swh_open_ocean_forecasting.ipynb`
- `era5_swh_spatiotemporal_spectral_regularization.ipynb`
- `era5_swh_coupled_oscillator_dynamics.ipynb`
- `era5_swh_wind_forcing_probabilistic_forecasting.ipynb`

Use `study_id` fields in the machine-readable inventory for stable tracking. Preserve a private source-to-release mapping for auditing; scientific methods and results must not be relabeled as different experiments. Do not imply that newly added validation code originated in the paper's benchmark pipeline.