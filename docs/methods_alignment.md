# Architecture and manuscript alignment

The reference PyTorch architecture defines a latent-field model with spatial, history, and phase/spectral encoders, a structured latent update composed of skew-symmetric channel mixing, antisymmetric spatial flux and restricted residual branches. An optional variant adds a spectral operator and a bounded external-reservoir correction.

The current `HiddenReservoirTransfer` is a gated residual mapping, not a separately time-evolved physical storage state. The spectral operator is a Fourier-domain latent residual. The boundary reservoir is a virtual correction field and should not be described as a rigorously conserved physical-energy store. An antisymmetric generator implies a continuous-time quadratic norm identity, not an exact norm-preserving discrete Euler step.

Labels in the manuscript may use different expansion(s) for SOC and BER; prefer the definitions in the archived implementation when describing executable behavior. This release does not claim new numerical verification of the physical interpretation.