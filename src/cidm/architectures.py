"""
CIDM architecture archive
=========================

Reference PyTorch implementation for the finalized model family:

1. CIDM / Closed Information Dynamics Model
   - General-purpose closed information dynamics backbone.

2. CIDM-BER / CIDM with Boundary Energy Reservoir
   - Physics-closure enhanced variant.
   - Adds a latent Spectral Operator Core and a Boundary Energy Reservoir (BER)
     residual closure layer.

This file is intended for architecture archiving and future experiment reuse. It
contains only model definitions and lightweight diagnostics; training/evaluation
loops are intentionally kept outside this file.

Expected input format
---------------------
`history`: Tensor[B, L, C, H, W]
`port`: optional Tensor[B, C_port, H, W] or Tensor[B, horizon, C_port, H, W]
`target`: optional Tensor[B, horizon, C, H, W] for diagnostics/loss construction

Output dictionary
-----------------
`pred`: Tensor[B, horizon, C, H, W]
`latent`: final latent state
`latent_seq`: Tensor[B, horizon, D, H, W]
`diagnostics`: scalar tensors for energy drift, SOC strength, BER strength, etc.

Authoring note
--------------
This is a cleaned reference architecture. Exact Colab benchmark implementations
may contain additional data-generation, plotting, logging, and ablation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CIDMConfig:
    channels: int = 1
    hist_len: int = 4
    horizon: int = 6
    latent_dim: int = 32
    hidden: int = 64
    port_channels: int = 1
    use_phase_encoder: bool = True
    use_time_encoder: bool = True
    use_hidden_reservoir: bool = True
    use_flux: bool = True
    use_reverse_diagnostics: bool = True
    # CIDM-BER options
    soc_modes1: int = 8
    soc_modes2: int = 8
    ber_add_scale: float = 0.05
    ber_enabled: bool = True
    soc_enabled: bool = True


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv_block(in_ch: int, out_ch: int, hidden: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, padding=1),
        nn.GroupNorm(groups, hidden),
        nn.SiLU(),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.GroupNorm(groups, hidden),
        nn.SiLU(),
        nn.Conv2d(hidden, out_ch, 1),
    )


class SpaceEncoder(nn.Module):
    """Encode the latest observed spatial field into latent information state."""

    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, latent_dim, 3, padding=1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeHistoryEncoder(nn.Module):
    """Encode a finite observation history as temporal context."""

    def __init__(self, channels: int, hist_len: int, latent_dim: int):
        super().__init__()
        self.hist_len = hist_len
        self.net = nn.Sequential(
            nn.Conv2d(channels * hist_len, latent_dim, 3, padding=1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: B, L, C, H, W
        b, l, c, h, w = history.shape
        if l != self.hist_len:
            raise ValueError(f"Expected hist_len={self.hist_len}, got {l}.")
        return self.net(history.reshape(b, l * c, h, w))


class PhaseSpectralEncoder(nn.Module):
    """FFT-based phase/amplitude encoder projected back to the spatial latent grid."""

    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.freq_net = nn.Sequential(
            nn.Conv2d(channels * 3, latent_dim, 1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
        )
        self.spatial_net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        amp = torch.log1p(torch.abs(z))
        feats = torch.cat([z.real, z.imag, amp], dim=1)
        f = self.freq_net(feats)
        # Return to the spatial support while retaining frequency-aware features.
        f_complex = torch.complex(f, torch.zeros_like(f))
        spatial = torch.fft.ifft2(f_complex, dim=(-2, -1), norm="ortho").real
        return self.spatial_net(spatial)


class FieldDecoder(nn.Module):
    """Decode latent information state to physical/observed field."""

    def __init__(self, latent_dim: int, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        return self.net(psi)


class SpectralFieldDecoder(nn.Module):
    """Small spectral residual decoder used by CIDM for phase-aware readout."""

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.SiLU(),
            nn.Conv2d(latent_dim, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(-2.2))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        z = torch.fft.fft2(psi, dim=(-2, -1), norm="ortho")
        # Simple low-pass emphasis for phase-stable readout.
        H, W = psi.shape[-2:]
        ky = torch.fft.fftfreq(H, device=psi.device).view(1, 1, H, 1)
        kx = torch.fft.fftfreq(W, device=psi.device).view(1, 1, 1, W)
        filt = torch.exp(-6.0 * (kx ** 2 + ky ** 2))
        low = torch.fft.ifft2(z * filt, dim=(-2, -1), norm="ortho").real
        return torch.sigmoid(self.scale) * self.mix(low)


class AntisymmetricFlux(nn.Module):
    """Conservative local exchange field.

    For each neighbor pair, the edge update is constructed as f(a,b)-f(b,a),
    then added to one side and subtracted from the other. This encodes the idea
    that local exchanges can redistribute information without creating it.
    """

    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Conv2d(2 * latent_dim, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_dim, 1),
        )

    def edge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.edge_net(torch.cat([a, b], dim=1)) - self.edge_net(torch.cat([b, a], dim=1))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(psi)
        # x direction
        right = torch.roll(psi, shifts=-1, dims=-1)
        fx = self.edge(psi, right)
        out = out + fx - torch.roll(fx, shifts=1, dims=-1)
        # y direction; safe for H=1 because roll is identity and antisymmetry cancels.
        down = torch.roll(psi, shifts=-1, dims=-2)
        fy = self.edge(psi, down)
        out = out + fy - torch.roll(fy, shifts=1, dims=-2)
        return out


class SkewChannelRotation(nn.Module):
    """Learned skew-symmetric channel mixing for approximately conservative flow."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.02)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        A = self.raw - self.raw.t()
        return torch.einsum("ij,bjhw->bihw", A, psi)


class HiddenReservoirTransfer(nn.Module):
    """Internal hidden reservoir transfer for unobserved latent exchange."""

    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.net = conv_block(latent_dim, latent_dim, hidden)
        self.scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale) * self.net(psi)


class ClosedInformationDynamicsCore(nn.Module):
    """CIDM closed latent vector field."""

    def __init__(self, latent_dim: int, hidden: int, use_flux: bool = True, use_hidden_reservoir: bool = True):
        super().__init__()
        self.use_flux = use_flux
        self.use_hidden_reservoir = use_hidden_reservoir
        self.skew = SkewChannelRotation(latent_dim)
        self.flux = AntisymmetricFlux(latent_dim, hidden)
        self.hidden = HiddenReservoirTransfer(latent_dim, hidden)
        self.local = conv_block(latent_dim, latent_dim, hidden)
        self.step_scale = nn.Parameter(torch.tensor(0.18))

    def vector_field(self, psi: torch.Tensor) -> torch.Tensor:
        dpsi = self.skew(psi)
        if self.use_flux:
            dpsi = dpsi + self.flux(psi)
        if self.use_hidden_reservoir:
            dpsi = dpsi + 0.25 * self.hidden(psi)
        # A small local residual stabilizes non-Hamiltonian observed fields.
        dpsi = dpsi + 0.10 * self.local(psi)
        return dpsi

    def forward(self, psi: torch.Tensor, dt_scale: float = 1.0) -> torch.Tensor:
        alpha = torch.tanh(self.step_scale) * dt_scale
        return psi + alpha * self.vector_field(psi)


class LatentSpectralOperatorCore(nn.Module):
    """Latent-domain spectral operator for CIDM-BER.

    This is a residual spectral vector field inside the latent dynamics, not an
    external FNO baseline. It learns complex low-frequency mode mixing and maps
    it back to the latent grid as a global correction to local closed dynamics.
    """

    def __init__(self, latent_dim: int, hidden: int = 64, modes1: int = 8, modes2: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / max(1, latent_dim)
        self.weight_pos = nn.Parameter(scale * torch.randn(latent_dim, latent_dim, modes1, modes2, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(latent_dim, latent_dim, modes1, modes2, 2))
        self.pre = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.GroupNorm(1, latent_dim),
            nn.SiLU(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(latent_dim, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(-1.8))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, wri: torch.Tensor) -> torch.Tensor:
        w = torch.view_as_complex(wri.contiguous())
        return torch.einsum("bihw,iohw->bohw", x, w)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        B, D, H, W = psi.shape
        x = self.pre(psi)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        out_ft = torch.zeros(B, D, H, W // 2 + 1, dtype=torch.cfloat, device=psi.device)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weight_pos[:, :, :m1, :m2])
        if H > 1:
            out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weight_neg[:, :, :m1, :m2])
        y = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1), norm="ortho")
        return torch.sigmoid(self.logit_scale) * self.post(y)


class BoundaryEnergyReservoir(nn.Module):
    """Boundary Energy Reservoir (BER) residual closure layer.

    BER is a virtual external energy reservoir, not a full external-state model.
    It learns a minimal residual exchange field that accounts for open-system
    effects not captured by the internal closed dynamics.
    """

    def __init__(self, latent_dim: int, port_channels: int, hidden: int):
        super().__init__()
        self.port_channels = max(1, port_channels)
        in_ch = latent_dim * 2 + self.port_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_dim, 3, padding=1),
        )
        mask_hidden = max(8, hidden // 2)
        self.mask = nn.Sequential(
            nn.Conv2d(in_ch, mask_hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mask_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(-3.0))

    def _port_to_grid(self, psi: torch.Tensor, port: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = psi.shape
        if port is None or port.numel() == 0:
            return torch.zeros(B, self.port_channels, H, W, device=psi.device, dtype=psi.dtype)
        if port.dim() != 4:
            raise ValueError("port must be Tensor[B, C_port, H, W] for one step.")
        p = port
        if p.shape[-2:] != (H, W):
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        if p.shape[1] < self.port_channels:
            pad = torch.zeros(B, self.port_channels - p.shape[1], H, W, device=psi.device, dtype=psi.dtype)
            p = torch.cat([p, pad], dim=1)
        elif p.shape[1] > self.port_channels:
            p = p[:, : self.port_channels]
        return p

    def forward(
        self,
        psi: torch.Tensor,
        internal_delta: torch.Tensor,
        port: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p = self._port_to_grid(psi, port)
        x = torch.cat([psi, internal_delta, p], dim=1)
        raw = torch.tanh(self.net(x))
        mask = self.mask(x)
        scale = torch.sigmoid(self.logit_scale)
        ber_delta = scale * mask * raw
        info = {
            "ber_strength": ber_delta.abs().mean(),
            "ber_mask_mean": mask.mean(),
            "ber_scale": scale,
        }
        return ber_delta, info


class CIDM(nn.Module):
    """Closed Information Dynamics Model.

    General-purpose主体架构. This version corresponds to the stabilized closed
    dynamics backbone: spatial information encoding, temporal history encoding,
    phase-spectral feature encoding, conservative local exchange, internal hidden
    reservoir transfer, and autoregressive latent rollout.
    """

    def __init__(self, config: CIDMConfig):
        super().__init__()
        self.config = config
        D = config.latent_dim
        H = config.hidden
        self.space_encoder = SpaceEncoder(config.channels, D)
        self.time_encoder = TimeHistoryEncoder(config.channels, config.hist_len, D)
        self.phase_encoder = PhaseSpectralEncoder(config.channels, D)
        fuse_inputs = D + (D if config.use_time_encoder else 0) + (D if config.use_phase_encoder else 0)
        self.fuse = nn.Sequential(nn.Conv2d(fuse_inputs, D, 1), nn.SiLU(), nn.Conv2d(D, D, 1))
        self.core = ClosedInformationDynamicsCore(D, H, use_flux=config.use_flux, use_hidden_reservoir=config.use_hidden_reservoir)
        self.decoder = FieldDecoder(D, config.channels, H)
        self.spectral_decoder = SpectralFieldDecoder(D, config.channels)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        last = history[:, -1]
        feats = [self.space_encoder(last)]
        if self.config.use_time_encoder:
            feats.append(self.time_encoder(history))
        if self.config.use_phase_encoder:
            feats.append(self.phase_encoder(last))
        return self.fuse(torch.cat(feats, dim=1))

    def decode(self, psi: torch.Tensor) -> torch.Tensor:
        return self.decoder(psi) + self.spectral_decoder(psi)

    def step(self, psi: torch.Tensor, step_index: int = 0, port: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        psi_next = self.core(psi)
        return psi_next, {"soc_strength": torch.zeros((), device=psi.device), "ber_strength": torch.zeros((), device=psi.device)}

    def _select_port_step(self, port: Optional[torch.Tensor], t: int) -> Optional[torch.Tensor]:
        if port is None:
            return None
        if port.dim() == 5:
            return port[:, min(t, port.shape[1] - 1)]
        return port

    def forward(
        self,
        history: torch.Tensor,
        horizon: Optional[int] = None,
        port: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        horizon = int(horizon or self.config.horizon)
        psi0 = self.encode(history)
        psi = psi0
        preds = []
        latent_seq = []
        soc_strengths, ber_strengths = [], []
        energy0 = (psi0 ** 2).mean(dim=(1, 2, 3))
        energies = []
        for t in range(horizon):
            psi, info = self.step(psi, t, self._select_port_step(port, t))
            y = self.decode(psi)
            preds.append(y)
            latent_seq.append(psi)
            energies.append((psi ** 2).mean(dim=(1, 2, 3)))
            soc_strengths.append(info.get("soc_strength", torch.zeros((), device=psi.device)))
            ber_strengths.append(info.get("ber_strength", torch.zeros((), device=psi.device)))
        pred = torch.stack(preds, dim=1)
        latent = torch.stack(latent_seq, dim=1)
        energy = torch.stack(energies, dim=1)
        diagnostics = {
            "energy_drift": (energy[:, -1] - energy0).abs().mean(),
            "soc_strength": torch.stack([s if s.dim() else s.expand(1) for s in soc_strengths]).mean(),
            "ber_strength": torch.stack([b if b.dim() else b.expand(1) for b in ber_strengths]).mean(),
        }
        if target is not None:
            diagnostics["rmse"] = torch.sqrt(F.mse_loss(pred, target))
            diagnostics["mae"] = F.l1_loss(pred, target)
        return {"pred": pred, "latent": psi, "latent_seq": latent, "diagnostics": diagnostics}


class CIDM_BER(CIDM):
    """CIDM with Spectral Operator Core and Boundary Energy Reservoir.

    Physics-enhanced variant for systems with salient wave/flow propagation,
    quasi-closed dynamics, boundary exchange, or energy closure requirements.
    """

    def __init__(self, config: CIDMConfig):
        super().__init__(config)
        self.soc = LatentSpectralOperatorCore(
            config.latent_dim,
            hidden=config.hidden,
            modes1=config.soc_modes1,
            modes2=config.soc_modes2,
        )
        self.ber = BoundaryEnergyReservoir(config.latent_dim, config.port_channels, config.hidden)
        self.ber_add_scale = float(config.ber_add_scale)
        self.soc_enabled = bool(config.soc_enabled)
        self.ber_enabled = bool(config.ber_enabled)

    def step(self, psi: torch.Tensor, step_index: int = 0, port: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        psi_int = self.core(psi)
        soc_delta = self.soc(psi) if self.soc_enabled else torch.zeros_like(psi)
        psi_int = psi_int + soc_delta
        internal_delta = psi_int - psi
        if self.ber_enabled:
            ber_delta, ber_info = self.ber(psi, internal_delta.detach(), port)
            psi_next = psi_int + self.ber_add_scale * ber_delta
        else:
            ber_delta = torch.zeros_like(psi)
            ber_info = {"ber_strength": torch.zeros((), device=psi.device), "ber_mask_mean": torch.zeros((), device=psi.device)}
            psi_next = psi_int
        info = {
            "soc_strength": soc_delta.abs().mean(),
            "ber_strength": ber_info["ber_strength"],
            "ber_mask_mean": ber_info.get("ber_mask_mean", torch.zeros((), device=psi.device)),
            "ber_budget_proxy": (ber_delta.mean(dim=(1, 2, 3)) - internal_delta.mean(dim=(1, 2, 3))).abs().mean(),
        }
        return psi_next, info


def build_cidm(config: Optional[CIDMConfig] = None) -> CIDM:
    return CIDM(config or CIDMConfig())


def build_cidm_ber(config: Optional[CIDMConfig] = None) -> CIDM_BER:
    cfg = config or CIDMConfig()
    return CIDM_BER(cfg)


# Backward-compatible aliases for previous internal experiment names.
ClosedInformationDynamicsModel = CIDM
ClosedInformationDynamicsBERModel = CIDM_BER
STICM_V3_FULL_ALIAS = CIDM
STICM_SOC_BER_ALIAS = CIDM_BER


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = CIDMConfig(channels=1, hist_len=4, horizon=1, latent_dim=4, hidden=8, port_channels=1)
    x = torch.randn(1, cfg.hist_len, cfg.channels, 4, 4)
    port = torch.randn(1, cfg.port_channels, 4, 4)
    for cls in [CIDM, CIDM_BER]:
        model = cls(cfg)
        out = model(x, port=port)
        print(cls.__name__, out["pred"].shape, "params=", count_parameters(model), out["diagnostics"])