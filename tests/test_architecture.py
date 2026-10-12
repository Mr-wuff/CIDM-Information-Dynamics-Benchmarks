import torch
from cidm.architectures import CIDMConfig, SkewChannelRotation, AntisymmetricFlux, build_cidm, build_cidm_ber

def test_skew_symmetric_generator():
    module=SkewChannelRotation(6)
    A=module.raw-module.raw.T
    assert torch.allclose(A+A.T,torch.zeros_like(A),atol=1e-7)

def test_flux_spatial_cancellation_periodic_boundary():
    module=AntisymmetricFlux(4,8)
    state=torch.randn(2,4,12,12)
    flux=module(state)
    assert torch.isfinite(flux).all()
    assert flux.sum(dim=(-1,-2)).abs().max().item()<1e-4

def test_model_output_contract():
    config=CIDMConfig(channels=2,hist_len=4,horizon=2,latent_dim=8,hidden=16,port_channels=1,soc_modes1=3,soc_modes2=3)
    history=torch.randn(2,4,2,12,12)
    for model in (build_cidm(config),build_cidm_ber(config)):
        output=model(history)
        assert output['pred'].shape==(2,2,2,12,12)
        assert output['latent_seq'].shape==(2,2,8,12,12)
        assert torch.isfinite(output['pred']).all()
