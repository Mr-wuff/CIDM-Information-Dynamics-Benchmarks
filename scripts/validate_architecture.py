"""Smoke-test the CIDM reference implementation using synthetic inputs.

This is not a reproduction of the manuscript's PDEBench experiments.
"""
import json
import torch
from cidm import CIDMConfig, build_cidm, build_cidm_ber, count_parameters

def check(factory):
    config=CIDMConfig(channels=2,hist_len=4,horizon=3,latent_dim=8,hidden=16,port_channels=2,soc_modes1=4,soc_modes2=4)
    model=factory(config)
    history=torch.randn(2,4,2,16,16)
    port=torch.randn(2,3,2,16,16)
    outputs=model(history,horizon=3,port=port)
    assert outputs['pred'].shape==(2,3,2,16,16)
    assert outputs['latent_seq'].shape==(2,3,8,16,16)
    assert torch.isfinite(outputs['pred']).all()
    outputs['pred'].mean().backward()
    grad_count=sum(p.grad is not None for p in model.parameters())
    return {'parameters':count_parameters(model),'gradient_tensors':grad_count,'prediction_shape':list(outputs['pred'].shape)}

if __name__=='__main__':
    torch.manual_seed(17)
    print(json.dumps({'cidm':check(build_cidm),'cidm_with_boundary_reservoir':check(build_cidm_ber)},indent=2))
