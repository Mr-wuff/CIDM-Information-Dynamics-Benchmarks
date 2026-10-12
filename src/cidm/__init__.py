"""Closed Information Dynamics Model: reference architectures."""
from .architectures import (CIDM, CIDM_BER, CIDMConfig, build_cidm,
                            build_cidm_ber, count_parameters)
__all__ = ['CIDM', 'CIDM_BER', 'CIDMConfig', 'build_cidm', 'build_cidm_ber', 'count_parameters']
