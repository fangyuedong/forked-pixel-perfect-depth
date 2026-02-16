"""
Type definitions for LanPaint inpainter.

These types are used to maintain state during Langevin dynamics iterations.
"""

from typing import NamedTuple, Optional
import torch


class LangevinState(NamedTuple):
    """
    State container for Langevin dynamics iterations.

    This namedtuple maintains the necessary state information between
    successive Langevin dynamics steps, enabling efficient computation
    through the Strang splitting scheme.

    Attributes:
        v: Velocity/momentum term q(τ)/√Γ. None for the first iteration,
           initialized after the first step. Shape: (B, 1, H, W)
        C: Constant force term from the harmonic potential. This represents
           the confining force that guides the diffusion toward the target.
           Shape: (B, 1, H, W)
        x0: Estimated clean image x₀ using Tweedie's formula:
            x₀ = x_t + score(x_t). This provides a denoised estimate
            used in computing the constant force. Shape: (B, 1, H, W)

    Note:
        The velocity term v represents the conjugate momentum in the
        Hamiltonian formulation of Langevin dynamics. It is initialized
        to None on the first step and computed during time evolution.

        The constant force C is recomputed at each half-step in the
        Strang splitting scheme to maintain second-order accuracy.
    """
    v: Optional[torch.Tensor]      # Velocity/momentum
    C: Optional[torch.Tensor]      # Constant force term
    x0: Optional[torch.Tensor]     # Estimated clean image x₀
