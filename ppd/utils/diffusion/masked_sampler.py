"""
Masked Euler Sampler for hybrid depth estimation.
This sampler preserves known regions (interior) from a reference depth map
while only generating/refining masked regions (edges).
"""

import torch
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.diffusion.schedule import LinearSchedule


class MaskedEulerSampler:
    """
    Euler sampler with mask-based region preservation.

    This sampler uses a keep_mask to specify which regions should be preserved
    from a known depth map (e.g., MoGe-2 interior). The masked regions (edges)
    are generated using the diffusion model, while the keep_mask regions are
    maintained throughout the sampling process.
    """

    def __init__(
        self,
        schedule: LinearSchedule,
        timesteps: Timesteps,
        prediction_type: 'velocity',
    ):
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type

    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        keep_mask: torch.Tensor = None,
        known_depth: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Step to the next timestep with optional mask preservation.

        Args:
            pred: Predicted velocity from the model
            x_t: Current latent at timestep t
            t: Current timestep
            keep_mask: Binary mask (1=keep known, 0=generate), shape (1, 1, H, W)
            known_depth: Known depth values to preserve in keep_mask regions
        """
        x_next = self.step_to(pred, x_t, t, self.get_next_timestep(t), **kwargs)

        # Apply mask preservation if provided
        if keep_mask is not None and known_depth is not None:
            # Inject known depth into masked regions
            x_next = self._apply_mask(x_next, known_depth, keep_mask, t)

        return x_next

    def step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        s = s[(...,) + (None,) * (x_t.ndim - s.ndim)] if s.ndim < x_t.ndim else s
        T = self.schedule.T
        # Step from x_t to x_s.
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        # Clamp x_s to x_0 and x_T if s is out of bound.
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        return pred_x_s

    def _apply_mask(
        self,
        x_t: torch.Tensor,
        known_depth: torch.Tensor,
        keep_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mask to preserve known regions.

        Similar to RePaint approach: inject noised known values at each timestep.
        The known depth is noised according to the current timestep, then blended
        with the generated latent using the keep_mask.

        Args:
            x_t: Current latent (generated + possibly mixed)
            known_depth: Known depth values (in latent space, x_0)
            keep_mask: Binary mask (1=keep known, 0=generate)
            t: Current timestep

        Returns:
            x_t: Masked latent with known regions preserved
        """
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        T = self.schedule.T

        # Create random noise for the known depth
        noise = torch.randn_like(known_depth)

        # Compute alpha_t and beta_t (linear schedule)
        alpha_t = 1 - t / T
        beta_t = t / T

        # Compute alpha_s for the next timestep (or use current for preservation)
        # For simple preservation, we mix the known depth with appropriate noise
        noised_known = alpha_t * known_depth + beta_t * noise

        # Apply mask: keep regions use noised_known, generate regions use x_t
        x_masked = keep_mask * noised_known + (1 - keep_mask) * x_t

        return x_masked

    def get_next_timestep(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the next sample timestep.
        Support multiple different timesteps t in a batch.
        If no more steps, return out of bound value -1 or T+1.
        """
        T = self.timesteps.T
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1

        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        s = s.where(next_idx < steps, -1)
        return s

    def initialize_latent_with_mask(
        self,
        shape: tuple,
        known_depth: torch.Tensor,
        keep_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initialize the latent with known regions masked.

        Args:
            shape: Shape of the latent (B, C, H, W)
            known_depth: Known depth values (x_0)
            keep_mask: Binary mask (1=keep known, 0=generate)
            device: Device to create tensors on

        Returns:
            latent: Initialized latent with known regions preserved
        """
        # Start with pure noise for all regions
        latent = torch.randn(shape, device=device)

        # Replace keep_mask regions with known depth
        latent = keep_mask * known_depth + (1 - keep_mask) * latent

        return latent
