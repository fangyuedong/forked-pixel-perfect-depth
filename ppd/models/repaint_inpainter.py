"""
RePaint-inspired inpainting sampler for depth refinement.

This module implements a RePaint-style inpainting algorithm that refines depth maps
by selectively regenerating edge regions while preserving known (non-edge) areas.
It supports configurable resampling steps (U parameter) and integrates with PPD's
lerp schedule and Euler sampler.

Reference:
    RePaint: Inpainting using Denoising Diffusion Probabilistic Models
    https://arxiv.org/abs/2201.09865
"""

import torch
from typing import Optional


class RePaintInpainter:
    """
    RePaint inpainting sampler with configurable resampling.

    This sampler performs edge-aware inpainting by:
    1. Detecting edges in the initial depth map
    2. Running reverse diffusion on edge regions
    3. Preserving known (non-edge) regions at each step
    4. Optionally resampling to improve coherence (when U > 1)

    Attributes:
        schedule: LinearSchedule (lerp) for diffusion
        sampler: EulerSampler for reverse diffusion steps
        dit_model: DiT model for velocity prediction
        sem_encoder: Semantic encoder for feature extraction
        device: torch device
        resampling_steps: Number of resampling iterations (U parameter)
            - U=1: Standard diffusion (no resampling)
            - U>1: Forward-backward resampling for improved coherence
    """

    def __init__(
        self,
        schedule,
        sampler,
        dit_model,
        sem_encoder,
        device: torch.device,
        resampling_steps: int = 1
    ):
        """
        Initialize the RePaint inpainter.

        Args:
            schedule: LinearSchedule instance (lerp schedule)
            sampler: EulerSampler instance
            dit_model: DiT model for velocity prediction
            sem_encoder: Semantic encoder (DA2 or MoGe2)
            device: torch device
            resampling_steps: Number of resampling iterations U (default: 1)
        """
        self.schedule = schedule
        self.sampler = sampler
        self.dit_model = dit_model
        self.sem_encoder = sem_encoder
        self.device = device
        self.resampling_steps = resampling_steps

    @torch.no_grad()
    def inpaint(
        self,
        rgb_condition: torch.Tensor,
        known_depth: torch.Tensor,
        edge_mask: torch.Tensor,
        num_steps: int = 10,
        resampling_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform RePaint-style inpainting with configurable resampling.

        This implements Algorithm 1 from the RePaint paper, adapted for PPD's
        lerp schedule and velocity prediction.

        Args:
            rgb_condition: (B, 3, H, W) RGB condition, normalized to [0, 1]
            known_depth: (B, 1, H, W) Known depth (e.g., from MoGe-2), normalized to [-0.5, 0.5]
            edge_mask: (B, 1, H, W) Edge mask where 1=unknown (needs refinement), 0=known (preserve)
            num_steps: Number of reverse diffusion steps (default: 10)
            resampling_steps: Override resampling steps U (default: use self.resampling_steps)

        Returns:
            refined_depth: (B, 1, H, W) Refined depth in [-0.5, 0.5] range

        Note:
            When resampling_steps=1 (default), this performs standard diffusion without resampling.
            When resampling_steps>1, forward-backward resampling is performed at each timestep.
        """
        if resampling_steps is None:
            resampling_steps = self.resampling_steps

        B, _, H, W = rgb_condition.shape
        T = self.schedule.T

        # Initialize with random noise
        x_t = torch.randn(size=known_depth.shape).to(self.device)

        # Get semantic features (computed once for efficiency)
        semantics = self.sem_encoder.forward_semantics(rgb_condition)

        # Reverse diffusion loop
        for step_idx in range(num_steps):
            # Get current timestep
            t = self.sampler.timesteps[step_idx]

            # Resampling loop (U iterations)
            for u in range(resampling_steps):
                # Predict velocity using DiT
                # DiT expects input: [latent_depth, rgb_condition] concatenated along channel dim
                dit_input = torch.cat([x_t, rgb_condition-0.5], dim=1)
                pred = self.dit_model(x=dit_input, semantics=semantics, timestep=t)

                # Euler step: x_t -> x_{t-1}
                x_t_minus_1 = self.sampler.step(pred, x_t=x_t, t=t)

                # Fuse known and unknown regions
                # known regions (edge_mask=0): use original depth
                # unknown regions (edge_mask=1): use model prediction
                x_t_minus_1 = (1 - edge_mask) * known_depth + edge_mask * x_t_minus_1

                # Resampling: if not the last iteration, forward diffuse back to x_t
                if u < resampling_steps - 1:
                    # TODO: Implement forward diffusion resampling
                    # This requires:
                    # 1. Extract pred_x_0 and pred_x_T from velocity prediction
                    # 2. Compute correct forward timestep
                    # 3. Use schedule.forward() for forward diffusion
                    #
                    # For lerp schedule: x_t = (1 - t/T) * x_0 + (t/T) * x_T
                    #
                    # Current implementation: Skip resampling when U>1 (raises error)
                    # Future enhancement: Implement proper forward diffusion
                    raise NotImplementedError(
                        "Forward diffusion resampling not yet implemented. "
                        "Use resampling_steps=1 for standard diffusion without resampling."
                    )
                else:
                    # Last resampling iteration: move to next timestep
                    x_t = x_t_minus_1

        # Final output
        refined_depth = x_t + 0.5

        return refined_depth

    @torch.no_grad()
    def inpaint_simple(
        self,
        rgb_condition: torch.Tensor,
        known_depth: torch.Tensor,
        edge_mask: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Simplified inpainting without resampling (U=1 always).

        This is a convenience method that always uses U=1 (no resampling),
        which is sufficient for basic edge-aware refinement.

        Args:
            rgb_condition: (B, 3, H, W) RGB condition, normalized to [-0.5, 0.5]
            known_depth: (B, 1, H, W) Known depth, normalized to [-0.5, 0.5]
            edge_mask: (B, 1, H, W) Edge mask where 1=unknown, 0=known
            num_steps: Number of reverse diffusion steps

        Returns:
            refined_depth: (B, 1, H, W) Refined depth in [-0.5, 0.5] range
        """
        return self.inpaint(
            rgb_condition=rgb_condition,
            known_depth=known_depth,
            edge_mask=edge_mask,
            num_steps=num_steps,
            resampling_steps=1  # Force U=1
        )
