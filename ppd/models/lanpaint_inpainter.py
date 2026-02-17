"""
LanPaint-style FLD inpainter for depth refinement.

This module implements a Fast Langevin Dynamics (FLD) based inpainter
for depth map refinement, adapted from the LanPaint paper.

Reference: https://arxiv.org/abs/2502.03491
"""

import torch
import torch.nn.functional as F
from functools import partial

from ppd.models.lanpaint_types import LangevinState
from ppd.models.lanpaint_utils import StochasticHarmonicOscillator


class LanPaintInpainter:
    """
    LanPaint-style inpainter using Fast Langevin Dynamics (FLD).

    This inpainter refines depth maps using:
    - Stochastic Harmonic Oscillator (SHO) for Langevin dynamics
    - BiG Score for bidirectional guidance (unknown → known, known → unknown)
    - Multiple FLD iterations per diffusion timestep for progressive refinement

    The algorithm progressively converges to the exact conditional distribution
    as the number of FLD iterations N → ∞.

    Args:
        schedule: LinearSchedule (lerp/Rectified Flow)
        sampler: EulerSampler for timestep iteration
        dit_model: DiT model for diffusion prediction
        sem_encoder: Semantic encoder (Depth Anything V2 or MoGe2)
        device: Computing device (cuda or cpu)
        n_steps: Number of FLD iterations per diffusion step (default: 5)
        step_size: Langevin step size eta (default: 0.2)
        lambda_big: BiG Score guidance strength (default: 16.0)
        friction: Friction coefficient Gamma (default: 15.0)
        beta: Ratio of y/x step sizes (default: 1.0)
        use_simplified_big: Use simplified BiG Score with single model inference
                           (default: True, set False for full dual-CFG version)

    Reference:
        LanPaint: https://arxiv.org/abs/2502.03491
        Section 3.2: Fast Langevin Dynamics for Conditional Diffusion
    """

    def __init__(self, schedule, sampler, dit_model, sem_encoder, device,
                 n_steps=5, step_size=0.2, lambda_big=16.0, friction=15.0, beta=1.0,
                 use_simplified_big=True):
        self.schedule = schedule
        self.sampler = sampler
        self.dit_model = dit_model
        self.sem_encoder = sem_encoder
        self.device = device
        self.n_steps = n_steps
        self.step_size = step_size
        self.lambda_big = lambda_big
        self.friction = friction
        self.beta = beta
        self.use_simplified_big = use_simplified_big

    @torch.no_grad()
    def inpaint(self, rgb_condition, known_depth, edge_mask, num_steps=None):
        """
        Perform LanPaint-style FLD inpainting.

        Args:
            rgb_condition: RGB conditioning image, shape (B, 3, H, W), range [0, 1]
            known_depth: Known depth from MoGe-2, shape (B, 1, H, W), range [0, 1]
            edge_mask: Edge mask where 1=edges (refine), 0=non-edges (preserve)
                       shape (B, 1, H, W)
            num_steps: Number of diffusion steps (default: use sampler.timesteps)

        Returns:
            Refined depth map, shape (B, 1, H, W), range [0, 1]
        """
        if num_steps is None:
            num_steps = len(self.sampler.timesteps)

        # 1. Initialize random noise
        x_t = torch.randn_like(known_depth)

        # 2. Compute semantics once
        semantics = self.sem_encoder.forward_semantics(rgb_condition)

        # 3. Outer loop: diffusion timesteps
        for step_idx in range(num_steps):
            t = self.sampler.timesteps[step_idx]

            # 4. Compute time parameters for RF/lerp schedule
            VE_Sigma, abt, Flow_t = self.compute_time_parameters(t)

            # 5. Replace step: ensure known regions are properly conditioned
            x_t = self.replace_step(x_t, known_depth, VE_Sigma, edge_mask)

            # 6. Convert to internal format
            x_t = self.model_to_internal(x_t, abt)

            # 7. Inner loop: FLD iterations
            args = None  # LangevinState(v, C, x0)
            for i in range(self.n_steps):
                # 7a. Prepare score function
                score_func = partial(self.compute_score,
                                    rgb_condition=rgb_condition,
                                    known_depth=known_depth,
                                    edge_mask=edge_mask,
                                    semantics=semantics,
                                    timestep=t)

                # 7b. Save previous state for Strang splitting
                x_t_prev = x_t.detach()

                # 7c. Langevin dynamics update
                x_t, args = self.langevin_dynamics(
                    x_t, score_func, edge_mask,
                    current_times=(VE_Sigma, abt, Flow_t),
                    args=args
                )

            # 8. Convert back to model format
            x_t = self.internal_to_model(x_t, abt)

            # 9. Standard denoise step (Euler sampler)
            dit_input = torch.cat([x_t, rgb_condition - 0.5], dim=1)
            pred = self.dit_model(x=dit_input, semantics=semantics, timestep=t)
            x_t = self.sampler.step(pred=pred, x_t=x_t, t=t)

        # 9. Final output (add 0.5 to convert from latent space)
        return torch.clamp(x_t + 0.5, 0.0, 1.0)

    def compute_time_parameters(self, timestep):
        """
        Compute LanPaint time parameters from PPD timestep.

        Converts PPD's lerp schedule (Rectified Flow) parameters to
        LanPaint's internal parameterization.

        Args:
            timestep: PPD timestep in [0, T]

        Returns:
            Tuple of (VE_Sigma, abt, Flow_t):
                - VE_Sigma: Noise level (formal for RF, set to s)
                - abt: Signal preservation coefficient ᾱ_t
                - Flow_t: RF time parameter s = t/T
        """
        s = timestep / self.schedule.T  # Normalize to [0, 1]

        # For Rectified Flow (lerp schedule), parameter conversion:
        Flow_t = s  # RF time
        abt = (1 - Flow_t) ** 2 / ((1 - Flow_t) ** 2 + Flow_t ** 2 + 1e-8)  # ᾱ_t
        VE_Sigma = Flow_t / (1 - Flow_t + 1e-8)  # Formal conversion for RF

        return VE_Sigma, abt, Flow_t

    def replace_step(self, x, known_depth, VE_Sigma, edge_mask):
        """
        Replace step: ensure known regions are properly conditioned.

        For RF/lerp, this simplifies to direct replacement in masked regions.

        Args:
            x: Current latent state
            known_depth: Known depth from MoGe-2 (in latent space: range [-0.5, 0.5])
            VE_Sigma: Noise level (not used for RF)
            edge_mask: Edge mask (1=edges, 0=non-edges)

        Returns:
            Updated latent state with known regions replaced
        """
        known_latent = known_depth  # Convert to latent space
        # edge_mask=1: edges (refine with x), edge_mask=0: non-edges (preserve known_latent)
        return x * edge_mask + known_latent * (1 - edge_mask)

    def model_to_internal(self, x, abt):
        """
        Convert from model format to internal format.

        For RF/Flow: x_t = x * (sqrt(abt) + sqrt(1-abt))

        Args:
            x: Model format latent
            abt: Signal preservation coefficient

        Returns:
            Internal format latent
        """
        factor = torch.sqrt(abt) + torch.sqrt(1 - abt + 1e-8)
        return x * factor

    def internal_to_model(self, x_t, abt):
        """
        Convert from internal format to model format.

        For RF/Flow: x = x_t / (sqrt(abt) + sqrt(1-abt))

        Args:
            x_t: Internal format latent
            abt: Signal preservation coefficient

        Returns:
            Model format latent
        """
        factor = torch.sqrt(abt) + torch.sqrt(1 - abt + 1e-8)
        return x_t / (factor + 1e-8)

    def compute_score(self, x_t, rgb_condition, known_depth, edge_mask, semantics, timestep):
        """
        Compute score function using BiG Score.

        BiG Score provides bidirectional guidance:
        - Unknown regions: standard score toward predicted x_0
        - Known regions: (1+λ)*(y - x_t) + λ*score_to_big

        Args:
            x_t: Current latent state (internal format)
            rgb_condition: RGB conditioning
            known_depth: Known depth from MoGe-2
            edge_mask: Edge mask
            semantics: Semantic features
            timestep: Current timestep

        Returns:
            Score function, shape (B, 1, H, W)
        """
        # Model prediction
        dit_input = torch.cat([x_t, rgb_condition - 0.5], dim=1)
        pred = self.dit_model(x=dit_input, semantics=semantics, timestep=timestep)

        # Convert velocity to x_0
        pred_x0, _ = self.schedule.convert_from_pred(pred, 'velocity', x_t, timestep)

        # Simplified BiG Score (single inference version)
        y = known_depth  # Convert to latent space

        # Standard score for unknown regions
        score_unknown = -(x_t - pred_x0)

        # BiG Score for known regions: (1+λ)*(y - x_t) + λ*score_unknown
        score_known = (1 + self.lambda_big) * (y - x_t) - self.lambda_big * score_unknown

        # Mix based on edge mask
        # edge_mask=1: edges (use score_unknown), edge_mask=0: non-edges (use score_known)
        score = score_unknown * edge_mask + score_known * (1 - edge_mask)
        return score

    def langevin_dynamics(self, x_t, score_func, mask, current_times, args=None):
        """
        Perform Langevin dynamics update using Stochastic Harmonic Oscillator.

        This implements the full FLD update with SHO, falling back to
        overdamped Langevin if numerical issues occur.

        Args:
            x_t: Current state (internal format)
            score_func: Score function
            mask: Edge mask
            current_times: Tuple of (VE_Sigma, abt, Flow_t)
            args: LangevinState (v, C, x0)

        Returns:
            Tuple of (x_t_next, LangevinState)
        """
        # 1. Prepare parameters
        sigma, abt, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y = self.prepare_step_size(
            current_times, self.step_size
        )

        # 2. Mix parameters based on mask
        # mask=1 (edges): use _x parameters (unknown regions), mask=0 (non-edges): use _y parameters (known regions)
        A = A_x * mask + A_y * (1 - mask)
        D = D_x * mask + D_y * (1 - mask)
        dt = dtx * mask + dty * (1 - mask)
        Gamma = Gamma_x * mask + Gamma_y * (1 - mask)

        # 3. Compute constant force term C
        def Coef_C(x_t):
            x0 = x_t + score_func(x_t)  # Tweedie estimator
            C = (torch.sqrt(abt) * x0 - x_t) / (1 - abt + 1e-8) + A * x_t
            return C, x0

        # 4. Run damped dynamics with Strang splitting
        def run_damped(x_t, args):
            if args is None:
                v = None
                C, x0 = Coef_C(x_t)
                x_t, v = self.advance_time(x_t, v, dt, Gamma, A, C, D)
            else:
                v = args.v
                C = args.C
                # Strang splitting: half step C, half step dynamics, half step C
                x_t, v = self.advance_time(x_t, v, dt/2, Gamma, A, C, D)
                C_new, x0 = Coef_C(x_t)
                v = v + torch.sqrt(Gamma + 1e-8) * (C_new - C) * dt
                x_t, v = self.advance_time(x_t, v, dt/2, Gamma, A, C_new, D)
                C = C_new
            return x_t, LangevinState(v, C, x0)

        # 5. Execute update with NaN fallback
        try:
            x_t_next, state = run_damped(x_t, args)
            if torch.isnan(x_t_next).any() or (state.v is not None and torch.isnan(state.v).any()):
                raise ValueError("NaN detected in Langevin dynamics")
            x_t = x_t_next
        except Exception as e:
            # Fallback to overdamped version
            x_t, state = self.run_overdamped(x_t, args, Coef_C, dt, A, D)

        return x_t, state

    def prepare_step_size(self, current_times, step_size):
        """
        Prepare all parameters for Langevin dynamics.

        Computes time steps, friction coefficients, harmonic potential strengths,
        and noise amplitudes for both unknown (x) and known (y) regions.

        Args:
            current_times: Tuple of (VE_Sigma, abt, Flow_t)
            step_size: Langevin step size eta

        Returns:
            Tuple of (sigma, abt, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y)
        """
        sigma, abt, flow_t = current_times

        # Time steps
        dtx = 2 * step_size * self.sigma_x(abt)
        dty = 2 * step_size * self.sigma_y(abt)

        # Friction parameters
        s = flow_t
        Gamma_hat_x = self.friction ** 2 * step_size * self.sigma_x(abt) / 0.1 * s ** 0
        Gamma_hat_y = self.friction ** 2 * step_size * self.sigma_y(abt) / 0.1 * s ** 0
        Gamma_hat_x /= 2.0
        Gamma_hat_y /= 2.0

        # Harmonic potential strengths
        A_t_x = (1) / (1 - abt + 1e-8) * dtx / 2
        A_t_y = (1 + self.lambda_big) / (1 - abt + 1e-8) * dty / 2

        # Normalize
        A_x = A_t_x / (dtx / 2 + 1e-8)
        A_y = A_t_y / (dty / 2 + 1e-8)
        Gamma_x = Gamma_hat_x / (dtx / 2 + 1e-8)
        Gamma_y = Gamma_hat_y / (dty / 2 + 1e-8)

        # Noise amplitudes
        D_x = torch.sqrt(2 * abt ** 0 + 1e-8)
        D_y = torch.sqrt(2 * abt ** 0 + 1e-8)

        return sigma, abt, dtx/2, dty/2, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y

    def sigma_x(self, abt):
        """Compute sigma_x for unknown regions."""
        return abt ** 0  # For RF, sigma_x = 1

    def sigma_y(self, abt):
        """Compute sigma_y for known regions."""
        beta = self.beta * abt ** 0  # For RF, beta = constant
        return beta

    def advance_time(self, x_t, v, dt, Gamma, A, C, D):
        """
        Advance time using Stochastic Harmonic Oscillator.

        Args:
            x_t: Current position
            v: Current velocity (can be None)
            dt: Time step
            Gamma: Friction coefficient
            A: Harmonic potential strength
            C: Constant force
            D: Noise amplitude

        Returns:
            Tuple of (x_t_next, v_next)
        """
        dtype = x_t.dtype
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            osc = StochasticHarmonicOscillator(Gamma, A, C, D)
            x_t, v = osc.dynamics(x_t, v, dt)

        x_t = x_t.to(dtype)
        v = v.to(dtype) if v is not None else None
        return x_t, v

    def run_overdamped(self, x_t, args, Coef_C, dt, A, D):
        """
        Overdamped Langevin dynamics (fallback implementation).

        This is a simplified version that ignores the velocity term,
        corresponding to the limit of infinite friction.

        Args:
            x_t: Current state
            args: LangevinState
            Coef_C: Function to compute constant force C
            dt: Time step
            A: Harmonic potential strength
            D: Noise amplitude

        Returns:
            Tuple of (x_t_next, LangevinState)
        """
        dtype = x_t.dtype
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            if args is None:
                C, x0 = Coef_C(x_t)
            else:
                C = args.C

            # Overdamped update: x = x*exp(-A*dt) + C*k + noise
            A_dt = A * dt
            exp_neg = torch.exp(-A_dt)
            eps = 1e-8
            abs_A = torch.abs(A)
            k = torch.where(abs_A < eps, dt, (-torch.expm1(-A_dt)) / (A + eps))
            k2 = torch.where(abs_A < eps, dt, (-torch.expm1(-2 * A_dt)) / (2 * A + eps))

            mean = exp_neg * x_t + k * C
            var = (D ** 2) * k2
            noise = torch.randn_like(x_t) * torch.sqrt(torch.clamp(var, min=0.0))
            x_t = (mean + noise).to(dtype)

        if args is None:
            C, x0 = Coef_C(x_t)
        else:
            C = args.C
            x0 = args.x0 if hasattr(args, 'x0') else (x_t + torch.zeros_like(x_t))

        return x_t, LangevinState(None, C, x0)
