"""
Numerically stable mathematical utilities for Langevin dynamics.

Adapted from LanPaint: https://arxiv.org/abs/2502.03491
"""

import torch


def epxm1_x(x):
    """
    Compute (exp(x) - 1) / x with numerical stability for x -> 0.

    This function handles the singularity at x=0 using Taylor expansion.

    Args:
        x: Input tensor

    Returns:
        (exp(x) - 1) / x, with proper handling for x -> 0

    Note:
        Taylor expansion for x -> 0: (e^x - 1)/x ≈ 1 + x/2 + x²/6
    """
    result = torch.special.expm1(x) / x
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x) < 1e-2
    # Taylor expansion: (e^x - 1)/x ≈ 1 + x/2 + x²/6
    result = torch.where(mask, 1 + x/2. + x**2 / 6., result)
    return result


def epxm1mx_x2(x):
    """
    Compute (exp(x) - 1 - x) / x² with numerical stability for x -> 0.

    This function handles the singularity at x=0 using Taylor expansion.

    Args:
        x: Input tensor

    Returns:
        (exp(x) - 1 - x) / x², with proper handling for x -> 0

    Note:
        Taylor expansion for x -> 0: (e^x - 1 - x)/x² ≈ 1/2 + x/6 + x²/24
    """
    result = (torch.special.expm1(x) - x) / (x * x)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x) < 1e-2
    # Taylor expansion: (e^x - 1 - x)/x² ≈ 1/2 + x/6 + x²/24
    result = torch.where(mask, 0.5 + x/6. + x**2 / 24., result)
    return result


def exp_1mcosh_GD(gamma_t, delta):
    """
    Compute e^(-Γt) * (1 - cosh(Γt√Δ)) / ((Γt)² * Δ) with numerical stability.

    This function handles the limit when Δ -> 0.

    Args:
        gamma_t: Product of friction coefficient Gamma and time t
        delta: Parameter Delta = 1 - 4A/Γ²

    Returns:
        The computed value with proper handling for delta -> 0

    Note:
        When delta -> 0, the limit is -exp(-Γt) / (Γt)²
    """
    # Limit when delta -> 0
    result = -torch.expm1(-gamma_t) / (gamma_t * gamma_t + 1e-8)
    mask = torch.abs(delta) > 1e-6

    # Full expression when delta != 0
    gamma_t_sqrt_delta = gamma_t * torch.sqrt(torch.clamp(delta, min=0.0))
    cosh_val = torch.cosh(gamma_t_sqrt_delta)
    numerator = torch.exp(-gamma_t) * (1 - cosh_val)
    denominator = (gamma_t * gamma_t) * delta
    result_full = numerator / (denominator + 1e-8)

    result = torch.where(mask, result_full, result)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


class StochasticHarmonicOscillator:
    """
    Stochastic Harmonic Oscillator solver for Langevin dynamics.

    This class implements the analytical solution for the underdamped,
    critically damped, and overdamped regimes of the harmonic oscillator
    with stochastic forcing.

    The dynamics follow:
        dz/dτ = q/√Γ
        dq/dτ = -Γz - ΓΔz/4 + C + ξ(τ)

    where ξ is Gaussian noise with <ξ(τ)ξ(τ')> = 2D δ(τ-τ').

    Args:
        Gamma: Friction coefficient Γ
        A: Harmonic potential strength A = Γ²(1-Δ)/4
        C: Constant force term
        D: Noise amplitude D

    Reference:
        LanPaint paper, Section 3.2 and Appendix A
    """

    def __init__(self, Gamma, A, C, D):
        """
        Initialize the stochastic harmonic oscillator.

        Args:
            Gamma: Friction coefficient (can be a tensor for spatially varying friction)
            A: Harmonic potential strength (can be a tensor)
            C: Constant force term (can be a tensor)
            D: Noise amplitude (can be a tensor)
        """
        self.Gamma = Gamma
        self.A = A
        self.C = C
        self.D = D

    def dynamics(self, y0, v0, t):
        """
        Compute position and velocity after time t.

        This method computes the analytical solution of the stochastic
        harmonic oscillator for time t, handling all three damping regimes.

        Args:
            y0: Initial position z(τ)
            v0: Initial velocity q(τ)/√Γ (can be None for first step)
            t: Time step Δτ

        Returns:
            Tuple of (y(t), v(t)): Position and velocity at time τ+t

        Note:
            The solution depends on the damping regime determined by Delta:
            - Delta < 0: Underdamped (oscillatory)
            - Delta ≈ 0: Critically damped
            - Delta > 0: Overdamped
        """
        # Compute Delta parameter
        Delta = 1 - 4 * self.A / (self.Gamma ** 2 + 1e-8)

        # Determine damping regime
        mask_underdamped = Delta < 0
        mask_critical = torch.abs(Delta) < 1e-6
        mask_overdamped = (~mask_underdamped) & (~mask_critical)

        # Compute solutions for each regime
        y_under, v_under = self._underdamped(y0, v0, t, Delta)
        y_crit, v_crit = self._critical(y0, v0, t)
        y_over, v_over = self._overdamped(y0, v0, t, Delta)

        # Select appropriate solution based on regime
        y = torch.where(mask_underdamped, y_under,
                       torch.where(mask_critical, y_crit, y_over))
        v = torch.where(mask_underdamped, v_under,
                       torch.where(mask_critical, v_crit, v_over))

        return y, v

    def _underdamped(self, y0, v0, t, Delta):
        """
        Underdamped regime (Delta < 0): oscillatory solution.

        The solution involves oscillations with frequency ω = √(-Δ) Γ/2.

        Args:
            y0: Initial position
            v0: Initial velocity
            t: Time step
            Delta: Damping parameter (negative)

        Returns:
            Tuple of (position, velocity)
        """
        omega = torch.sqrt(-Delta + 1e-8) * self.Gamma / 2
        gt = self.Gamma * t / 2

        exp_neg = torch.exp(-gt)
        cos_omega = torch.cos(omega * t)
        sin_omega = torch.sin(omega * t)

        # Position: combination of decaying exponential and oscillatory terms
        term1 = exp_neg * (self.Gamma * cos_omega + 2 * omega * sin_omega)
        term1 = term1 / (self.Gamma + 1e-8)
        y = term1 * y0

        term2 = 2 * exp_neg * sin_omega / (omega + 1e-8)
        y = y + term2 * v0

        term3 = 4 * self.C / (self.Gamma ** 2 + 1e-8)
        term3 = term3 * (1 - exp_neg * cos_omega)
        y = y + term3

        # Velocity: derivative of position
        term1_v = exp_neg * ((-self.Gamma ** 2 + 4 * omega ** 2) * cos_omega +
                            3 * self.Gamma * omega * sin_omega)
        term1_v = term1_v / (2 * omega * self.Gamma + 1e-8)
        v = term1_v * y0

        term2_v = exp_neg * (self.Gamma * cos_omega - 2 * omega * sin_omega)
        term2_v = term2_v / (self.Gamma + 1e-8)
        v = v + term2_v * v0

        term3_v = 4 * self.C * (1 + self.Gamma * t) / (self.Gamma + 1e-8)
        term3_v = term3_v * exp_neg * sin_omega / (omega + 1e-8)
        v = v + term3_v

        return y, v

    def _critical(self, y0, v0, t):
        """
        Critically damped regime (Delta ≈ 0): optimal damping solution.

        This is the boundary case between underdamped and overdamped,
        providing the fastest convergence without oscillations.

        Args:
            y0: Initial position
            v0: Initial velocity
            t: Time step

        Returns:
            Tuple of (position, velocity)
        """
        gt = self.Gamma * t / 2
        exp_neg = torch.exp(-gt)

        # Position: exponential decay without oscillation
        y = exp_neg * (1 + gt / 2) * y0 + t * exp_neg * v0
        y = y + (4 * self.C / (self.Gamma ** 2 + 1e-8)) * (1 - exp_neg * (1 + gt / 2))

        # Velocity
        v = -exp_neg * (gt / 2) * (self.Gamma / 2) * y0
        v = v + exp_neg * (1 - gt / 2) * v0
        v = v + (4 * self.C / (self.Gamma ** 2 + 1e-8)) * (1 - exp_neg * (1 - gt / 2)) * (self.Gamma / 2)

        return y, v

    def _overdamped(self, y0, v0, t, Delta):
        """
        Overdamped regime (Delta > 0): slow decay solution.

        The solution involves two decaying exponentials with different rates.

        Args:
            y0: Initial position
            v0: Initial velocity
            t: Time step
            Delta: Damping parameter (positive)

        Returns:
            Tuple of (position, velocity)
        """
        sqrt_delta = torch.sqrt(Delta + 1e-8)
        gt = self.Gamma * t / 2

        exp_pos = torch.exp(-gt * (1 - sqrt_delta))
        exp_neg = torch.exp(-gt * (1 + sqrt_delta))

        # Position: sum of two decaying exponentials
        term1 = (self.Gamma * (exp_pos + exp_neg) + sqrt_delta * (exp_pos - exp_neg))
        term1 = term1 / (2 * self.Gamma * sqrt_delta + 1e-8)
        y = term1 * y0

        term2 = (exp_pos - exp_neg) / (sqrt_delta + 1e-8)
        y = y + term2 * v0

        term3 = 4 * self.C / (self.Gamma ** 2 + 1e-8)
        term3 = term3 * (1 - (exp_pos + exp_neg) / 2)
        y = y + term3

        # Velocity
        term1_v = ((self.Gamma ** 2 * (1 + Delta) * (exp_pos - exp_neg) +
                   2 * self.Gamma * sqrt_delta * (exp_pos + exp_neg)))
        term1_v = term1_v / (4 * self.Gamma ** 2 * sqrt_delta + 1e-8)
        v = term1_v * y0

        term2_v = (self.Gamma * (exp_pos + exp_neg) + sqrt_delta * (exp_pos - exp_neg))
        term2_v = term2_v / (2 * sqrt_delta + 1e-8)
        v = v + term2_v * v0

        term3_v = 4 * self.C / (self.Gamma ** 2 + 1e-8)
        term3_v = term3_v * (1 - (exp_pos + exp_neg) / 2) * self.Gamma
        v = v + term3_v

        return y, v
