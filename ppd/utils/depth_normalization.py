"""
Depth normalization utilities for converting between metric depth and PPD format.

This module provides functions to normalize metric depth (e.g., from MoGe-2) to the
[-0.5, 0.5] range expected by Pixel-Perfect Depth, and to denormalize back to metric scale.

The normalization follows the same logic as ppd_train.py:get_gt():
1. Log transform: log(depth + 1)
2. Quantile clipping (2nd to 98th percentile)
3. Normalize to [-0.5, 0.5]
"""

import torch
from typing import Tuple, Optional


def normalize_depth_for_ppd(
    depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Normalize metric depth to PPD format [-0.5, 0.5].

    This function applies the same normalization as used in PPD training:
    1. Log transform: depth_log = log(depth + 1.0)
    2. Quantile clipping at 2% and 98%
    3. Normalize to [0, 1] then shift to [-0.5, 0.5]

    Args:
        depth: (B, 1, H, W) metric depth in meters (e.g., from MoGe-2)
        mask: (B, 1, H, W) valid pixel mask (1=valid, 0=invalid). If None, all pixels are valid.

    Returns:
        normalized_depth: (B, 1, H, W) depth in [-0.5, 0.5] range
        normalization_params: tuple of (min_val, max_val) used for denormalization
            - min_val: (B, 1, 1, 1) minimum log depth values per batch
            - max_val: (B, 1, 1, 1) maximum log depth values per batch

    Reference:
        ppd/models/ppd_train.py:102-123 - get_gt() function
    """
    if mask is None:
        mask = torch.ones_like(depth, dtype=torch.bool)
    else:
        mask = mask.bool()

    B = depth.shape[0]
    min_val = []
    max_val = []

    # Clip extreme values and apply log transform
    clip_mask = mask & (depth < 80.)
    depth_log = torch.log(depth + 1.0)

    # Compute quantiles for each batch
    for i in range(B):
        i_depth = depth_log[i]
        i_mask = clip_mask[i]
        i_min_val = torch.quantile(i_depth[i_mask], 0.02)
        i_max_val = torch.quantile(i_depth[i_mask], 0.98)
        min_val.append(i_min_val)
        max_val.append(i_max_val)

    min_val = torch.stack(min_val)
    max_val = torch.stack(max_val)

    # Handle edge case where depth range is too small
    invalid_mask = (max_val - min_val) < 1e-6
    if invalid_mask.any():
        max_val[invalid_mask] = min_val[invalid_mask] + 1e-6

    # Reshape for broadcasting
    min_val = min_val[:, None, None, None]
    max_val = max_val[:, None, None, None]

    # Normalize to [0, 1] then shift to [-0.5, 0.5]
    depth_normalized = (depth_log - min_val) / (max_val - min_val)
    # depth_normalized = torch.clamp(depth_normalized, -0.5, 1.0)
    depth_normalized = depth_normalized - 0.5

    return depth_normalized, (min_val, max_val)


def denormalize_depth_from_ppd(
    normalized_depth: torch.Tensor,
    original_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalization_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Denormalize depth from PPD format back to metric scale.

    This function reverses the normalization applied by normalize_depth_for_ppd().
    Either the original metric depth or the normalization parameters must be provided.

    Args:
        normalized_depth: (B, 1, H, W) depth in [-0.5, 0.5] range
        original_depth: (B, 1, H, W) original metric depth in meters (used to compute params)
        mask: (B, 1, H, W) valid pixel mask (1=valid, 0=invalid). If None, all pixels are valid.
        normalization_params: Pre-computed (min_val, max_val) tuple from normalize_depth_for_ppd.
            If provided, original_depth and mask are not needed.

    Returns:
        metric_depth: (B, 1, H, W) depth in meters (approximately matches original scale)
    """
    if normalization_params is not None:
        min_val, max_val = normalization_params
    else:
        # Recompute normalization params from original depth
        _, normalization_params = normalize_depth_for_ppd(original_depth, mask)
        min_val, max_val = normalization_params

    # Reverse the normalization
    # First shift from [-0.5, 0.5] to [0, 1]
    depth_normalized = normalized_depth + 0.5

    # Then denormalize using the same min/max values
    depth_log = depth_normalized * (max_val - min_val) + min_val

    # Reverse the log transform
    metric_depth = torch.exp(depth_log) - 1.0

    return metric_depth
