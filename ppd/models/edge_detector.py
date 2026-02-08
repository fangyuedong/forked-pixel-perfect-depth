"""
Edge detection for depth maps using Sobel filtering.

This module provides edge detection functionality to identify regions in depth maps
that require refinement. It uses Sobel gradient computation followed by thresholding
and morphological dilation.
"""

import cv2
import numpy as np
import torch
from typing import Tuple


class EdgeDetector:
    """
    Edge detector for depth maps using Sobel filtering.

    This detector identifies edges in depth maps by computing the gradient magnitude
    using Sobel filters, applying thresholding, and optionally dilating the edges.

    Attributes:
        threshold: Gradient magnitude threshold for edge detection (default: 0.1)
        dilation_kernel_size: Size of the structuring element for dilation (default: 3)
    """

    def __init__(self, threshold: float = 0.1, dilation_kernel_size: int = 3):
        """
        Initialize the edge detector.

        Args:
            threshold: Gradient magnitude threshold (0-1 range for normalized depth)
            dilation_kernel_size: Size of dilation kernel (odd number recommended)
        """
        self.threshold = threshold
        self.dilation_kernel_size = dilation_kernel_size

    def detect_edges(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Detect edges in a depth map.

        Args:
            depth: (B, 1, H, W) depth tensor, can be in any range (e.g., [-0.5, 0.5] for PPD)

        Returns:
            edge_mask: (B, 1, H, W) binary mask where:
                - 1 = edge/unknown region (needs refinement)
                - 0 = non-edge/known region (preserve original)
        """
        B, _, H, W = depth.shape
        device = depth.device
        dtype = depth.dtype

        # Process each batch element
        edge_masks = []
        for b in range(B):
            # Convert to numpy for OpenCV operations
            depth_np = depth[b, 0].cpu().numpy().astype(np.float32)

            # Normalize to [0, 1] for gradient computation if needed
            depth_min = depth_np.min()
            depth_max = depth_np.max()
            if depth_max > depth_min:
                depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = depth_np

            # Compute gradients using Sobel filter
            grad_x = cv2.Sobel(depth_normalized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_normalized, cv2.CV_64F, 0, 1, ksize=3)

            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize gradient to [0, 1] for consistent thresholding
            grad_max = gradient_magnitude.max()
            if grad_max > 0:
                gradient_magnitude = gradient_magnitude / grad_max

            # Apply threshold
            edge_mask_np = (gradient_magnitude > self.threshold).astype(np.uint8)

            # Apply dilation if kernel size > 0
            if self.dilation_kernel_size > 0:
                kernel = np.ones(
                    (self.dilation_kernel_size, self.dilation_kernel_size),
                    dtype=np.uint8
                )
                edge_mask_np = cv2.dilate(edge_mask_np, kernel, iterations=1)

            # Convert back to torch tensor
            edge_mask_tensor = torch.from_numpy(edge_mask_np).to(device=device, dtype=dtype)
            edge_masks.append(edge_mask_tensor)

        # Stack along batch dimension
        edge_mask = torch.stack(edge_masks).unsqueeze(1)  # (B, 1, H, W)

        return edge_mask

    def detect_edges_numpy(self, depth: np.ndarray) -> np.ndarray:
        """
        Detect edges in a depth map (numpy version).

        Args:
            depth: (H, W) depth map as numpy array

        Returns:
            edge_mask: (H, W) binary mask where 1=edge, 0=non-edge
        """
        # Normalize to [0, 1] for gradient computation
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = depth

        # Compute gradients using Sobel filter
        grad_x = cv2.Sobel(depth_normalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_normalized, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradient to [0, 1]
        grad_max = gradient_magnitude.max()
        if grad_max > 0:
            gradient_magnitude = gradient_magnitude / grad_max

        # Apply threshold
        edge_mask = (gradient_magnitude > self.threshold).astype(np.uint8)

        # Apply dilation
        if self.dilation_kernel_size > 0:
            kernel = np.ones(
                (self.dilation_kernel_size, self.dilation_kernel_size),
                dtype=np.uint8
            )
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        return edge_mask
