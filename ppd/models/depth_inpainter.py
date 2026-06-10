"""
Depth inpainting pipeline using PPD + LanPaint with RANSAC-based normalization.

Pipeline:
  1. PPD predicts relative depth (ppd_rel ~[0,1])
  2. RANSAC: log(gt+1) = a * ppd_rel + b
  3. GT → PPD space: gt_ppd = (log(gt+1) - b) / a - 0.5
  4. LanPaint inpainting with gt_ppd
  5. RANSAC back: log(gt+1) = a2 * (result+0.5) + b2 → metric depth
"""

import cv2
import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from ppd.models.ppd import PixelPerfectDepth
from ppd.models.lanpaint_inpainter import LanPaintInpainter
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.transform import resize_keep_aspect
from ppd.utils.utils import has_native_bf16


def fit_ransac(pred, gt_log, seed=42):
    """Fit log(gt+1) = a * pred + b, return (a, b)."""
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        RANSACRegressor(max_trials=1000, random_state=seed),
    )
    model.fit(pred[:, None], gt_log[:, None])
    a = model.named_steps['ransacregressor'].estimator_.coef_.item()
    b = model.named_steps['ransacregressor'].estimator_.intercept_.item()
    return a, b


class DepthInpaintPipeline:
    """Sparse metric depth → dense metric depth via PPD + LanPaint."""

    def __init__(self, device, semantics_model='DA2', sampling_steps=10,
                 fld_steps=5, fld_step_size=0.2, fld_lambda=16.0, fld_friction=15.0,
                 debug_dir=None):
        self.device = device

        if semantics_model == 'MoGe2':
            semantics_pth = 'checkpoints/moge2.pt'
            model_pth = 'checkpoints/ppd_moge.pth'
        else:
            semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
            model_pth = 'checkpoints/ppd.pth'

        self.model = PixelPerfectDepth(
            semantics_model=semantics_model,
            semantics_pth=semantics_pth,
            sampling_steps=sampling_steps,
        )
        self.model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
        self.model = self.model.to(device).eval()

        schedule = LinearSchedule(T=1000)
        timesteps = Timesteps(T=1000, steps=sampling_steps, device=device)
        sampler = EulerSampler(schedule, timesteps, 'velocity')

        self.inpainter = LanPaintInpainter(
            schedule=schedule, sampler=sampler,
            dit_model=self.model.dit, sem_encoder=self.model.sem_encoder,
            device=device, n_steps=fld_steps, step_size=fld_step_size,
            lambda_big=fld_lambda, friction=fld_friction,
        )
        self.sampling_steps = sampling_steps
        self.debug_dir = debug_dir

    def __call__(self, image_bgr, sparse_depth, intrinsic=None):
        """Inpaint sparse metric depth using RGB image.

        Args:
            image_bgr: (H, W, 3) uint8 BGR image.
            sparse_depth: (H, W) float metric depth in meters. 0 = invalid.
            intrinsic: (3, 3) camera intrinsic. Defaults to KITTI image_02.

        Returns:
            dense_depth: (H, W) float64 dense metric depth in meters (original resolution).
            dense_depth_ppd: (rH, rW) float64 dense metric depth at PPD resolution.
            resized_image: (rH, rW, 3) uint8 BGR resized image (for point cloud).
            intrinsic_scaled: (3, 3) intrinsic scaled to resized resolution.
        """
        H, W = image_bgr.shape[:2]
        resize_image = resize_keep_aspect(image_bgr)
        rH, rW = resize_image.shape[:2]
        rgb_resize = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

        depth_resized = cv2.resize(sparse_depth, (rW, rH), interpolation=cv2.INTER_NEAREST)
        valid = depth_resized > 0

        # Step 1: PPD relative depth
        depth_ppd, _ = self.model.infer_image(image_bgr)
        ppd_rel = depth_ppd.squeeze().cpu().numpy()

        # Step 2: RANSAC
        ppd_valid = ppd_rel[valid].astype(np.float32)
        gt_log_valid = np.log(depth_resized[valid].astype(np.float32) + 1.0)
        a, b = fit_ransac(ppd_valid, gt_log_valid)

        # Step 3: GT → PPD space
        gt_log = np.log(depth_resized.astype(np.float32) + 1.0)
        gt_in_ppd = np.where(valid, (gt_log - b) / a - 0.5, 0.0)

        # Step 4: Inpaint
        known_depth = torch.from_numpy(gt_in_ppd.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        inpaint_mask = 1.0 - torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        rgb_condition = torch.from_numpy(rgb_resize / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Debug: save RGB + known_depth overlay at PPD resolution
        if self.debug_dir:
            import os
            from kitti_utils import visualize_depth_overlay
            os.makedirs(self.debug_dir, exist_ok=True)
            known_mask = (1.0 - inpaint_mask.squeeze().cpu().numpy()).astype(bool)
            known_metric = np.where(known_mask, depth_resized, 0.0)
            overlay = visualize_depth_overlay(rgb_resize, known_metric, alpha=0.6, point_size=1)
            cv2.imwrite(os.path.join(self.debug_dir, "known_depth_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            refined = self.inpainter.inpaint(
                rgb_condition=rgb_condition,
                known_depth=known_depth,
                edge_mask=inpaint_mask,
                num_steps=self.sampling_steps,
            )

        ppd_inpaint_rel = refined.squeeze().cpu().numpy()

        # Step 5: RANSAC back to metric
        inpaint_01 = ppd_inpaint_rel + 0.5
        inpaint_valid = inpaint_01[valid].astype(np.float32)
        a2, b2 = fit_ransac(inpaint_valid, gt_log_valid)

        inpaint_metric = np.exp(a2 * inpaint_01 + b2) - 1.0
        inpaint_metric = np.clip(inpaint_metric, 0, 200)
        dense_depth = cv2.resize(inpaint_metric, (W, H), interpolation=cv2.INTER_LINEAR)

        # Intrinsic scaled to resized resolution
        if intrinsic is None:
            intrinsic = np.array([
                [707.0912, 0.0, 601.8873],
                [0.0, 707.0912, 183.1104],
                [0.0, 0.0, 1.0],
            ])
        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[0, 0] *= rW / W
        intrinsic_scaled[1, 1] *= rH / H
        intrinsic_scaled[0, 2] *= rW / W
        intrinsic_scaled[1, 2] *= rH / H

        return dense_depth, inpaint_metric, resize_image, intrinsic_scaled
