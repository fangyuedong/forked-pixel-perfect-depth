"""
Step 3: Verify if PPD's depth distribution causes near-distance distortion.

Pipeline:
  1. Run PPD + MoGe metric alignment → depth_metric_ppd
  2. Mask with KITTI GT valid pixels → depth_metric_ppd_masked (save as KITTI 16-bit PNG)
  3. Run run_kitti_inpaint pipeline with depth_metric_ppd_masked → depth + point cloud

All outputs go to step3/
"""

import os

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from kitti_utils import read_depth_png, save_gt_point_cloud
from ppd.moge.model.v2 import MoGeModel
from ppd.models.ppd import PixelPerfectDepth
from ppd.models.lanpaint_inpainter import LanPaintInpainter
from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.depth2pcd import depth2pcd
from ppd.utils.depth_normalization import normalize_depth_for_ppd, denormalize_depth_from_ppd
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.set_seed import set_seed
from ppd.utils.transform import resize_keep_aspect
from ppd.utils.utils import has_native_bf16

OUTDIR = "step3"
RGB_PATH = "/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png"
DEPTH_PATH = "/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png"


def save_depth_as_kitti_png(depth_meters: np.ndarray, path: str):
    """Save a depth map in KITTI 16-bit PNG format (depth_m = pixel / 256)."""
    encoded = np.clip(np.round(depth_meters * 256), 0, 65535).astype(np.uint16)
    cv2.imwrite(path, encoded)


def run_inpaint(rgb_path: str, depth_png_path: str, out_name: str,
                model, inpainter, DEVICE, sampling_steps):
    """Run the PPD+LanPaint inpainting pipeline on a single image pair."""
    image = cv2.imread(rgb_path)
    H, W = image.shape[:2]
    depth_meters = read_depth_png(depth_png_path)

    resize_image = resize_keep_aspect(image)
    rH, rW = resize_image.shape[:2]

    depth_resized = cv2.resize(depth_meters, (rW, rH), interpolation=cv2.INTER_NEAREST)

    mask = (depth_resized > 0).astype(np.float32)
    valid_max = depth_resized[depth_resized > 0].max() if np.any(depth_resized > 0) else 80.0
    depth_filled = depth_resized.copy()
    depth_filled[depth_filled == 0] = valid_max

    depth_tensor = torch.from_numpy(depth_filled).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    depth_norm, norm_params = normalize_depth_for_ppd(depth_tensor, mask_tensor)

    inpaint_mask = 1.0 - mask_tensor

    rgb_resize = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    rgb_condition = torch.from_numpy(rgb_resize / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
    with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
        refined_norm = inpainter.inpaint(
            rgb_condition=rgb_condition,
            known_depth=depth_norm,
            edge_mask=inpaint_mask,
            num_steps=sampling_steps,
        )

    refined_metric = denormalize_depth_from_ppd(refined_norm, depth_tensor, mask_tensor, norm_params)
    refined_np = refined_metric.squeeze().cpu().numpy()
    refined_full = cv2.resize(refined_np, (W, H), interpolation=cv2.INTER_LINEAR)

    # Save depth npy
    np.save(os.path.join(OUTDIR, f"{out_name}.npy"), refined_full)

    # Save point cloud
    intrinsic = np.array([
        [707.0912 * rW / W, 0.0, 601.8873 * rW / W],
        [0.0, 707.0912 * rH / H, 183.1104 * rH / H],
        [0.0, 0.0, 1.0],
    ])
    rgb_pcd = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    pcd = depth2pcd(refined_np, intrinsic, color=rgb_pcd, ret_pcd=True)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
    pcd = pcd.select_by_index(ind)
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32)
    )
    o3d.io.write_point_cloud(os.path.join(OUTDIR, f"{out_name}.ply"), pcd)
    print(f"Saved: {out_name}.ply, {out_name}.npy")

    # Save depth vis (side-by-side)
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    d = (refined_full - refined_full.min()) / (refined_full.max() - refined_full.min() + 1e-8) * 255
    d_color = (cmap(d / 255.0)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    gap = np.ones((H, 50, 3), dtype=np.uint8) * 255
    combined = cv2.hconcat([image, gap, d_color])
    cv2.imwrite(os.path.join(OUTDIR, f"{out_name}.png"), combined)


if __name__ == "__main__":
    set_seed(666)
    os.makedirs(OUTDIR, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    semantics_model = 'DA2'
    sampling_steps = 10

    semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
    model_pth = 'checkpoints/ppd.pth'

    print("Loading models...")
    moge = MoGeModel.from_pretrained("checkpoints/moge2.pt").to(DEVICE).eval()
    model = PixelPerfectDepth(
        semantics_model=semantics_model,
        semantics_pth=semantics_pth,
        sampling_steps=sampling_steps,
    )
    model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
    model = model.to(DEVICE).eval()

    schedule = LinearSchedule(T=1000)
    timesteps = Timesteps(T=1000, steps=sampling_steps, device=DEVICE)
    sampler = EulerSampler(schedule, timesteps, 'velocity')
    inpainter = LanPaintInpainter(
        schedule=schedule, sampler=sampler,
        dit_model=model.dit, sem_encoder=model.sem_encoder,
        device=DEVICE, n_steps=5, step_size=0.2,
        lambda_big=16.0, friction=15.0,
    )

    image = cv2.imread(RGB_PATH)
    H, W = image.shape[:2]
    kitti_depth = read_depth_png(DEPTH_PATH)

    # === Step 1: PPD + MoGe metric depth ===
    print("\n[Step 1] PPD + MoGe metric depth prediction...")
    resize_image = resize_keep_aspect(image)
    rH, rW = resize_image.shape[:2]

    depth_ppd, _ = model.infer_image(image)
    depth_ppd_np = depth_ppd.squeeze().cpu().numpy()

    moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    moge_tensor = torch.tensor(moge_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)
    moge_depth, moge_mask, intrinsic_moge = moge.infer(moge_tensor)
    moge_depth[~moge_mask] = moge_depth[moge_mask].max()

    # Align PPD relative depth to MoGe metric scale
    depth_metric_ppd = recover_metric_depth_ransac(depth_ppd_np, moge_depth, moge_mask)
    # Resize to original resolution
    depth_metric_ppd_full = cv2.resize(depth_metric_ppd, (W, H), interpolation=cv2.INTER_LINEAR)

    np.save(os.path.join(OUTDIR, "depth_metric_ppd.npy"), depth_metric_ppd_full)
    save_gt_point_cloud(
        depth_metric_ppd_full, cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        os.path.join(OUTDIR, "depth_metric_ppd.ply"),
    )
    print(f"Saved: depth_metric_ppd.npy, depth_metric_ppd.ply")

    # === Step 2: Mask PPD depth with KITTI GT sparse mask ===
    print("\n[Step 2] Masking PPD depth with KITTI GT mask...")
    kitti_mask = kitti_depth > 0
    depth_metric_ppd_masked = np.where(kitti_mask, depth_metric_ppd_full, 0.0)

    masked_png_path = os.path.join(OUTDIR, "depth_metric_ppd_masked.png")
    save_depth_as_kitti_png(depth_metric_ppd_masked, masked_png_path)
    np.save(os.path.join(OUTDIR, "depth_metric_ppd_masked.npy"), depth_metric_ppd_masked)
    print(f"Saved: depth_metric_ppd_masked.png, depth_metric_ppd_masked.npy")

    # Sanity check: compare distributions
    gt_valid = kitti_depth[kitti_mask]
    ppd_masked_valid = depth_metric_ppd_masked[kitti_mask]
    print(f"  KITTI GT  — mean: {gt_valid.mean():.2f}, std: {gt_valid.std():.2f}, "
          f"range: [{gt_valid.min():.2f}, {gt_valid.max():.2f}]")
    print(f"  PPD masked — mean: {ppd_masked_valid.mean():.2f}, std: {ppd_masked_valid.std():.2f}, "
          f"range: [{ppd_masked_valid.min():.2f}, {ppd_masked_valid.max():.2f}]")

    # === Step 3: Inpaint with PPD-masked depth (PPD distribution) ===
    print("\n[Step 3] Inpainting with PPD-masked depth...")
    run_inpaint(RGB_PATH, masked_png_path, "inpaint_ppd_masked",
                model, inpainter, DEVICE, sampling_steps)

    # === Also inpaint with original KITTI GT for comparison ===
    print("\n[Baseline] Inpainting with original KITTI GT depth...")
    run_inpaint(RGB_PATH, DEPTH_PATH, "inpaint_kitti_gt",
                model, inpainter, DEVICE, sampling_steps)

    print(f"\nDone! All results in {OUTDIR}/")
