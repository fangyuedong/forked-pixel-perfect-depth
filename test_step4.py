"""
Step 4: Use RANSAC to map GT depth directly into PPD's relative space.

Pipeline:
  1. Run PPD → relative depth (ppd_rel, ~[0,1])
  2. RANSAC: log(gt+1) = a * ppd_rel + b  (on valid GT pixels)
  3. Convert GT to PPD space: gt_ppd = (log(gt+1) - b) / a - 0.5
  4. Inpaint with gt_ppd as known depth → ppd_inpaint_rel
  5. RANSAC: log(gt+1) = a2 * (ppd_inpaint_rel+0.5) + b2 → metric depth
  6. Save depth map + point cloud

All outputs go to step4/
"""

import os

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from kitti_utils import read_depth_png, save_gt_point_cloud
from ppd.models.ppd import PixelPerfectDepth
from ppd.models.lanpaint_inpainter import LanPaintInpainter
from ppd.utils.depth2pcd import depth2pcd
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.set_seed import set_seed
from ppd.utils.transform import resize_keep_aspect
from ppd.utils.utils import has_native_bf16

OUTDIR = "step4"
RGB_PATH = "/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/image_02/data/0000000015.png"
DEPTH_PATH = "/home/fanguedong/Dataset/kitti_depth/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02/0000000015.png"


def fit_ransac(pred, gt_log):
    """Fit log(gt+1) = a * pred + b, return (a, b)."""
    ransac_model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        RANSACRegressor(max_trials=1000),
    )
    ransac_model.fit(pred[:, None], gt_log[:, None])
    a = ransac_model.named_steps['ransacregressor'].estimator_.coef_.item()
    b = ransac_model.named_steps['ransacregressor'].estimator_.intercept_.item()
    return a, b


def save_vis(depth, image, name):
    """Save side-by-side depth visualization."""
    H, W = image.shape[:2]
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    d_color = (cmap(d / 255.0)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    gap = np.ones((H, 50, 3), dtype=np.uint8) * 255
    combined = cv2.hconcat([image, gap, d_color])
    cv2.imwrite(os.path.join(OUTDIR, f"{name}.png"), combined)


def save_pcd(depth, rgb, name, intrinsic):
    """Save point cloud with outlier filtering."""
    pcd = depth2pcd(depth, intrinsic, color=rgb, ret_pcd=True)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
    pcd = pcd.select_by_index(ind)
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32)
    )
    path = os.path.join(OUTDIR, f"{name}.ply")
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved: {path}")


if __name__ == "__main__":
    set_seed(666)
    os.makedirs(OUTDIR, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampling_steps = 10
    semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
    model_pth = 'checkpoints/ppd.pth'

    print("Loading models...")
    model = PixelPerfectDepth(
        semantics_model='DA2', semantics_pth=semantics_pth,
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

    resize_image = resize_keep_aspect(image)
    rH, rW = resize_image.shape[:2]
    rgb_resize = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

    # === Step 1: PPD relative depth ===
    print("\n[Step 1] PPD relative depth...")
    depth_ppd, _ = model.infer_image(image)  # (1,1,rH,rW), ~[0,1]
    ppd_rel = depth_ppd.squeeze().cpu().numpy()

    # Resize GT to match PPD resolution
    kitti_resized = cv2.resize(kitti_depth, (rW, rH), interpolation=cv2.INTER_NEAREST)
    valid = kitti_resized > 0

    # === Step 2: RANSAC fit ===
    print("\n[Step 2] RANSAC: log(gt+1) = a * ppd_rel + b ...")
    ppd_valid = ppd_rel[valid].astype(np.float32)
    gt_log_valid = np.log(kitti_resized[valid].astype(np.float32) + 1.0)
    a, b = fit_ransac(ppd_valid, gt_log_valid)
    print(f"  RANSAC: a={a:.4f}, b={b:.4f}")

    # === Step 3: Convert GT to PPD relative space ===
    print("\n[Step 3] Converting GT to PPD relative space...")
    gt_log = np.log(kitti_resized.astype(np.float32) + 1.0)
    gt_in_ppd = np.where(valid, (gt_log - b) / a - 0.5, 0.0)

    print(f"  gt_in_ppd range: [{gt_in_ppd[valid].min():.4f}, {gt_in_ppd[valid].max():.4f}]")
    print(f"  ppd_rel range:   [{ppd_rel.min():.4f}, {ppd_rel.max():.4f}]")

    # === Step 4: Inpaint ===
    print("\n[Step 4] Inpainting...")
    known_depth = torch.from_numpy(gt_in_ppd).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    mask_tensor = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    inpaint_mask = 1.0 - mask_tensor  # 1 = unknown, 0 = known

    rgb_condition = torch.from_numpy(rgb_resize / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
    with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
        refined = inpainter.inpaint(
            rgb_condition=rgb_condition,
            known_depth=known_depth,
            edge_mask=inpaint_mask,
            num_steps=sampling_steps,
        )

    ppd_inpaint_rel = refined.squeeze().cpu().numpy()  # [-0.5, 0.5]

    # === Step 5: RANSAC back to metric ===
    print("\n[Step 5] RANSAC: inpaint → metric...")
    inpaint_01 = ppd_inpaint_rel + 0.5  # shift to [0, 1]
    inpaint_valid = inpaint_01[valid].astype(np.float32)
    a2, b2 = fit_ransac(inpaint_valid, gt_log_valid)
    print(f"  RANSAC: a2={a2:.4f}, b2={b2:.4f}")

    inpaint_metric_log = a2 * inpaint_01 + b2
    inpaint_metric = np.exp(inpaint_metric_log) - 1.0
    inpaint_metric = np.clip(inpaint_metric, 0, 200)

    # Resize to original resolution
    inpaint_full = cv2.resize(inpaint_metric, (W, H), interpolation=cv2.INTER_LINEAR)

    # Intrinsic (scaled to resized image)
    intrinsic = np.array([
        [707.0912 * rW / W, 0.0, 601.8873 * rW / W],
        [0.0, 707.0912 * rH / H, 183.1104 * rH / H],
        [0.0, 0.0, 1.0],
    ])

    # === Step 6: Save results ===
    print("\n[Step 6] Saving results...")
    np.save(os.path.join(OUTDIR, "inpaint_metric.npy"), inpaint_full)
    save_vis(inpaint_full, image, "inpaint_metric")
    save_pcd(inpaint_metric, rgb_resize, "inpaint_metric", intrinsic)

    # Also save PPD raw prediction for comparison
    ppd_full = cv2.resize(ppd_rel, (W, H), interpolation=cv2.INTER_LINEAR)
    save_vis(ppd_full, image, "ppd_raw")
    np.save(os.path.join(OUTDIR, "ppd_raw.npy"), ppd_full)

    # Compare at valid pixels
    print(f"\n  Comparison at valid GT pixels:")
    print(f"  KITTI GT:          mean={kitti_depth[kitti_depth>0].mean():.2f}")
    print(f"  Inpaint metric:    mean={inpaint_full[kitti_depth>0].mean():.2f}")

    print(f"\nDone! Results in {OUTDIR}/")
