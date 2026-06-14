"""
KITTI depth completion evaluation using the refine_step7 pipeline
(PPD + multi-scale filter + DepthInpaintPipeline).

Metrics: RMSE, MAE, REL, δ₁
Outputs: GT/pred point clouds, overlay depth maps, per-sample metrics summary.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch

from kitti_utils import read_depth_png, save_gt_point_cloud, visualize_depth_overlay
from ppd.models.depth_inpainter import DepthInpaintPipeline
from ppd.utils.depth2pcd import depth2pcd
from ppd.utils.set_seed import set_seed
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# ViGeo imports
# sys.path.insert(0, "/home/fanguedong/Code/ViGeo")
from utils import multi_scale_filter_depth
from utils3d.torch.maps import depth_map_to_point_map


# ── Path / data utilities ────────────────────────────────────────────────────
def load_kitti_intrinsic(path):
    with open(path) as f:
        vals = [float(x) for x in f.read().split()]
    return np.array(vals, dtype=np.float32).reshape(3, 3)


def derive_paths(image_path: Path, data_root: Path):
    """Derive GT/velodyne/intrinsic paths from the rgb image path.

    Filename pattern:
        <prefix>_image_<frame>_image_<cam>.png
        <prefix>_groundtruth_depth_<frame>_image_<cam>.png
        <prefix>_velodyne_raw_<frame>_image_<cam>.png
        <prefix>_image_<frame>_image_<cam>.txt
    """
    stem = image_path.stem
    m = re.match(r"^(.*)_image_(\d+)_image_(\d+)$", stem)
    if not m:
        raise ValueError(f"Unexpected KITTI filename pattern: {stem}")
    base, frame_idx, cam = m.group(1), m.group(2), m.group(3)
    gt = data_root / "groundtruth_depth" / f"{base}_groundtruth_depth_{frame_idx}_image_{cam}.png"
    velo = data_root / "velodyne_raw" / f"{base}_velodyne_raw_{frame_idx}_image_{cam}.png"
    intr = data_root / "intrinsics" / f"{base}_image_{frame_idx}_image_{cam}.txt"
    return gt, velo, intr


def list_samples(data_root: Path, limit=None, pattern=None):
    image_dir = data_root / "image"
    samples = sorted(image_dir.glob("*.png"))
    if pattern:
        samples = [s for s in samples if pattern in s.stem]
    if limit:
        samples = samples[:limit]
    return samples


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    p = pred[mask]
    g = gt[mask]
    diff = p - g
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    rel = float(np.mean(np.abs(diff) / g))
    ratio = np.maximum(g / p, p / g)
    d1 = float(np.mean(ratio < 1.25))
    return {"RMSE": rmse, "MAE": mae, "REL": rel, "delta1": d1}


# ── RANSAC ───────────────────────────────────────────────────────────────────
def fit_ransac(pred, gt_log, seed=42):
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        RANSACRegressor(max_trials=1000, random_state=seed),
    )
    model.fit(pred[:, None], gt_log[:, None])
    a = model.named_steps["ransacregressor"].estimator_.coef_.item()
    b = model.named_steps["ransacregressor"].estimator_.intercept_.item()
    return a, b


# ── Visualization ────────────────────────────────────────────────────────────
def save_pred_pcd(depth, rgb, intrinsic, path):
    """Save dense predicted depth as colored point cloud."""
    pcd = depth2pcd(depth, intrinsic, color=rgb, ret_pcd=True)
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) * np.array([1, -1, -1])
    )
    o3d.io.write_point_cloud(str(path), pcd)


def side_by_side(image, depth, path):
    cmap = matplotlib.colormaps.get_cmap("Spectral")
    H = image.shape[0]
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    d_color = (cmap(d / 255.0)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    gap = np.ones((H, 50, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(path), cv2.hconcat([image, gap, d_color]))


# ── Core refine logic (adapted from refine_step7.py) ─────────────────────────
def refine_one(pipeline, image_bgr, sparse_depth, intrinsic, device):
    """Run the refine_step7 pipeline on a single sample.

    Returns dense_depth at original resolution (H, W) in meters.
    """
    H, W = image_bgr.shape[:2]

    # 1. PPD relative depth (reuse pipeline's internal model)
    depth_ppd, resized_bgr = pipeline.model.infer_image(image_bgr)
    ppd_rel = depth_ppd.squeeze().cpu().numpy()
    rH, rW = resized_bgr.shape[:2]

    sp_resized = cv2.resize(sparse_depth, (rW, rH), interpolation=cv2.INTER_NEAREST)
    valid = sp_resized > 0

    # 2. RANSAC → PPD metric depth
    ppd_valid = ppd_rel[valid].astype(np.float32)
    gt_log_valid = np.log(sp_resized[valid].astype(np.float32) + 1.0)
    a, b = fit_ransac(ppd_valid, gt_log_valid)
    ppd_metric = np.exp(a * ppd_rel.astype(np.float32) + b) - 1.0
    ppd_metric = np.clip(ppd_metric, 0, 200)

    # 3. Multi-scale filter sparse depth
    int_norm = intrinsic.copy()
    int_norm[0, :] /= rW
    int_norm[1, :] /= rH
    int_norm_t = torch.from_numpy(int_norm).float().to(device).unsqueeze(0)

    sparse_t = torch.from_numpy(sp_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    mono_t = torch.from_numpy(ppd_metric.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(valid).unsqueeze(0).to(device)

    points_gt = depth_map_to_point_map(sparse_t.squeeze(1), intrinsics=int_norm_t)
    points_pred = depth_map_to_point_map(mono_t.squeeze(1), intrinsics=int_norm_t)

    focal = 1.0 / (1.0 / int_norm[0, 0] ** 2 + 1.0 / int_norm[1, 1] ** 2) ** 0.5
    focal_t = torch.tensor(focal, dtype=torch.float32, device=device).unsqueeze(0)

    filtered_mask = multi_scale_filter_depth(
        points_pred, points_gt,
        mask_t & (sparse_t.squeeze(1) > 0.0001),
        focal=focal_t,
    )

    # Map filtered depth back to original resolution
    # Resize the depth values directly (not mask × original) to avoid position misalignment
    filtered_ppd_np = (sparse_t * filtered_mask.unsqueeze(1).float()).squeeze().cpu().numpy()
    sparse_filtered = cv2.resize(filtered_ppd_np, (W, H), interpolation=cv2.INTER_NEAREST)

    # 4. DepthInpaintPipeline
    dense_depth, _, _, _ = pipeline(image_bgr, sparse_filtered, intrinsic=intrinsic)
    return dense_depth


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KITTI depth completion eval (refine_step7 pipeline)")
    parser.add_argument(
        "--data_root",
        default="/home/fanguedong/Downloads/data_depth_selection/depth_selection/val_selection_cropped",
    )
    parser.add_argument("--output_dir", default="eval_kitti_output")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples (mini test = 10)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantics_model", default="MoGe2", choices=["MoGe2", "DA2"])
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--no_visuals", action="store_true", help="Skip saving point clouds / overlays")
    parser.add_argument("--pattern", default=None, help="Substring filter on filenames")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    samples = list_samples(data_root, limit=args.limit, pattern=args.pattern)
    if not samples:
        raise RuntimeError(f"No samples found under {data_root / 'image'}")
    print(f"Found {len(samples)} samples to evaluate")

    # Create pipeline once (loads PPD model once)
    print("Loading DepthInpaintPipeline …")
    pipeline = DepthInpaintPipeline(
        device=device, semantics_model=args.semantics_model, sampling_steps=args.sampling_steps
    )

    accum = {"RMSE": [], "MAE": [], "REL": [], "delta1": []}

    for idx, image_path in enumerate(samples):
        set_seed(666)
        gt_path, velo_path, intr_path = derive_paths(image_path, data_root)
        for p in (gt_path, velo_path, intr_path):
            if not p.exists():
                raise FileNotFoundError(p)

        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sparse_depth = read_depth_png(str(velo_path))
        gt_depth = read_depth_png(str(gt_path)).astype(np.float32)
        intrinsic = load_kitti_intrinsic(str(intr_path))

        stem = image_path.stem
        print(f"\n[{idx + 1}/{len(samples)}] {stem}")
        refined = refine_one(pipeline, image_bgr, sparse_depth, intrinsic, device)

        gt_mask = gt_depth > 0
        metrics = compute_metrics(refined, gt_depth, gt_mask)
        for k in accum:
            accum[k].append(metrics[k])
        print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
              f"REL={metrics['REL']:.4f}  delta1={metrics['delta1']:.4f}")

        if not args.no_visuals:
            sample_dir = out_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            # GT point cloud (sparse)
            save_gt_point_cloud(gt_depth, image_rgb, str(sample_dir / f"{stem}_gt.ply"),
                                intrinsic=intrinsic)
            # Predicted dense point cloud
            save_pred_pcd(refined, image_rgb, intrinsic, sample_dir / f"{stem}_pred.ply")
            # Overlay depth image
            overlay = visualize_depth_overlay(image_rgb, refined, alpha=0.6, point_size=1)
            cv2.imwrite(str(sample_dir / f"{stem}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            np.save(sample_dir / f"{stem}_pred_depth.npy", refined)

    print("\n=== Mean over {} samples ===".format(len(samples)))
    for k in accum:
        print(f"  {k:<7} = {float(np.mean(accum[k])):.4f}")

    summary_path = out_dir / "metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"# KITTI depth completion eval (N={len(samples)})\n")
        f.write("sample\tRMSE\tMAE\tREL\tdelta1\n")
        for image_path, vals in zip(samples, zip(accum["RMSE"], accum["MAE"], accum["REL"], accum["delta1"])):
            f.write(f"{image_path.stem}\t" + "\t".join(f"{v:.6f}" for v in vals) + "\n")
        f.write("\nmean\t" + "\t".join(f"{float(np.mean(accum[k])):.6f}" for k in accum) + "\n")
    print(f"Saved per-sample metrics to {summary_path}")


if __name__ == "__main__":
    main()
