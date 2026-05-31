"""
KITTI Depth Inpainting: Sparse GT depth → Dense depth via PPD + LanPaint.

Usage:
    python run_kitti_inpaint.py \
        --rgb <rgb_path> --depth <depth_png_path> \
        --save_pcd --save_overlay

    # Adjust FLD parameters
    python run_kitti_inpaint.py \
        --rgb <rgb_path> --depth <depth_png_path> \
        --fld_steps 10 --fld_lambda 32.0
"""

import argparse
import os

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch

from kitti_utils import read_depth_png, visualize_depth_overlay
from ppd.models.lanpaint_inpainter import LanPaintInpainter
from ppd.models.ppd import PixelPerfectDepth
from ppd.utils.depth2pcd import depth2pcd
from ppd.utils.depth_normalization import normalize_depth_for_ppd, denormalize_depth_from_ppd
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.set_seed import set_seed
from ppd.utils.transform import resize_keep_aspect
from ppd.utils.utils import has_native_bf16


def main():
    set_seed(666)

    parser = argparse.ArgumentParser(description='KITTI Depth Inpainting: PPD + LanPaint')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to 16-bit KITTI depth PNG')
    parser.add_argument('--outdir', default='kitti_inpaint_output', help='Output directory')
    parser.add_argument('--semantics_model', default='DA2', choices=['MoGe2', 'DA2'])
    parser.add_argument('--sampling_steps', type=int, default=10)
    parser.add_argument('--fld_steps', type=int, default=5)
    parser.add_argument('--fld_step_size', type=float, default=0.2)
    parser.add_argument('--fld_lambda', type=float, default=16.0)
    parser.add_argument('--fld_friction', type=float, default=15.0)
    parser.add_argument('--save_pcd', action='store_true', help='Save point cloud')
    parser.add_argument('--save_overlay', action='store_true', help='Save depth overlay on RGB')
    parser.add_argument('--save_npy', action='store_true', help='Save raw depth as .npy')
    args = parser.parse_args()

    # Device
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    # Checkpoints
    if args.semantics_model == 'MoGe2':
        semantics_pth = 'checkpoints/moge2.pt'
        model_pth = 'checkpoints/ppd_moge.pth'
    else:
        semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
        model_pth = 'checkpoints/ppd.pth'

    print(f'Loading models on {DEVICE}...')
    model = PixelPerfectDepth(
        semantics_model=args.semantics_model,
        semantics_pth=semantics_pth,
        sampling_steps=args.sampling_steps,
    )
    model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
    model = model.to(DEVICE).eval()

    schedule = LinearSchedule(T=1000)
    timesteps = Timesteps(T=1000, steps=args.sampling_steps, device=DEVICE)
    sampler = EulerSampler(schedule, timesteps, 'velocity')

    inpainter = LanPaintInpainter(
        schedule=schedule,
        sampler=sampler,
        dit_model=model.dit,
        sem_encoder=model.sem_encoder,
        device=DEVICE,
        n_steps=args.fld_steps,
        step_size=args.fld_step_size,
        lambda_big=args.fld_lambda,
        friction=args.fld_friction,
    )

    # Read inputs
    image = cv2.imread(args.rgb)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB: {args.rgb}")
    H, W = image.shape[:2]

    depth_meters = read_depth_png(args.depth)
    if depth_meters.shape != (H, W):
        raise ValueError(f"Depth shape {depth_meters.shape} != image shape {(H, W)}")

    # Resize for PPD
    resize_image = resize_keep_aspect(image)
    rH, rW = resize_image.shape[:2]

    # Resize sparse depth to match PPD input resolution
    depth_resized = cv2.resize(depth_meters, (rW, rH), interpolation=cv2.INTER_NEAREST)

    # Build mask (valid GT pixels)
    mask = (depth_resized > 0).astype(np.float32)

    # For zero-depth regions, fill with a reasonable value so normalization works
    # Use the max of valid depth as placeholder (will be overwritten by inpainting)
    valid_max = depth_resized[depth_resized > 0].max() if np.any(depth_resized > 0) else 80.0
    depth_filled = depth_resized.copy()
    depth_filled[depth_filled == 0] = valid_max

    # Normalize to PPD format
    depth_tensor = torch.from_numpy(depth_filled).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    depth_norm, norm_params = normalize_depth_for_ppd(depth_tensor, mask_tensor)

    # Inpainting mask: 1 = unknown (zero depth), 0 = known (sparse GT)
    # LanPaintInpainter: edge_mask=1 means refine, edge_mask=0 means preserve
    inpaint_mask = (1.0 - mask_tensor)

    # RGB condition
    rgb_resize = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    rgb_condition = torch.from_numpy(rgb_resize / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # Run inpainting
    autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
    with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
        refined_norm = inpainter.inpaint(
            rgb_condition=rgb_condition,
            known_depth=depth_norm,
            edge_mask=inpaint_mask,
            num_steps=args.sampling_steps,
        )

    # Denormalize back to metric
    refined_metric = denormalize_depth_from_ppd(refined_norm, depth_tensor, mask_tensor, norm_params)
    print(f'compare {depth_tensor[mask_tensor.bool()].mean()} vs {refined_metric[mask_tensor.bool()].mean()}')
    refined_np = refined_metric.squeeze().cpu().numpy()

    # Resize back to original resolution
    refined_full = cv2.resize(refined_np, (W, H), interpolation=cv2.INTER_LINEAR)

    # Output
    os.makedirs(args.outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.rgb))[0]
    cmap = matplotlib.colormaps.get_cmap('Spectral')

    # Depth visualization
    depth_vis = (refined_full - refined_full.min()) / (refined_full.max() - refined_full.min() + 1e-8) * 255
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis_color = (cmap(depth_vis / 255.0)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    gap = np.ones((H, 50, 3), dtype=np.uint8) * 255
    combined = cv2.hconcat([image, gap, depth_vis_color])
    cv2.imwrite(os.path.join(args.outdir, f'{basename}.png'), combined)
    print(f'Saved comparison: {args.outdir}/{basename}.png')

    if args.save_npy:
        np.save(os.path.join(args.outdir, f'{basename}.npy'), refined_full)
        print(f'Saved depth npy: {args.outdir}/{basename}.npy')

    if args.save_overlay:
        rgb_for_overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = visualize_depth_overlay(rgb_for_overlay, refined_full, alpha=0.6, point_size=1)
        cv2.imwrite(os.path.join(args.outdir, f'{basename}_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f'Saved overlay: {args.outdir}/{basename}_overlay.png')

    if args.save_pcd:
        # Use a simple pinhole intrinsic estimate for point cloud
        # focal = rW * 0.9  # approximate focal length
        # intrinsic = np.array([
        #     [focal, 0, rW / 2],
        #     [0, focal, rH / 2],
        #     [0, 0, 1],
        # ], dtype=np.float64)
        intrinsic = np.array([
            [707.0912*rW/W, 0.0, 601.8873*rW/W],
            [0.0, 707.0912*rH/H, 183.1104*rH/H],
            [0.0, 0.0, 1.0],
        ])
        rgb_pcd = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        pcd = depth2pcd(refined_np, intrinsic, color=rgb_pcd, ret_pcd=True)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        pcd = pcd.select_by_index(ind)
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32)
        )
        ply_path = os.path.join(args.outdir, f'{basename}.ply')
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f'Saved point cloud: {ply_path}')


if __name__ == '__main__':
    main()
