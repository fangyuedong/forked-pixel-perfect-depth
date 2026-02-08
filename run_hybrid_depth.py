"""
Hybrid Depth Estimation Pipeline: MoGe-2 + Pixel-Perfect Depth + RePaint

This script implements a two-stage depth estimation pipeline:
1. Stage 1: MoGe-2 provides metric absolute depth
2. Stage 2: Pixel-Perfect Depth with RePaint-style edge-aware refinement

The pipeline:
- Uses MoGe-2 for initial metric depth estimation
- Detects edges in the depth map
- Refines edge regions using PPD with RePaint-style inpainting
- Preserves non-edge regions from the original MoGe-2 depth

Usage:
    python run_hybrid_depth.py --img_path assets/examples/images

    # With full output (PNG + NPY + PCD)
    python run_hybrid_depth.py --img_path assets/examples/images --save_npy --save_pcd

    # With edge mask debugging
    python run_hybrid_depth.py --img_path assets/examples/images --save_edge_mask

Reference:
    - Pixel-Perfect Depth: https://github.com/fengxyxl/Pixel-Perfect-Depth
    - RePaint: https://arxiv.org/abs/2201.09865
"""

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import open3d as o3d

from ppd.utils.set_seed import set_seed
from ppd.utils.depth2pcd import depth2pcd
from ppd.moge.model.v2 import MoGeModel
from ppd.models.ppd import PixelPerfectDepth
from ppd.models.edge_detector import EdgeDetector
from ppd.models.repaint_inpainter import RePaintInpainter
from ppd.utils.depth_normalization import normalize_depth_for_ppd, denormalize_depth_from_ppd
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.transform import resize_keep_aspect, image2tensor
from ppd.utils.utils import has_native_bf16


if __name__ == '__main__':
    set_seed(666)

    parser = argparse.ArgumentParser(
        description='Hybrid Depth: MoGe-2 + PPD + RePaint'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        default='assets/examples/images',
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='hybrid_depth_output',
        help='Output directory'
    )
    parser.add_argument(
        '--semantics_model',
        type=str,
        default='DA2',
        choices=['MoGe2', 'DA2'],
        help='Semantic model backend (DA2 or MoGe2)'
    )
    parser.add_argument(
        '--sampling_steps',
        type=int,
        default=10,
        help='Number of diffusion sampling steps'
    )
    parser.add_argument(
        '--resampling_steps',
        type=int,
        default=1,
        help='RePaint resampling steps U (default: 1, no resampling)'
    )
    parser.add_argument(
        '--edge_threshold',
        type=float,
        default=0.1,
        help='Edge detection threshold (0-1 range)'
    )
    parser.add_argument(
        '--edge_dilation',
        type=int,
        default=3,
        help='Edge dilation kernel size in pixels'
    )
    parser.add_argument(
        '--no_edge_mask',
        action='store_true',
        help='Disable edge detection and use full inpainting (for debugging)'
    )
    parser.add_argument(
        '--apply_filter',
        action='store_false',
        default=True,
        help='Apply statistical outlier filter to point cloud'
    )
    parser.add_argument(
        '--pred_only',
        action='store_true',
        help='Only save predicted depth (no side-by-side comparison)'
    )
    parser.add_argument(
        '--save_npy',
        action='store_true',
        help='Save raw depth as .npy files'
    )
    parser.add_argument(
        '--save_pcd',
        action='store_true',
        help='Save point clouds as .ply files'
    )
    parser.add_argument(
        '--save_edge_mask',
        action='store_true',
        help='Save edge masks for debugging'
    )

    args = parser.parse_args()

    # Device setup
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    # Select checkpoints based on semantic model
    if args.semantics_model == 'MoGe2':
        semantics_pth = 'checkpoints/moge2.pt'
        model_pth = 'checkpoints/ppd_moge.pth'
    else:
        semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
        model_pth = 'checkpoints/ppd.pth'

    print(f'Loading models on {DEVICE}...')
    print(f'  Semantic model: {args.semantics_model}')
    print(f'  Sampling steps: {args.sampling_steps}')
    print(f'  Resampling steps: {args.resampling_steps}')

    # Load MoGe-2 for metric depth
    moge = MoGeModel.from_pretrained("checkpoints/moge2.pt").to(DEVICE).eval()

    # Load Pixel-Perfect Depth model
    model = PixelPerfectDepth(
        semantics_model=args.semantics_model,
        semantics_pth=semantics_pth,
        sampling_steps=args.sampling_steps
    )
    model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
    model = model.to(DEVICE).eval()

    # Initialize diffusion components
    schedule = LinearSchedule(T=1000)
    timesteps = Timesteps(T=1000, steps=args.sampling_steps, device=DEVICE)
    sampler = EulerSampler(schedule, timesteps, 'velocity')

    # Initialize RePaint inpainter
    inpainter = RePaintInpainter(
        schedule=schedule,
        sampler=sampler,
        dit_model=model.dit,
        sem_encoder=model.sem_encoder,
        device=DEVICE,
        resampling_steps=args.resampling_steps
    )

    # Initialize edge detector
    edge_detector = EdgeDetector(
        threshold=args.edge_threshold,
        dilation_kernel_size=args.edge_dilation
    )

    # Collect input filenames
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = sorted(glob.glob(os.path.join(args.img_path, '**/*'), recursive=True))

    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral')

    print(f'Processing {len(filenames)} images...')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        image = cv2.imread(filename)
        H, W = image.shape[:2]

        # Resize the image to match the training resolution area while keeping the original aspect ratio
        resize_image = resize_keep_aspect(image)

        # ===== Stage 1: MoGe-2 Metric Depth =====
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image_tensor = torch.tensor(
            moge_image / 255,
            dtype=torch.float32,
            device=DEVICE
        ).permute(2, 0, 1)

        moge_depth, mask, intrinsic = moge.infer(moge_image_tensor)
        moge_depth[~mask] = moge_depth[mask].max()

        # Normalize MoGe-2 depth to PPD format
        moge_depth_tensor = torch.from_numpy(moge_depth).unsqueeze(0).unsqueeze(0).to(DEVICE)
        moge_mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)
        moge_depth_norm, norm_params = normalize_depth_for_ppd(moge_depth_tensor, moge_mask_tensor)

        # ===== Edge Detection =====
        if args.no_edge_mask:
            # 禁用边缘检测：所有区域都需要重新生成
            edge_mask = torch.ones_like(moge_depth_norm)
        else:
            # 正常边缘检测
            edge_mask = edge_detector.detect_edges(moge_depth_norm)

        # ===== RGB Condition for PPD =====
        rgb_condition = moge_image_tensor.unsqueeze(0)  # (1, 3, H, W) in [-0.5, 0.5]

        # ===== Stage 2: PPD + RePaint Refinement =====
        autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
        with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
            refined_depth_norm = inpainter.inpaint(
                rgb_condition=rgb_condition,
                known_depth=moge_depth_norm,
                edge_mask=edge_mask,
                num_steps=args.sampling_steps
            )
            print(refined_depth_norm.shape, refined_depth_norm.mean().item())

        # Convert back to metric scale
        refined_depth_metric = denormalize_depth_from_ppd(
            refined_depth_norm,
            moge_depth_tensor,
            moge_mask_tensor,
            norm_params
        )
        refined_depth_metric_np = refined_depth_metric.squeeze().cpu().numpy()

        # Resize depth back to original image size for visualization
        refined_depth_np = cv2.resize(refined_depth_metric_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # ===== Save Visualization =====
        depth_vis = (
            (refined_depth_np - refined_depth_np.min()) /
            (refined_depth_np.max() - refined_depth_np.min()) * 255
        )
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(
                os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'),
                depth_vis
            )
        else:
            split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([image, split_region, depth_vis])
            cv2.imwrite(
                os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'),
                combined
            )

        # ===== Save Edge Mask (optional) =====
        if args.save_edge_mask:
            edge_vis = (edge_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            # Resize edge mask back to original image size
            edge_vis = cv2.resize(edge_vis, (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(args.outdir, 'edge_' + os.path.splitext(os.path.basename(filename))[0] + '.png'),
                edge_vis
            )

        # ===== Save NPY =====
        if args.save_npy:
            npy_dir = os.path.join(args.outdir, 'depth_npy')
            os.makedirs(npy_dir, exist_ok=True)
            np.save(
                os.path.join(npy_dir, os.path.splitext(os.path.basename(filename))[0] + '.npy'),
                refined_depth_np
            )

        # ===== Save Point Cloud =====
        if args.save_pcd:
            pcd_dir = os.path.join(args.outdir, 'depth_pcd')
            os.makedirs(pcd_dir, exist_ok=True)

            # Scale intrinsic to match the resized image dimensions
            resize_H, resize_W = resize_image.shape[:2]
            intrinsic_scaled = intrinsic.copy()
            intrinsic_scaled[0, 0] *= resize_W
            intrinsic_scaled[1, 1] *= resize_H
            intrinsic_scaled[0, 2] *= resize_W
            intrinsic_scaled[1, 2] *= resize_H

            pcd = depth2pcd(
                refined_depth_metric_np,  # Use depth at resized resolution
                intrinsic_scaled,
                color=cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB),  # Use resized image
                input_mask=mask,
                ret_pcd=True
            )
            if args.apply_filter:
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
                pcd = pcd.select_by_index(ind)
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32)
            )
            o3d.io.write_point_cloud(
                os.path.join(pcd_dir, os.path.splitext(os.path.basename(filename))[0] + '.ply'),
                pcd
            )

    print(f'Done! Results saved to {args.outdir}')
