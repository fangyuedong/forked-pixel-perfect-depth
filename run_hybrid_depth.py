import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from ppd.utils.set_seed import set_seed
from ppd.utils.align_depth_func import (
    recover_metric_depth_ransac,
    detect_depth_edges,
    detect_canny_edges
)
from ppd.moge.model.v2 import MoGeModel
from ppd.models.ppd import PixelPerfectDepth


def main():
    set_seed(666)  # Set random seed

    parser = argparse.ArgumentParser(description='Hybrid Pixel-Perfect Depth with MoGe-2')
    parser.add_argument('--img_path', type=str, default='assets/examples/images',
                        help='Path to input image or directory')
    parser.add_argument('--outdir', type=str, default='hybrid_depth_vis',
                        help='Output directory for depth visualizations')
    parser.add_argument('--semantics_model', type=str, default='MoGe2',
                        choices=['MoGe2', 'DA2'],
                        help='Semantic encoder for PPD')
    parser.add_argument('--sampling_steps', type=int, default=10,
                        help='Number of diffusion sampling steps')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save raw depth predictions as .npy files')
    parser.add_argument('--pred_only', action='store_true',
                        help='Only display/save predicted depth (no input image)')
    parser.add_argument('--edge_detection', type=str, default='gradient',
                        choices=['gradient', 'canny'],
                        help='Edge detection method')
    parser.add_argument('--gradient_threshold', type=float, default=0.05,
                        help='Gradient threshold for edge detection')
    parser.add_argument('--dilation_iter', type=int, default=3,
                        help='Number of dilation iterations to expand edge mask')
    parser.add_argument('--save_edge_mask', action='store_true',
                        help='Save edge mask visualization')
    parser.add_argument('--save_moge', action='store_true',
                        help='Save MoGe-2 depth visualization')

    args = parser.parse_args()

    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    print(f'Using device: {DEVICE}')

    # Set up model paths
    if args.semantics_model == 'MoGe2':
        semantics_pth = 'checkpoints/moge2.pt'
        model_pth = 'checkpoints/ppd_moge.pth'
    else:
        semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
        model_pth = 'checkpoints/ppd.pth'

    # Load MoGe-2 model
    print('Loading MoGe-2 model...')
    moge = MoGeModel.from_pretrained("checkpoints/moge2.pt").to(DEVICE).eval()

    # Load PPD model
    print('Loading PPD model...')
    model = PixelPerfectDepth(
        semantics_model=args.semantics_model,
        semantics_pth=semantics_pth,
        sampling_steps=args.sampling_steps
    )
    model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
    model = model.to(DEVICE).eval()

    # Load input images
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = sorted(filenames)

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Color map for visualization
    cmap = matplotlib.colormaps.get_cmap('Spectral')

    print(f'Processing {len(filenames)} images...')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        # Load and resize image
        image = cv2.imread(filename)
        if image is None:
            print(f'Failed to load {filename}, skipping...')
            continue
        H, W = image.shape[:2]

        # Get PPD resize dimensions
        with torch.no_grad():
            from ppd.utils.transform import resize_keep_aspect
            resize_image = resize_keep_aspect(image)
            resize_H, resize_W = resize_image.shape[:2]

        # Convert for MoGe-2 inference
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

        # Get MoGe-2 depth and mask
        moge_depth, mask, intrinsic = moge.infer(moge_image)
        moge_depth[~mask] = moge_depth[mask].max()

        # Convert to numpy for edge detection
        moge_depth_np = moge_depth.squeeze().cpu().numpy()

        # Detect edges
        if args.edge_detection == 'gradient':
            edge_mask = detect_depth_edges(
                moge_depth_np,
                gradient_threshold=args.gradient_threshold,
                dilation_iter=args.dilation_iter
            )
        else:  # canny
            edge_mask = detect_canny_edges(
                moge_depth_np,
                dilation_iter=args.dilation_iter
            )

        # Apply edge mask to original MoGe-2 depth
        mask_np = mask.squeeze().cpu().numpy()

        # Run hybrid PPD inference
        hybrid_depth_tensor, resize_image = model.infer_hybrid(
            image=image,
            moge_depth=moge_depth_np * mask_np,  # Apply mask to moge depth
            edge_mask=edge_mask,
            sampling_steps=args.sampling_steps
        )
        hybrid_depth = hybrid_depth_tensor.squeeze().cpu().numpy()

        # Resize hybrid depth to original image size
        hybrid_depth_resized = cv2.resize(hybrid_depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize and visualize hybrid depth
        vis_hybrid = (hybrid_depth_resized - hybrid_depth_resized.min()) / \
                     (hybrid_depth_resized.max() - hybrid_depth_resized.min() + 1e-8)
        vis_hybrid = (vis_hybrid * 255).astype(np.uint8)
        vis_hybrid = (cmap(vis_hybrid)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Get base filename
        base_name = os.path.splitext(os.path.basename(filename))[0]

        # Save hybrid depth visualization
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), vis_hybrid)
        else:
            split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([image, split_region, vis_hybrid])
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), combined_result)

        # Save MoGe-2 depth visualization if requested
        if args.save_moge:
            moge_depth_resized = cv2.resize(moge_depth_np, (W, H), interpolation=cv2.INTER_LINEAR)
            vis_moge = (moge_depth_resized - moge_depth_resized.min()) / \
                       (moge_depth_resized.max() - moge_depth_resized.min() + 1e-8)
            vis_moge = (vis_moge * 255).astype(np.uint8)
            vis_moge = (cmap(vis_moge)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(os.path.join(args.outdir, base_name + '_moge.png'), vis_moge)

        # Save edge mask visualization if requested
        if args.save_edge_mask:
            edge_mask_resized = cv2.resize(edge_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            vis_edge = (edge_mask_resized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.outdir, base_name + '_edge_mask.png'), vis_edge)

        # Save raw depth as .npy if requested
        if args.save_npy:
            depth_npy_dir = os.path.join(args.outdir, 'depth_npy')
            os.makedirs(depth_npy_dir, exist_ok=True)
            npy_path = os.path.join(depth_npy_dir, base_name + '.npy')
            np.save(npy_path, hybrid_depth_resized)

    print(f'Done! Results saved to {args.outdir}')


if __name__ == '__main__':
    main()
