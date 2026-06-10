import argparse

import cv2
import numpy as np
import matplotlib.cm as cm
import open3d as o3d


def read_depth_png(path: str) -> np.ndarray:
    """Read a KITTI 16-bit PNG depth map and convert to meters.

    KITTI convention: depth_meters = pixel_value / 256.0
    A value of 0 indicates an invalid / missing measurement.
    """
    depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    if depth_raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG, got {depth_raw.dtype}")
    return depth_raw.astype(np.float64) / 256.0


def write_depth_png(depth: np.ndarray, path: str):
    """Write a depth map as KITTI 16-bit PNG.

    KITTI convention: depth_meters = pixel_value / 256.0
    Values <= 0 are saved as 0 (invalid).
    """
    encoded = np.clip(np.round(depth * 256), 0, 65535).astype(np.uint16)
    cv2.imwrite(path, encoded)


def inflate_depth(depth: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Simulate depth inflation: expand each valid pixel outward, with foreground
    (closer / smaller depth) taking priority over background when expansions overlap.

    For every output pixel, the value is the *minimum* valid depth among all valid
    pixels inside a (kernel_size x kernel_size) elliptical neighbourhood.
    The coverage is equivalent to morphological dilation of the validity mask.
    """
    valid = depth > 0
    if kernel_size <= 1 or not np.any(valid):
        return depth.copy()

    LARGE = float(1e6)
    filled = np.where(valid, depth, LARGE).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # erode → minimum in neighbourhood → foreground depth wins
    eroded = cv2.erode(filled, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=LARGE)

    # dilate mask → expanded coverage
    dilated_valid = cv2.dilate(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    return np.where((dilated_valid > 0) & (eroded < LARGE), eroded.astype(depth.dtype), 0.0)


def dilate_depth(depth: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Dilate sparse depth map so each valid point becomes a larger patch.

    Uses nearest-neighbor interpolation to avoid blurring depth values.
    """
    valid = depth > 0
    if kernel_size <= 1 or not np.any(valid):
        return depth.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Dilate the depth values: mask invalid as 0, dilate picks max in neighborhood
    dilated = cv2.dilate(depth.astype(np.float32), kernel)
    # Dilate the validity mask to know which pixels to keep
    dilated_valid = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1)
    result = np.where(dilated_valid > 0, dilated, 0)
    return result


def visualize_depth_overlay(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.6,
    colormap: str = "magma",
    max_depth: float = None,
    point_size: int = 3,
) -> np.ndarray:
    """Overlay a semi-transparent depth map on an RGB image.

    Args:
        image: RGB image (H, W, 3), uint8.
        depth: Depth in meters (H, W). 0 = invalid (shown as transparent).
        alpha: Transparency of the depth overlay.
        colormap: Matplotlib colormap name.
        max_depth: Clip depth at this value for normalization.
                   Defaults to the 95th percentile of valid pixels.
        point_size: Kernel size for dilating sparse depth points (1 = original).

    Returns:
        Blended (H, W, 3) uint8 image.
    """
    if image.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"Shape mismatch: image {image.shape[:2]} vs depth {depth.shape[:2]}"
        )

    depth = dilate_depth(depth, kernel_size=point_size)

    valid = depth > 0
    if not np.any(valid):
        return image.copy()

    max_val = max_depth if max_depth is not None else float(np.percentile(depth[valid], 95))
    norm = np.clip(depth / max_val, 0, 1)

    cmap = cm.get_cmap(colormap)  # noqa: deprecated but works
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    mask = valid.astype(np.float32)
    blended = image.astype(np.float32) * (1 - alpha * mask[..., None]) + \
              colored.astype(np.float32) * (alpha * mask[..., None])
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_gt_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    save_path: str,
    intrinsic: np.ndarray = None,
    flip: bool = True,
):
    """Save sparse GT depth as a colored point cloud (.ply).

    Args:
        depth: (H, W) depth in meters. 0 = invalid.
        rgb: (H, W, 3) RGB image, uint8.
        save_path: Output .ply path.
        intrinsic: (3, 3) camera intrinsic matrix.
                   Defaults to KITTI image_02 rectified: fx=fy=707.0912.
        flip: Flip Y and Z axes for Open3D convention.
    """
    if intrinsic is None:
        intrinsic = np.array([
            [707.0912, 0.0, 601.8873],
            [0.0, 707.0912, 183.1104],
            [0.0, 0.0, 1.0],
        ])

    H, W = depth.shape
    mask = depth > 0

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth.reshape(-1)
    X = ((u.reshape(-1) - cx) / fx) * Z
    Y = ((v.reshape(-1) - cy) / fy) * Z
    points = np.stack([X, Y, Z], axis=1)

    flat_mask = mask.reshape(-1)
    points = points[flat_mask]
    colors = (rgb.reshape(-1, 3)[flat_mask]).astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if flip:
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points) * np.array([1, -1, -1])
        )

    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved point cloud ({len(points)} points): {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI depth visualization")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to 16-bit depth PNG")
    parser.add_argument("--output", default="depth_overlay.png", help="Output path")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay transparency")
    parser.add_argument("--colormap", default="magma", help="Matplotlib colormap")
    parser.add_argument("--max_depth", type=float, default=None, help="Max depth for normalization (m)")
    parser.add_argument("--point_size", type=int, default=5, help="Dilation kernel size for depth points (1=original)")
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read RGB image: {args.rgb}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = read_depth_png(args.depth)

    vis = visualize_depth_overlay(rgb, depth, alpha=args.alpha, colormap=args.colormap, max_depth=args.max_depth, point_size=args.point_size)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, vis_bgr)
    print(f"Saved to {args.output}")
