import torch
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
model = make_pipeline(poly_features, ransac)


def detect_depth_edges(depth, gradient_threshold=0.05, dilation_iter=3):
    """
    Detect depth edges using gradient-based approach.

    Args:
        depth: numpy array of shape (H, W) - depth map
        gradient_threshold: float - threshold for edge detection (relative to max gradient)
        dilation_iter: int - number of dilation iterations to expand edge mask

    Returns:
        edge_mask: numpy array of shape (H, W) - binary mask (1 for edges, 0 for interior)
    """
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize and apply threshold
    max_grad = gradient_magnitude.max()
    if max_grad > 0:
        normalized_grad = gradient_magnitude / max_grad
    else:
        normalized_grad = gradient_magnitude

    # Create edge mask
    edge_mask = (normalized_grad > gradient_threshold).astype(np.uint8)

    # Dilate edges to create a wider region for refinement
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=dilation_iter)

    return edge_mask


def detect_canny_edges(depth, dilation_iter=3):
    """
    Detect depth edges using Canny edge detector.

    Args:
        depth: numpy array of shape (H, W) - depth map
        dilation_iter: int - number of dilation iterations to expand edge mask

    Returns:
        edge_mask: numpy array of shape (H, W) - binary mask (1 for edges, 0 for interior)
    """
    # Normalize depth to 0-255 range for Canny
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(depth_normalized, 50, 150)

    # Convert to binary mask (1 for edges, 0 for interior)
    edge_mask = (edges > 0).astype(np.uint8)

    # Dilate edges to create a wider region for refinement
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=dilation_iter)

    return edge_mask

def recover_metric_depth_ransac(pred, gt, mask, log=True):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mask_gt = gt[mask].astype(np.float32)
    ori_mask_gt = mask_gt
    mask_pred = pred[mask].astype(np.float32)

    ## depth -> log depth
    mask_gt = np.log(mask_gt + 1.)

    try:
        model.fit(mask_pred[:, None], mask_gt[:, None])
        a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
        a = a.item()
        b = b.item()
    except:
        a, b = 1, 0
        
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(mask_pred)
        gt_mean = np.mean(mask_gt)
        pred_metric = pred * (gt_mean / pred_mean)

    ## log depth -> depth
    pred_metric = np.exp(pred_metric) - 1.
    pred_metric = np.clip(pred_metric, 1e-3, np.max(ori_mask_gt))
    return pred_metric