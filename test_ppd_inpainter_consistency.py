"""
Test script to verify PixelPerfectDepth.infer_image and LanPaintInpainter.inpaint consistency.

This test validates that when:
1. edge_mask = all ones (all pixels marked as edges, need inpainting)
2. n_steps = 0 (disable Langevin dynamics, no FLD iteration)
3. semantics_model = DA2 (use DepthAnythingV2)

Then both methods should produce identical outputs, proving that LanPaintInpainter's
base inference flow is correct when used with PPD.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

from ppd.utils.set_seed import set_seed
from ppd.models.ppd import PixelPerfectDepth
from ppd.models.lanpaint_inpainter import LanPaintInpainter
from ppd.utils.transform import resize_keep_aspect, image2tensor
from ppd.utils.utils import has_native_bf16


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_stats(name, tensor):
    """Print tensor statistics."""
    print(f"  {name} shape: {tensor.shape}")
    print(f"  {name} dtype: {tensor.dtype}")
    print(f"  {name} device: {tensor.device}")
    print(f"  {name} range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"  {name} mean: {tensor.mean().item():.4f}")
    print(f"  {name} std: {tensor.std().item():.4f}")


def compare_tensors(ppd_output, inpaint_output, tolerance=1e-5):
    """Compare two tensors and return detailed comparison results."""
    results = {
        'shape_match': ppd_output.shape == inpaint_output.shape,
        'dtype_match': ppd_output.dtype == inpaint_output.dtype,
        'max_diff': torch.max(torch.abs(ppd_output - inpaint_output)).item(),
        'mean_diff': torch.mean(torch.abs(ppd_output - inpaint_output)).item(),
        'std_diff': torch.std(ppd_output - inpaint_output).item(),
    }

    # Check if values are close within tolerance
    results['values_match'] = results['max_diff'] < tolerance

    # Check statistics match
    results['stats_match'] = (
        abs(ppd_output.mean().item() - inpaint_output.mean().item()) < tolerance and
        abs(ppd_output.std().item() - inpaint_output.std().item()) < tolerance
    )

    return results


def main():
    # Configuration
    SEED = 666
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    SEMANTICS_MODEL = 'DA2'
    SEMANTICS_PTH = 'checkpoints/depth_anything_v2_vitl.pth'
    MODEL_PTH = 'checkpoints/ppd.pth'
    SAMPLING_STEPS = 10
    TEST_IMAGE_PATH = 'assets/examples/images/0001.jpg'
    OUTPUT_DIR = 'test_consistency_output'

    print_section("Test PixelPerfectDepth.infer_image vs LanPaintInpainter.inpaint Consistency")

    print(f"\nConfiguration:")
    print(f"  Input image: {TEST_IMAGE_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Semantics model: {SEMANTICS_MODEL}")
    print(f"  Sampling steps: {SAMPLING_STEPS}")
    print(f"  Random seed: {SEED}")
    print(f"  edge_mask: all ones (all regions need inpainting)")
    print(f"  n_steps: 0 (disable Langevin dynamics)")

    # Set random seed for reproducibility
    set_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        # Try alternative paths
        alternative_paths = [
            'assets/examples/images/0001.jpg',
            'assets/examples/0001.jpg',
            'assets/0001.jpg',
        ]
        found = False
        for path in alternative_paths:
            if os.path.exists(path):
                TEST_IMAGE_PATH = path
                found = True
                break
        if not found:
            print(f"\nError: Test image not found at {TEST_IMAGE_PATH}")
            print("Please provide a valid test image path.")
            return 1

    print_section("Loading Models")

    try:
        # Load PixelPerfectDepth model
        print(f"\nLoading PixelPerfectDepth model...")
        ppd_model = PixelPerfectDepth(
            semantics_model=SEMANTICS_MODEL,
            semantics_pth=SEMANTICS_PTH,
            sampling_steps=SAMPLING_STEPS
        )
        ppd_model.load_state_dict(torch.load(MODEL_PTH, map_location='cpu'), strict=False)
        ppd_model = ppd_model.to(DEVICE).eval()
        ppd_model.requires_grad_(False)
        print(f"  PPD model loaded successfully")

        # Use the SAME sampler and schedule from PPD model for LanPaintInpainter
        # This ensures exact consistency
        print(f"\nLoading LanPaintInpainter (using PPD's sampler)...")
        inpainter = LanPaintInpainter(
            schedule=ppd_model.schedule,  # Use same schedule
            sampler=ppd_model.sampler,     # Use same sampler
            dit_model=ppd_model.dit,
            sem_encoder=ppd_model.sem_encoder,
            device=DEVICE,
            n_steps=0,  # Disable Langevin dynamics
            step_size=0.2,
            lambda_big=16.0,
            friction=15.0
        )
        print(f"  LanPaintInpainter loaded successfully (n_steps=0)")
        print(f"  Using PPD's schedule and sampler")

    except Exception as e:
        print(f"\nError loading models: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("Preparing Input")

    try:
        # Load and preprocess image
        image = Image.open(TEST_IMAGE_PATH).convert('RGB')
        image = np.array(image)[:, :, ::-1]  # PIL RGB to BGR for cv2 compatibility

        # Resize image
        resize_image = resize_keep_aspect(image.copy())
        image_tensor = image2tensor(resize_image)
        image_tensor = image_tensor.to(DEVICE)

        print(f"\nInput image loaded:")
        print(f"  Original size: {image.shape[:2]}")
        print(f"  Resized to: {resize_image.shape[:2]}")
        print_stats("image_tensor", image_tensor)

        # Prepare known_depth (dummy values, will be ignored since edge_mask=all ones)
        # But we need it to match shapes
        H, W = resize_image.shape[:2]
        known_depth = torch.zeros(1, 1, H, W, device=DEVICE)

        # Create edge_mask = all ones
        edge_mask = torch.ones(1, 1, H, W, device=DEVICE)

        print(f"\nAdditional inputs:")
        print(f"  known_depth shape: {known_depth.shape}")
        print(f"  edge_mask shape: {edge_mask.shape}")
        print(f"  edge_mask all ones: {torch.all(edge_mask == 1.0).item()}")

    except Exception as e:
        print(f"\nError preparing input: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("Running PixelPerfectDepth.forward_test (direct call)")

    try:
        # Reset seed before PPD inference
        set_seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
        with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
            # Call forward_test directly with the exact same image_tensor
            ppd_depth = ppd_model.forward_test(image_tensor)

        print_stats("PPD output", ppd_depth)

    except Exception as e:
        print(f"\nError running PPD inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("Running LanPaintInpainter.inpaint")

    try:
        # Reset seed before inpaint inference (crucial for same random noise initialization)
        set_seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # Note: inpainter.inpaint expects rgb_condition in [0, 1] range
        # image2tensor returns values in [0, 1] range
        rgb_condition = image_tensor  # Already in [0, 1] range

        autocast_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
        with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
            inpaint_depth = inpainter.inpaint(
                rgb_condition=rgb_condition,
                known_depth=known_depth,
                edge_mask=edge_mask,
                num_steps=SAMPLING_STEPS
            )

        print_stats("Inpaint output", inpaint_depth)

    except Exception as e:
        print(f"\nError running inpaint inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("Comparison Results")

    # Note: PPD returns latent + 0.5 (no clamping), while inpaint returns clamp(x_t + 0.5, 0, 1)
    # For fair comparison, we should compare the unclamped versions or apply same clamping
    ppd_depth_clamped = torch.clamp(ppd_depth, 0.0, 1.0)

    # Compare outputs
    results = compare_tensors(ppd_depth_clamped, inpaint_output=inpaint_depth, tolerance=1e-5)

    print(f"\nShape match: {'✓' if results['shape_match'] else '✗'}")
    print(f"Dtype match: {'✓' if results['dtype_match'] else '✗'}")
    print(f"Statistics match: {'✓' if results['stats_match'] else '✗'}")
    print(f"Values match (max_diff < 1e-5): {'✓' if results['values_match'] else '✗'}")

    print(f"\nDifference statistics:")
    print(f"  Maximum absolute difference: {results['max_diff']:.10f}")
    print(f"  Mean absolute difference: {results['mean_diff']:.10f}")
    print(f"  Std of difference: {results['std_diff']:.10f}")

    # Also compare unclamped PPD output to inpaint output (to show clamping effect)
    print(f"\nNote: PPD output is not clamped, inpaint output is clamped to [0, 1]")
    print(f"  PPD unclamped range: [{ppd_depth.min().item():.4f}, {ppd_depth.max().item():.4f}]")
    print(f"  PPD clamped range: [{ppd_depth_clamped.min().item():.4f}, {ppd_depth_clamped.max().item():.4f}]")
    print(f"  Inpaint range: [{inpaint_depth.min().item():.4f}, {inpaint_depth.max().item():.4f}]")

    # Overall result
    all_pass = results['shape_match'] and results['values_match'] and results['stats_match']

    if all_pass:
        print_section("TEST PASSED")
        print("\n✅ Test PASSED! The two methods produce identical outputs.")
        print("\nThis confirms that when:")
        print("  - edge_mask = all ones (all regions need inpainting)")
        print("  - n_steps = 0 (Langevin dynamics disabled)")
        print("  - semantics_model = DA2 (DepthAnythingV2)")
        print("\nLanPaintInpainter.inpaint behaves identically to PixelPerfectDepth.forward_test,")
        print("verifying the correctness of the base inference flow.")
    else:
        print_section("TEST FAILED")
        print("\n❌ Test FAILED! The outputs differ.")
        print("\nPossible reasons:")
        print("  1. Random seed not properly reset between runs")
        print("  2. Different preprocessing of inputs")
        print("  3. Different model state or weights")
        print("  4. Numerical precision differences")
        print("  5. Difference in how semantics are computed")
        print("  6. Difference in the internal format conversions (model_to_internal/internal_to_model)")

    # Optional: Save outputs for visual inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'ppd_output.npy'), ppd_depth.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, 'ppd_output_clamped.npy'), ppd_depth_clamped.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, 'inpaint_output.npy'), inpaint_depth.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, 'difference.npy'), (ppd_depth_clamped - inpaint_depth).cpu().numpy())
    print(f"\nOutputs saved to {OUTPUT_DIR}/")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
