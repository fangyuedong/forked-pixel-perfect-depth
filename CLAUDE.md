# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixel-Perfect Depth is a monocular depth estimation model using pixel-space diffusion transformers. The model integrates discriminative representations (Vision Transformers) into generative modeling (Diffusion Transformers) to produce high-quality depth maps that can generate flying-pixel-free point clouds.

This repo extends PPD with two capabilities built on top of the base model:
- **Depth Inpainting** (`DepthInpaintPipeline`): Sparse metric depth (e.g., KITTI LiDAR) → dense metric depth via PPD + LanPaint
- **Depth Refinement** (in progress): Reducing artifacts (high-reflectivity expansion, calibration drift) in sparse depth before inpainting

## Common Commands

### Conda Environment
```bash
conda activate visionreasoner
```

### Installation
```bash
pip install -r requirements.txt
```

### Inference
```bash
# Depth estimation on images (default path: assets/examples/images, default output: depth_vis)
python run.py

# Point cloud generation (requires MoGe2 for metric depth)
python run_point_cloud.py --save_pcd

# Video depth estimation (requires PPVD model)
python run_video.py --video_path assets/examples/video/0001.mp4
```

### KITTI Depth Inpainting
```bash
# Single sample test (default DA2 semantics)
python test_step6.py

# Batch test with custom samples and output directory
python test_step6.py --outdir step6/custom --pairs rgb1.png depth1.png rgb2.png depth2.png ...

# With MoGe2 semantics
python test_step6.py --pairs rgb.png depth.png  # (modify pipeline in script)
```

### KITTI Depth Refinement (WIP)
```bash
# Step 1: Generate inflated depth from sparse GT
python refine_step1.py   # → refine_step1/

# Step 4: Debug scale alignment with downsampled depth
python refine_step4.py   # → refine_step4/
```

### Training
```bash
# Stage 1: Pretraining on Hypersim at 512x512 resolution
python main.py --cfg_file ppd/configs/train_pretrain.yaml pl_trainer.devices=8

# Stage 2: Fine-tuning on mixed datasets at 1024x768 resolution
python main.py --cfg_file ppd/configs/train_finetune.yaml pl_trainer.devices=8

# Validation/testing
python main.py --cfg_file ppd/configs/train_finetune.yaml --entry val
```

### Required Checkpoints
Place these in `checkpoints/`:
- `ppd.pth` - Main PPD model (Hugging Face)
- `ppd_moge.pth` - PPD with MoGe2 semantics (Google Drive)
- `moge2.pt` - MoGe2 model for metric depth (Hugging Face)
- `depth_anything_v2_vitl.pth` - Depth Anything V2 (Hugging Face)
- `ppvd.pth` - Video model (Google Drive)
- `pi3.safetensors` - Pi3 video model (Hugging Face)

## High-Level Architecture

### Model Architecture
The model uses a three-stage pipeline:

1. **Semantic Encoder** (`ppd/models/depth_anything_v2/` or `ppd/moge/`):
   - Provides initial depth estimation and semantic features
   - Two backends: Depth Anything V2 (DA2) or MoGe2
   - Loaded in eval mode with frozen gradients

2. **Diffusion Transformer (DiT)** (`ppd/models/dit.py`):
   - Pure transformer architecture (no convolutional layers)
   - Architecture: depth=24, hidden_size=1024, patch_size=8, num_heads=16
   - Input channels: 4 (concatenated latent depth + RGB condition)
   - Output channels: 1 (depth)

3. **Diffusion Pipeline** (`ppd/utils/diffusion/`):
   - Schedule: Linear interpolation (lerp) with T=1000
   - Sampler: Euler sampler with velocity prediction
   - Training timesteps: Logit-normal distribution
   - Sampling timesteps: Uniform with 4 steps (inference: 10 steps default)

### Data Flow
```
Input Image → Semantic Encoder → Semantics
                    ↓
RGB Condition (image - 0.5)
                    ↓
Random Noise Latent ←→ DiT ←→ Sampler (iterative refinement)
                    ↓
Final Depth (latent + 0.5)
```

### Training Strategy
- **Stage 1 (Pretrain)**: 512x512 resolution on Hypersim dataset
- **Stage 2 (Fine-tune)**: 1024x768 resolution on 5 mixed datasets (Hypersim, UrbanSyn, UnrealStereo4K, vKITTI, TartanAir)
- **Validation**: Evaluated on NYU, DIODE, ETH3D, ScanNet, and KITTI

## Depth Inpainting Pipeline

### Core Class: `DepthInpaintPipeline` (`ppd/models/depth_inpainter.py`)

Transforms sparse metric depth into dense metric depth:

```
Sparse Metric Depth + RGB Image
  → PPD relative depth prediction
  → RANSAC: log(gt+1) = a * ppd_rel + b  (fit on valid GT pixels)
  → GT → PPD space: gt_ppd = (log(gt+1) - b) / a - 0.5
  → LanPaint inpainting with gt_ppd as known depth
  → RANSAC back to metric: log(gt+1) = a2 * (result+0.5) + b2
  → Dense metric depth
```

Key parameters:
- `semantics_model`: `'DA2'` (default) or `'MoGe2'`
- `sampling_steps`: Diffusion sampling steps (default 10)
- `fld_steps`, `fld_step_size`, `fld_lambda`, `fld_friction`: LanPaint FLD parameters
- `debug_dir`: If set, saves RGB + known_depth overlay at PPD resolution for debugging

### LanPaint Inpainter (`ppd/models/lanpaint_inpainter.py`)

Implements Fast Langevin Dynamics (FLD) for depth inpainting with BiG Score (Bidirectional Guidance). Key bug fixes applied:
- `compute_score` converts x_t from internal format back to model format before DiT call
- Final replacement step restores known pixels exactly after diffusion
- Replace noise regenerated each outer loop iteration (Step 8 verification)

Supporting modules:
- `ppd/models/lanpaint_types.py` - LangevinState dataclass
- `ppd/models/lanpaint_utils.py` - StochasticHarmonicOscillator

### KITTI Utilities (`kitti_utils.py`)

- `read_depth_png(path)` / `write_depth_png(depth, path)` - KITTI 16-bit PNG (depth_meters = pixel / 256.0)
- `inflate_depth(depth, kernel_size)` - Simulate depth inflation, foreground covers background
- `dilate_depth(depth, kernel_size)` - Standard morphological dilation
- `visualize_depth_overlay(image, depth, ...)` - Semi-transparent depth on RGB
- `save_gt_point_cloud(depth, rgb, save_path, ...)` - Sparse GT → colored .ply

### LanPaint Parameter Guide

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fld_steps` (N) | 5 | FLD iterations per diffusion step. Larger → better known-region preservation, more compute |
| `fld_step_size` (η) | 0.2 | Langevin base step size. Adaptive: `η × (1 - ᾱ_t)` |
| `fld_lambda` (λ) | 16.0 | BiG Score guidance strength. Larger → stronger pull toward known values |
| `fld_friction` (Γ) | 15.0 | Langevin damping. Larger → more stable, less exploration |

| Goal | Adjustment |
|------|-----------|
| Better known-region preservation | ↑λ (16→32) or ↑N (5→10) |
| Faster inference | ↓N (5→3) or ↓sampling_steps |
| Numerical instability | ↓η (0.2→0.1), ↑Γ (15→20) |
| Known-region over-sharpening | ↓λ (16→8) |

## Code Structure

### Entry Points
- `main.py` - Training entry point using PyTorch Lightning
  - Uses OmegaConf for configuration merging with CLI overrides
  - Entry functions defined in `ppd/entrys/`: `train_net`, `val`, `predict`
- `run.py` - Single image depth estimation
- `run_point_cloud.py` - Image to point cloud (metric depth required)
- `run_video.py` - Video depth estimation (PPVD model)

### KITTI Depth Scripts (root directory)
- `test_step6.py` - Batch depth inpainting test with `DepthInpaintPipeline`
- `refine_step1.py` - Depth inflation test
- `refine_step3.py` - Pipeline test with various depth inputs and semantics models
- `refine_step4.py` - Scale alignment debugging with downsampled depth
- `run_kitti_inpaint.py` - CLI for KITTI depth inpainting
- `run_hybrid_depth.py` - MoGe-2 + PPD hybrid pipeline

### Package Structure (`ppd/`)
- `models/` - Core model implementations
  - `ppd.py` - Inference model for images
  - `ppd_train.py` - Training wrapper (extends ppd.py)
  - `ppvd.py` - Video depth estimation model
  - `dit.py` - Diffusion Transformer
  - `dit_video.py` - Video DiT variant
  - `depth_inpainter.py` - `DepthInpaintPipeline` for sparse→dense depth
  - `lanpaint_inpainter.py` - LanPaint FLD inpainter
  - `lanpaint_types.py` - LangevinState dataclass
  - `lanpaint_utils.py` - StochasticHarmonicOscillator
  - Supporting: `attention.py`, `mlp.py`, `patch_embed.py`

- `utils/` - Utilities
  - `diffusion/` - Schedule, sampler, timesteps
  - `transform.py` - Image transformations (resize, crop, tensor conversion)
  - `depth2pcd.py` - Depth to point cloud conversion
  - `depth_normalization.py` - Depth normalization for PPD (log + quantile + scale to [-0.5, 0.5])
  - `align_depth_func.py` - Depth alignment
  - `video_utils.py` - Video processing

- `configs/` - YAML configurations
  - `train_pretrain.yaml` - Pretraining config
  - `train_finetune.yaml` - Fine-tuning config
  - `config.py` - Config loader with CLI override support

- `data/` - Dataset loading and transforms
- `datasets/` - Dataset metadata and split files
- `entrys/` - Training/validation entry functions

### Configuration System
- Uses OmegaConf for YAML-based configuration
- CLI arguments can override any config value (e.g., `pl_trainer.devices=8`)
- Key config sections: `data`, `model`, `callbacks`, `logger`, `pl_trainer`
- Config merging: `cfg = OmegaConf.merge(cfg, cli_cfg)`

### External Dependencies
- `RePaint/` - Denoising diffusion inpainting submodule
- `utils3d` - 3D utilities (custom fork: EasternJournalist/utils3d@c5daf6f)

## Key Implementation Details

### Resolution Handling
- Training resolution: Fixed (512x512 for pretrain, 1024x768 for fine-tune)
- Inference: Flexible resolution/aspect ratio via `resize_keep_aspect()`
- KITTI image (374×1238) resizes to (480×1616) for PPD
- Input condition: RGB normalized to [-0.5, 0.5]

### Diffusion Details
- Pixel-space diffusion (no VAE, operates directly in image space)
- Prediction type: `v_lerp` (velocity prediction for linear schedule)
- Sampler step formula: `latent = sampler.step(pred, x_t=latent, t=timestep)`
- Model input: Concatenated [latent_depth, rgb_condition] along channel dim

### Model Loading
- Uses `torch.load(..., strict=False)` for flexible loading
- Models set to `.eval()` mode with `.requires_grad_(False)` for inference
- Supports both CUDA and MPS (Apple Silicon) devices

### RANSAC Normalization
- Maps between metric depth and PPD relative space: `log(gt+1) = a * ppd_rel + b`
- Forward RANSAC: fits all valid sparse depth pixels to PPD prediction
- Inverse RANSAC: maps inpainted result back to metric using same valid pixels
- Ensures known pixels are perfectly preserved (verified by forward/inverse coefficient match)

### KITTI Defaults
- Intrinsic (image_02 rectified): fx=fy=707.0912, cx=601.8873, cy=183.1104
- Depth format: 16-bit PNG, `depth_meters = pixel_value / 256.0`, 0 = invalid
- Dataset path: `/home/fanguedong/Dataset/kitti_depth/`

### Training Configuration
- Framework: PyTorch Lightning with DDP strategy
- Optimizer: AdamW (lr=1e-4, weight_decay=0.0)
- Precision: bf16-mixed
- Batch size: 4
- Monitoring: val/relative_abs_rel/dataloader_idx_1 (NYU dataset)
