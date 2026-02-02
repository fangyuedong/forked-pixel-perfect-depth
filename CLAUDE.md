# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixel-Perfect Depth is a monocular depth estimation model using pixel-space diffusion transformers. The model integrates discriminative representations (Vision Transformers) into generative modeling (Diffusion Transformers) to produce high-quality depth maps that can generate flying-pixel-free point clouds.

## Common Commands

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

## Code Structure

### Entry Points
- `main.py` - Training entry point using PyTorch Lightning
  - Uses OmegaConf for configuration merging with CLI overrides
  - Entry functions defined in `ppd/entrys/`: `train_net`, `val`, `predict`
- `run.py` - Single image depth estimation
- `run_point_cloud.py` - Image to point cloud (metric depth required)
- `run_video.py` - Video depth estimation (PPVD model)

### Package Structure (`ppd/`)
- `models/` - Core model implementations
  - `ppd.py` - Inference model for images
  - `ppd_train.py` - Training wrapper (extends ppd.py)
  - `ppvd.py` - Video depth estimation model
  - `dit.py` - Diffusion Transformer
  - `dit_video.py` - Video DiT variant
  - Supporting: `attention.py`, `mlp.py`, `patch_embed.py`

- `utils/` - Utilities
  - `diffusion/` - Schedule, sampler, timesteps
  - `transform.py` - Image transformations (resize, crop, tensor conversion)
  - `depth2pcd.py` - Depth to point cloud conversion
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

### Training Configuration
- Framework: PyTorch Lightning with DDP strategy
- Optimizer: AdamW (lr=1e-4, weight_decay=0.0)
- Precision: bf16-mixed
- Batch size: 4
- Monitoring: val/relative_abs_rel/dataloader_idx_1 (NYU dataset)
