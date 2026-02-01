# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixel-Perfect Depth is a monocular depth estimation model using pixel-space diffusion transformers. Key innovation: performs diffusion generation directly in pixel space (no VAE), producing flying-pixel-free point clouds suitable for 3D applications.

The model combines discriminative (ViT semantics) and generative (DiT) paradigms with a pure transformer architecture (no convolutional layers).

## Model Architecture

### Core Components

- **PixelPerfectDepth** (`ppd/models/ppd.py`): Main model with semantic encoder + DiT diffusion pipeline
- **DiT** (`ppd/models/dit.py`): Diffusion transformer, 500M params (24 layers, 1024 hidden, 16 heads), 8×8 patches, rotary position embeddings
- **PixelPerfectVideoDepth** (`ppd/models/ppvd.py`): Video variant with temporal modeling, processes 16-frame chunks with 3-frame overlap

### Semantic Encoder Options

- **MoGe2** (`ppd/moge/model/v2.py`): Provides metric depth + camera intrinsics + semantic features. Used for point cloud generation and as semantic encoder.
- **DepthAnythingV2** (`ppd/models/depth_anything_v2/`): Provides relative depth + semantic features. ViT-Large encoder.
- **Pi3**: Video-specific semantic encoder for PPVD.

### Diffusion Process

The diffusion operates in pixel space (not latent space):
1. Input: Image (condition) + random noise latent
2. Semantic encoding via chosen encoder
3. Loop through timesteps: concatenate latent + condition → DiT predicts velocity → sampler updates latent
4. Output: depth map (latent + 0.5)

Key details:
- Velocity prediction type
- Linear noise schedule (T=1000)
- Only 4-20 sampling steps during inference
- Log-normal timestep distribution during training

## Entry Points

### `run.py` - Image depth estimation
```bash
python run.py --img_path assets/examples/images --semantics_model DA2
```
- Outputs depth visualizations
- `--save_npy`: Save raw depth as .npy
- Supports both DA2 and MoGe2 semantics

### `run_point_cloud.py` - Point cloud generation
```bash
python run_point_cloud.py --save_pcd --apply_filter
```
- Uses MoGe2 for metric depth alignment (RANSAC)
- Generates .ply point clouds
- `--apply_filter`: Statistical outlier removal

### `run_video.py` - Video depth estimation
```bash
python run_video.py
```
- Processes 16-frame chunks with overlap
- Outputs video with depth overlay
- Requires PPVD model + Pi3 encoder

### `main.py` - Training
```bash
# Stage 1: Pre-training (512x512 on Hypersim)
python main.py --cfg_file ppd/configs/train_pretrain.yaml pl_trainer.devices=8

# Stage 2: Fine-tuning (1024x768 on 5 datasets)
python main.py --cfg_file ppd/configs/train_finetune.yaml pl_trainer.devices=8
```

## Training Pipeline

Two-stage curriculum:

**Stage 1 (Pre-training):**
- Resolution: 512×512
- Dataset: Hypersim only
- Config: `ppd/configs/train_pretrain.yaml`
- Up to 500 epochs

**Stage 2 (Fine-tuning):**
- Resolution: 1024×768
- Datasets: Hypersim + UrbanSyn + UnrealStereo4K + VKITTI + TartanAir
- Config: `ppd/configs/train_finetune.yaml`
- Adds multi-scale gradient loss (20% weight)

Training uses PyTorch Lightning + Hydra configs, supports multi-GPU DDP.

## Important Implementation Details

### Resolution Handling
Models train at fixed resolution (512×512 or 1024×768) but support flexible resolutions/aspect ratios during inference via `resize_keep_aspect()`.

### DiT Semantic Fusion
Semantics are fused at the middle layer of DiT (layer 11/24) through `proj_fusion` module, which concatenates and processes both representations.

### Metric Depth Recovery
`ppd/utils/align_depth_func.py` contains `recover_metric_depth_ransac()` which aligns relative PPD depth to MoGe metric depth using polynomial regression with RANSAC outlier rejection.

### Edge Masking for Hybrid Approaches
When combining MoGe and PPD, edge detection can be done via:
- Gradient-based: `detect_depth_edges(depth, gradient_threshold, dilation_iter)`
- Canny-based: `detect_canny_edges(depth, dilation_iter)`

## Configuration System

Uses Hydra + OmegaConf. Base configs in `ppd/configs/`:
- `train_pretrain.yaml`: Pre-training configuration
- `train_finetune.yaml`: Fine-tuning configuration

Command-line overrides work via `key=value` syntax (e.g., `pl_trainer.devices=8`).

## Checkpoint Requirements

Place models in `checkpoints/` directory:
- `ppd.pth`: PPD with DA2 semantics
- `ppd_moge.pth`: PPD with MoGe2 semantics
- `moge2.pt`: MoGe-2 model (rename from downloaded `model.pt`)
- `depth_anything_v2_vitl.pth`: DepthAnythingV2 encoder
- `ppvd.pth`: PPVD model
- `pi3.safetensors`: Pi3 encoder (for video)

## Datasets

**Training:** Hypersim (both stages), UrbanSyn, UnrealStereo4K, VKITTI, TartanAir (fine-tuning only)

**Validation:** NYU v2, DIODE, ETH3D, ScanNet, KITTI

Dataset modules are in `ppd/data/` and `ppd/datasets/`. Multi-dataset training via `general_datamodule.py`.
