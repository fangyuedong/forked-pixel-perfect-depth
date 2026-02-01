from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from omegaconf import DictConfig
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.diffusion.masked_sampler import MaskedEulerSampler
from ppd.utils.transform import image2tensor, resize_1024, resize_1024_crop, resize_keep_aspect

from ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from ppd.models.dit import DiT

class PixelPerfectDepth(nn.Module):
    def __init__(
        self,
        semantics_model='MoGe2',
        semantics_pth='checkpoints/moge2.pt',
        sampling_steps=10,
        ):
        super().__init__()
        self.sampling_steps = sampling_steps
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = DEVICE

        if semantics_model == 'MoGe2':
            from ppd.moge.model.v2 import MoGeModel
            self.sem_encoder = MoGeModel.from_pretrained(semantics_pth)
        else:
            self.sem_encoder = DepthAnythingV2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024]
            )
            self.sem_encoder.load_state_dict(torch.load(semantics_pth, map_location='cpu'), strict=False)
        self.sem_encoder = self.sem_encoder.to(self.device).eval()
        self.sem_encoder.requires_grad_(False)

        self.configure_diffusion()
        self.dit = DiT()

    def configure_diffusion(self):
        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.sampling_steps,
            device=self.device,
            )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type='velocity'
            )
    
    @torch.no_grad()
    def infer_image(self, image, use_fp16: bool = True):
        # Resize the image to match the training resolution area while keeping the original aspect ratio.
        resize_image = resize_keep_aspect(image)
        image = image2tensor(resize_image)
        image = image.to(self.device)
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            depth = self.forward_test(image)
        return depth, resize_image
    
    @torch.no_grad()
    def forward_test(self, image):

        semantics = self.semantics_prompt(image)
        cond = image - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)
        
        for timestep in self.sampling_timesteps:
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)
        return latent + 0.5


    @torch.no_grad()
    def semantics_prompt(self, image):
        with torch.no_grad():
            semantics = self.sem_encoder.forward_semantics(image)
        return semantics

    @torch.no_grad()
    def infer_hybrid(
        self,
        image,
        moge_depth,
        edge_mask,
        sampling_steps=10,
        use_fp16: bool = True,
    ):
        """
        Hybrid inference that refines only edge regions using PPD while keeping
        interior regions from MoGe-2 depth map.

        Args:
            image: numpy array (H, W, 3) - input image
            moge_depth: numpy array (H, W) - MoGe-2 depth map (metric depth)
            edge_mask: numpy array (H, W) - binary mask (1=edges, 0=interior)
            sampling_steps: int - number of diffusion sampling steps
            use_fp16: bool - whether to use mixed precision

        Returns:
            hybrid_depth: torch tensor (1, 1, H, W) - hybrid depth map
            resize_image: numpy array - resized image used for inference
        """
        # Resize the image to match the training resolution area while keeping the original aspect ratio.
        resize_image = resize_keep_aspect(image)
        image = image2tensor(resize_image)
        image = image.to(self.device)
        cond = image - 0.5

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            # Resize MoGe-2 depth and edge mask to match the model input resolution
            H, W = resize_image.shape[:2]

            # Convert MoGe-2 depth to torch tensor and resize
            moge_depth_tensor = torch.tensor(moge_depth, dtype=torch.float32, device=self.device)
            moge_depth_tensor = torch.nn.functional.interpolate(
                moge_depth_tensor.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            # Normalize MoGe-2 depth to [0, 1] range for latent space
            moge_depth_min, moge_depth_max = moge_depth_tensor.min(), moge_depth_tensor.max()
            moge_depth_normalized = (moge_depth_tensor - moge_depth_min) / (moge_depth_max - moge_depth_min + 1e-8)

            # Resize edge mask
            edge_mask_tensor = torch.tensor(edge_mask, dtype=torch.float32, device=self.device)
            edge_mask_tensor = torch.nn.functional.interpolate(
                edge_mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            )

            # Create keep_mask (1 for interior, 0 for edges)
            keep_mask = 1.0 - edge_mask_tensor

            # Get semantics from encoder
            semantics = self.semantics_prompt(image)

            # Initialize latent with known regions preserved
            # Shape: (1, 1, H, W) - batch_size=1, channels=1
            latent_shape = (cond.shape[0], 1, cond.shape[2], cond.shape[3])

            # Create masked sampler with specified sampling steps
            masked_sampler = MaskedEulerSampler(
                schedule=self.schedule,
                timesteps=Timesteps(T=self.schedule.T, steps=sampling_steps, device=self.device),
                prediction_type='velocity'
            )

            # Initialize latent with mask (keep interior from MoGe-2, noise for edges)
            latent = masked_sampler.initialize_latent_with_mask(
                latent_shape,
                known_depth=moge_depth_normalized.unsqueeze(0).unsqueeze(0) - 0.5,  # Convert to latent space [-1, 1]
                keep_mask=keep_mask,
                device=self.device
            )

            # Run masked diffusion sampling
            for timestep in masked_sampler.timesteps:
                # Concatenate latent + condition for DiT input
                input = torch.cat([latent, cond], dim=1)
                pred = self.dit(x=input, semantics=semantics, timestep=timestep)

                # Step with mask preservation
                latent = masked_sampler.step(
                    pred=pred,
                    x_t=latent,
                    t=timestep,
                    keep_mask=keep_mask,
                    known_depth=moge_depth_normalized.unsqueeze(0).unsqueeze(0) - 0.5
                )

            # Convert back from latent space (latent + 0.5)
            hybrid_depth = latent + 0.5

            # Denormalize back to MoGe-2 depth scale
            hybrid_depth = hybrid_depth * (moge_depth_max - moge_depth_min + 1e-8) + moge_depth_min

        return hybrid_depth, resize_image
