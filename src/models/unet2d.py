"""Wrapper around diffusers UNet2DModel.

For 64x64 RGB diffusion we follow the DDPM-paper recipe (Ho et al. 2020,
Sec 4): 4 levels, channel multipliers (1, 2, 2, 4), 2 residual blocks
per level, attention at the second-deepest resolution (16x16). We use
96 base channels (instead of the paper's 128) to fit alongside other
GPU users; this lands around ~38M params, in the same range as the
paper's 35.7M-param CIFAR-10 model.
"""
from diffusers import UNet2DModel


def build_unet2d(sample_size: int = 32, in_channels: int = 3, out_channels: int = 3):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(96, 192, 192, 384),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
