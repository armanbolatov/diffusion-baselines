"""Wrapper around diffusers UNet2DModel for 32x32 RGB image diffusion."""
from diffusers import UNet2DModel


def build_unet2d(sample_size: int = 32, in_channels: int = 3, out_channels: int = 3):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
