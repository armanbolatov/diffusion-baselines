"""Wrapper around diffusers UNet2DModel.

Two configs, chosen by ``sample_size``:

- 64x64 (CelebA): the ermongroup/ddim recipe — base ch 128, mult [1,2,2,2,4]
  (5 levels), 2 res blocks, attention at 16x16, dropout 0.1. ~78M params, the
  standard setup for good CelebA-64 FID.
- 32x32: a lighter 4-level ch-96 model (~38M) for quick experiments.
"""
from diffusers import UNet2DModel


def build_unet2d(sample_size: int = 32, in_channels: int = 3,
                 out_channels: int = 3, dropout: float = 0.0):
    if sample_size >= 64:
        block_out_channels = (128, 256, 256, 256, 512)           # ch=128 * [1,2,2,2,4]
        down = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D",  # attention at 16x16
                "DownBlock2D", "DownBlock2D")
        up = ("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    else:
        block_out_channels = (96, 192, 192, 384)
        down = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    return UNet2DModel(
        sample_size=sample_size, in_channels=in_channels, out_channels=out_channels,
        layers_per_block=2, block_out_channels=block_out_channels,
        down_block_types=down, up_block_types=up, dropout=dropout,
    )
