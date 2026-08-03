"""Image-model dispatcher."""
def build_image_model(args, sample_size: int = 64):
    from .unet2d import build_unet2d
    return build_unet2d(sample_size=sample_size, in_channels=3, out_channels=3,
                        dropout=float(getattr(args, "dropout", 0.0)))
