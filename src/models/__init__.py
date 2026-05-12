"""Image-model dispatcher."""
def build_image_model(args):
    name = getattr(args, "model", "unet2d").lower()
    if name == "unet2d":
        from .unet2d import build_unet2d
        return build_unet2d(sample_size=32, in_channels=3, out_channels=3)
    if name == "dit":
        from .dit import build_dit
        return build_dit(sample_size=32, in_channels=3, out_channels=3,
                         patch_size=args.dit_patch_size,
                         hidden=args.dit_hidden,
                         depth=args.dit_depth,
                         heads=args.dit_heads)
    raise ValueError(f"unknown image model: {name}")
