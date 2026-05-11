"""Task dispatcher."""
def get_task(args):
    name = args.task
    if name == "image_cifar10":
        from .image_cifar10 import ImageCIFAR10
        return ImageCIFAR10()
    if name == "text_text8":
        from .text_text8 import TextText8
        return TextText8()
    raise ValueError(f"unknown task: {name}")
