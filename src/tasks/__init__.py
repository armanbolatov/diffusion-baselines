"""Task dispatcher."""
def get_task(args):
    name = args.task
    if name == "image_celeba":
        from .image_celeba import ImageCelebA
        return ImageCelebA()
    if name == "text_text8":
        from .text_text8 import TextText8
        return TextText8()
    raise ValueError(f"unknown task: {name}")
