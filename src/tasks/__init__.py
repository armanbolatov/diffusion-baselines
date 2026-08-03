"""Task dispatcher."""
def get_task(args):
    name = args.task
    if name == "image_celeba":
        from .image_celeba import ImageCelebA
        return ImageCelebA()
    if name == "text_text8":
        from .text_text8 import TextText8
        return TextText8()
    if name == "text_lm1b":
        from .text_lm1b import TextLM1B
        return TextLM1B()
    if name == "mol_qm9":
        from .mol_qm9 import MolQM9
        return MolQM9()
    raise ValueError(f"unknown task: {name}")
