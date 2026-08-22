"""CelebA loader (raw aligned CelebA, 178x218). Center-cropped to 140x140 (the
standard DDPM/DDIM face crop), resized to image_size, scaled to [-1, 1]."""
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CROP = 140   # central face crop for CelebA-64 DDPM/DDIM


class _CelebA(Dataset):
    def __init__(self, split: str, tx):
        self.ds = load_dataset("flwrlabs/celeba", split=split)
        self.tx = tx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["image"].convert("RGB")
        return self.tx(img), 0


def eval_transform(image_size):
    """Deterministic crop/resize/scale (no augmentation)."""
    return transforms.Compose([
        transforms.CenterCrop(CROP),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2 - 1),
    ])


def celeba_loaders(data_dir, batch_size, eval_batch_size, num_workers=2, image_size=32):
    train_tx = transforms.Compose([
        transforms.CenterCrop(CROP),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2 - 1),
    ])
    eval_tx = eval_transform(image_size)
    train_ds = _CelebA("train", train_tx)
    eval_ds  = _CelebA("valid", eval_tx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=eval_batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    return train_loader, eval_loader
