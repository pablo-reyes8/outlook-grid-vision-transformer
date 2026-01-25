import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment


def _seed_worker_factory(seed: int):
    """
    Crea un worker_init_fn que fija seeds de python/numpy/torch en cada worker
    de forma determinística a partir de `seed`.
    """
    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker


class SVHNLabelFix(Dataset):
    """
    SVHN original labels sometimes use 10 to represent digit '0'.
    This wrapper normalizes labels to {0..9} safely.
    """
    def __init__(self, base_ds: Dataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = int(y)
        if y == 10:
            y = 0
        return x, y


def get_svhn_datasets(
    data_dir: str = "./data",
    val_split: float = 0.0,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
    img_size: int = 32,
    seed: int = 7,):
    """
    SVHN 32x32 con LAS MISMAS transforms que tu CIFAR-100 pipeline:
      Resize (si img_size != 32)
      RandomCrop + padding
      RandomHorizontalFlip
      RandAugment
      ToTensor + Normalize
      RandomErasing
    """
    if img_size < 32:
        raise ValueError(f"img_size must be >= 32 for SVHN. Got {img_size}.")

    # Stats comunes para SVHN (RGB)
    svhn_mean = (0.4377, 0.4438, 0.4728)
    svhn_std  = (0.1980, 0.2010, 0.1970)

    crop_padding = max(4, img_size // 8)

    train_ops = []
    if img_size != 32:
        train_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))

    train_ops += [
        transforms.RandomCrop(img_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(svhn_mean, svhn_std),
        transforms.RandomErasing(
            p=random_erasing_p,
            scale=erasing_scale,
            ratio=erasing_ratio,
            value="random",),]

    train_transform = transforms.Compose(train_ops)

    test_ops = []
    if img_size != 32:
        test_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))

    test_ops += [
        transforms.ToTensor(),
        transforms.Normalize(svhn_mean, svhn_std),]

    test_transform = transforms.Compose(test_ops)

    full_train = datasets.SVHN(
        root=data_dir,
        split="train",
        download=True,
        transform=train_transform,)
    
    test_ds = datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=test_transform,)

    full_train = SVHNLabelFix(full_train)
    test_ds = SVHNLabelFix(test_ds)

    if val_split > 0.0:
        n_total = len(full_train)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            full_train,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),)
    else:
        train_ds = full_train
        val_ds = None

    return train_ds, val_ds, test_ds


def get_svhn_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    val_split: float = 0.0,
    pin_memory: bool = True,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    img_size: int = 32,
    seed: int = 7,):
    """
    Devuelve (train_loader, val_loader, test_loader) para SVHN,
    con LAS MISMAS transforms del pipeline CIFAR-100.
    """
    train_ds, val_ds, test_ds = get_svhn_datasets(
        data_dir=data_dir,
        val_split=val_split,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude,
        random_erasing_p=random_erasing_p,
        img_size=img_size,
        seed=seed,)

    dl_gen = torch.Generator().manual_seed(seed)
    worker_init_fn = _seed_worker_factory(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=dl_gen,
        worker_init_fn=worker_init_fn,)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            generator=dl_gen,
            worker_init_fn=worker_init_fn,)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=dl_gen,
        worker_init_fn=worker_init_fn,)

    return train_loader, val_loader, test_loader

