from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_oxford_pets37_datasets(
    data_dir: str = "./data",
    img_size: int = 96,
    val_split: float = 0.0,
    seed: int = 7,
    target_type: str = "category"):
    
    """
    Oxford-IIIT Pet (37 clases) desde torchvision.
    Split oficial:
      - trainval
      - test

    Preproceso (sin augs):
      Resize(96x96) + ToTensor + Normalize(ImageNet)
    """
    ops = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    transform = transforms.Compose(ops)

    # train/val (oficial "trainval")
    trainval_ds = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types=target_type,
        download=True,
        transform=transform)

    # test oficial
    test_ds = datasets.OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types=target_type,
        download=True,
        transform=transform)

    if val_split and val_split > 0.0:
        n_total = len(trainval_ds)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        g_split = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(trainval_ds, [n_train, n_val], generator=g_split)
    else:
        train_ds, val_ds = trainval_ds, None

    return train_ds, val_ds, test_ds

def get_oxford_pets37_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    img_size: int = 96,
    val_split: float = 0.0,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 7,
    target_type: str = "category") -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:

    train_ds, val_ds, test_ds = get_oxford_pets37_datasets(
        data_dir=data_dir,
        img_size=img_size,
        val_split=val_split,
        seed=seed,
        target_type=target_type)

    g_loader = torch.Generator().manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g_loader)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **common)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common)

    return train_loader, val_loader, test_loader