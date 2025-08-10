import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from refcolorrestore.dataset import SavedRestorationDataset
from refcolorrestore.shelf import model_dict


def load_dataset(
        image_dir: str,
        batch_size: int, num_workers: int, shuffle=False) -> DataLoader:
    dataset = SavedRestorationDataset(data_dir=image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=False
    )
    return loader


def build_model(model: str, load_path: str = None, **kwargs) -> nn.Module:
    net = model_dict[model](**kwargs)
    if load_path:
        net.load_state_dict(torch.load(load_path))
    return net
