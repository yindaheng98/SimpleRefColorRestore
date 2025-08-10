import os
from abc import abstractmethod
from typing import NamedTuple, Tuple
from tqdm import tqdm

import torch
import torchvision

from gaussian_splatting.camera import Camera


class DualCameraDataset:

    @abstractmethod
    def to(self, device) -> 'DualCameraDataset':
        return self

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[Camera, Camera]:
        raise NotImplementedError


class RestorationTuple(NamedTuple):
    color_distorted: torch.FloatTensor
    reference: torch.FloatTensor


class RestorationDataset:

    @abstractmethod
    def to(self, device) -> 'DualCameraDataset':
        return self

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[RestorationTuple, torch.FloatTensor]:
        raise NotImplementedError

    def save_image_tuple(self, idx, image_dir):
        restoration, gt = self[idx]
        torchvision.utils.save_image(restoration.color_distorted, os.path.join(image_dir, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(restoration.reference, os.path.join(image_dir, '{0:05d}'.format(idx) + ".lr.png"))
        torchvision.utils.save_image(gt, os.path.join(image_dir, '{0:05d}'.format(idx) + ".gt.png"))

    def save_dataset(self, image_dir):
        os.makedirs(image_dir, exist_ok=True)
        for idx in tqdm(range(len(self)), desc="Saving images"):
            self.save_image_tuple(idx, image_dir)
