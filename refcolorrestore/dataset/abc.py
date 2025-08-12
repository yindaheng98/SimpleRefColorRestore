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
    def to(self, device) -> 'RestorationDataset':
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


class SavedRestorationDataset(RestorationDataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def __len__(self) -> int:
        n = 0
        while os.path.exists(os.path.join(self.data_dir, '{0:05d}'.format(n) + ".png")) and \
                os.path.exists(os.path.join(self.data_dir, '{0:05d}'.format(n) + ".lr.png")) and \
                os.path.exists(os.path.join(self.data_dir, '{0:05d}'.format(n) + ".gt.png")):
            n += 1
        return n

    def __getitem__(self, idx) -> Tuple[RestorationTuple, torch.FloatTensor]:
        color_distorted = torchvision.io.read_image(os.path.join(self.data_dir, '{0:05d}'.format(idx) + ".png")) / 255.0
        reference = torchvision.io.read_image(os.path.join(self.data_dir, '{0:05d}'.format(idx) + ".lr.png")) / 255.0
        ground_truth = torchvision.io.read_image(os.path.join(self.data_dir, '{0:05d}'.format(idx) + ".gt.png")) / 255.0
        return RestorationTuple(color_distorted, reference), ground_truth
