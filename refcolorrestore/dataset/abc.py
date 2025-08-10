from abc import abstractmethod
from typing import NamedTuple, Tuple

import torch

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
