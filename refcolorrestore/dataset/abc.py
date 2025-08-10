from abc import abstractmethod
from typing import Tuple

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
