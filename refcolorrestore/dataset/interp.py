from typing import Tuple
import torch

from gaussian_splatting import Camera
from extrinterp import ExtrinsicInterpolator

from .abc import DualCameraDataset


class DualExtrinsicInterpolationDataset(DualCameraDataset):
    def __init__(
        self,
            base_dataset: ExtrinsicInterpolator,
            FoVx: float = 90.0*torch.pi/180, FoVy: float = 90.0*torch.pi/180,
            image_height: int = 1000, image_width: int = 1000, downsample=4):
        self.cameras = base_dataset
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.downsample = downsample
        self.image_height = image_height // 4 * 4
        self.image_width = image_width // 4 * 4

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Tuple[Camera, Camera]:
        extr = self.cameras[idx]
        camera_hr = extr.to_camera(
            image_height=self.image_height,
            image_width=self.image_width,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            device=self.cameras[idx].R.device
        )
        camera_lr = extr.to_camera(
            image_height=self.image_height // self.downsample,
            image_width=self.image_width // self.downsample,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            device=self.cameras[idx].R.device
        )
        return camera_lr, camera_hr

    def to(self, device):
        self.cameras = self.cameras.to(device)
        return self
