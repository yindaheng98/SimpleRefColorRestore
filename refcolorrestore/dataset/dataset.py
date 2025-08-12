from typing import Tuple

import torch

from gaussian_splatting import GaussianModel

from .abc import DualCameraDataset, RestorationDataset, RestorationTuple


class DualCamera2RestorationDataset(RestorationDataset):
    def __init__(self, cameras: DualCameraDataset, color_distorted_gaussians: GaussianModel, ground_truth_gaussians: GaussianModel):
        self.cameras = cameras
        self.ground_truth_gaussians = ground_truth_gaussians
        self.color_distorted_gaussians = color_distorted_gaussians

    def to(self, device) -> 'DualCamera2RestorationDataset':
        self.cameras = self.cameras.to(device)
        self.ground_truth_gaussians = self.ground_truth_gaussians.to(device)
        self.color_distorted_gaussians = self.color_distorted_gaussians.to(device)
        return self

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx) -> Tuple[RestorationTuple, torch.FloatTensor]:
        lr_camera, hr_camera = self.cameras[idx]
        color_distorted = self.color_distorted_gaussians(hr_camera)['render']
        ground_truth = self.ground_truth_gaussians(hr_camera)['render']
        reference = self.ground_truth_gaussians(lr_camera)['render']
        return RestorationTuple(color_distorted, reference), ground_truth
