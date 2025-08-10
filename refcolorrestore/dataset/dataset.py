import os
from typing import Tuple

import torch
import torchvision
import tqdm

from gaussian_splatting import GaussianModel

from .abc import DualCameraDataset, RestorationDataset, RestorationTuple


class DualCameraRestorationDataset:
    def __init__(self, cameras: DualCameraDataset, color_distorted_gaussians: GaussianModel, ground_truth_gaussians: GaussianModel):
        self.cameras = cameras
        self.ground_truth_gaussians = ground_truth_gaussians
        self.color_distorted_gaussians = color_distorted_gaussians

    def to(self, device) -> 'RestorationDataset':
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

    def save_image_tuple(self, idx, image_dir):
        restoration, gt = self[idx]
        torchvision.utils.save_image(restoration.color_distorted, os.path.join(image_dir, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(restoration.reference, os.path.join(image_dir, '{0:05d}'.format(idx) + ".lr.png"))
        torchvision.utils.save_image(gt, os.path.join(image_dir, '{0:05d}'.format(idx) + ".gt.png"))

    def save_dataset(self, image_dir):
        os.makedirs(image_dir, exist_ok=True)
        for idx in tqdm.tqdm(range(len(self.cameras)), desc="Saving images"):
            self.save_image_tuple(idx, image_dir)
