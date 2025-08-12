from typing import Tuple
import torch
import os
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset, prepare_gaussians
from extrinterp import ExtrinsicInterpolator
from refcolorrestore.dataset import Extrinsic2DualCameraDataset, DualCamera2RestorationDataset


def prepare_rendering(
        sh_degree: int, source: str, device: str, n: int, window_size: int,
        trainable_camera: bool = False,
        load_ply: str = None, load_ply_gt: str = None,
        load_camera: str = None,
        use_intrinsics: int | dict = 0
) -> Tuple[CameraDataset, GaussianModel]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_depth=False)
    if isinstance(use_intrinsics, int):
        i = use_intrinsics
        use_intrinsics = dict(
            image_height=dataset[i].image_height, image_width=dataset[i].image_width,
            FoVx=dataset[i].FoVx, FoVy=dataset[i].FoVy)
    elif not isinstance(use_intrinsics, dict):
        raise ValueError("Invalid use_intrinsics format")
    dataset = Extrinsic2DualCameraDataset(ExtrinsicInterpolator(dataset=dataset, n=n, window_size=window_size), **use_intrinsics)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply)
    gaussians_gt = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply_gt)
    return dataset, gaussians, gaussians_gt


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--destination_gt", required=True, type=str)
    parser.add_argument("--iteration_gt", required=True, type=int)
    parser.add_argument("--interp_n", required=True, type=int)
    parser.add_argument("--interp_window_size", type=int, default=3)
    parser.add_argument("--use_intrinsics", type=str, default="0", help="Use intrinsics for rendering, can be an integer index or a dict with keys: image_height, image_width, FoVx, FoVy")
    parser.add_argument("--downsample", default=4, type=int)
    parser.add_argument("--data_dir", required=True, type=str)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    load_ply_gt = os.path.join(args.destination_gt, "point_cloud", "iteration_" + str(args.iteration_gt), "point_cloud.ply")
    with torch.no_grad():
        cameras, gaussians, gaussians_gt = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device,
            n=args.interp_n, window_size=args.interp_window_size,
            trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_ply_gt=load_ply_gt, load_camera=args.load_camera,
            use_intrinsics=eval(args.use_intrinsics)
        )
        dataset = DualCamera2RestorationDataset(cameras=cameras, color_distorted_gaussians=gaussians, ground_truth_gaussians=gaussians_gt)
        os.makedirs(args.data_dir, exist_ok=True)
        dataset.save_dataset(args.data_dir)
