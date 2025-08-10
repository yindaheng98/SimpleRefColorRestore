import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torchvision
from gaussian_splatting.utils import psnr, ssim
from gaussian_splatting.utils.lpipsPyTorch import lpips

from refcolorrestore.shelf import model_dict
from refcolorrestore.prepare import load_dataset, build_model


def rendering(net: nn.Module, dataset: DataLoader, render_path: str, log_path: str, device: str, save_image: bool):
    net = net.eval().to(device)
    shutil.rmtree(render_path, ignore_errors=True)
    os.makedirs(render_path, exist_ok=True)
    pbar = tqdm(dataset, desc="Rendering progress")
    with open(log_path, "w") as f:
        f.write(f"frame,psnr,ssim,lpips\n")
    for idx, (tup, ground_truth) in enumerate(pbar):
        restore = net(tup.color_distorted.to(device), tup.reference.to(device))
        ground_truth = ground_truth.to(device)
        scores = {
            "PSNR": psnr(restore, ground_truth).mean().item(),
            "SSIM": ssim(restore, ground_truth).mean().item(),
            "LPIPS": lpips(restore, ground_truth).mean().item(),
        }
        pbar.set_postfix(scores)
        with open(log_path, "a") as f:
            f.write(f"{idx},{scores['PSNR']},{scores['SSIM']},{scores['LPIPS']}\n")
        if save_image:
            torchvision.utils.save_image(restore, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--model", choices=list(model_dict.keys()), default="dualresnet")
    parser.add_argument("--model_destination", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--save_image", action='store_true')
    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    loader = load_dataset(image_dir=args.source, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    model_path = os.path.join(args.model_destination, f"{args.model}.pth")
    model = build_model(model=args.model, load_path=model_path, **configs)
    with torch.no_grad():
        rendering(model, loader, os.path.join(args.destination, args.model), os.path.join(args.destination, args.model + '.csv'), device=args.device, save_image=args.save_image)
