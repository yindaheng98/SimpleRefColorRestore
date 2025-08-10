import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from refcolorrestore.shelf import model_dict
from refcolorrestore.prepare import load_dataset, build_model


def training(net: nn.Module, epoch: int, loader: DataLoader, device: str):
    net = torch.nn.DataParallel(net.train()).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, weight_decay=0, betas=[0.9, 0.99])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)
    loss_fn = torch.nn.MSELoss()
    for e in range(epoch):
        loss_sum = 0
        pbar = tqdm(loader)
        for i, (tup, ground_truth) in enumerate(pbar):
            optimizer.zero_grad()
            restore = net(tup.color_distorted.to(device), tup.reference.to(device))
            loss = loss_fn(restore, ground_truth.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.set_description(f"epoch {e} loss=%.6f" % (loss_sum / (i + 1)))
        scheduler.step()
    return net.module


def save_model(model: nn.Module, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--model", choices=list(model_dict.keys()), default="dualresnet")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epoch", default=30, type=int)
    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    loader = load_dataset(image_dir=args.source, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    model = build_model(model=args.model, **configs)
    model = training(net=model, epoch=args.epoch, loader=loader, device=args.device)
    model_path = os.path.join(args.destination, f"{args.model}.pth")
    model = save_model(model=model, save_path=model_path)
