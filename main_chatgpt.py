# train_loop.py
import argparse
import json
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import TwoBranchCNN
from data import AvirisDataset                 

# ------------------------- CLI arguments ------------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Two-branch CNN training")
    # data / io
    p.add_argument("--image_path", type=str, default="/mnt/c/data/AVIRIS/",help="Root directory that contains AVIRIS scenes")    
    p.add_argument("--srf_path", type=str,default="srf/Landsat8_BGRI_SRF.xls",help="Spectral response file (xls)")
    p.add_argument('--image_number', type=int, default=4)
    p.add_argument("--results_dir", type=str, default="runs",help="Where to store checkpoints & logs")
    # training hyper-params
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--step_size", type=int, default=100,
                   help="LR scheduler step")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="LR decay factor")
    p.add_argument("--seed", type=int, default=3407)
    # data specifics
    p.add_argument("--patch_size", type=int, default=31)
    p.add_argument("--stride", type=int, default=31)
    p.add_argument("--training_ratio", type=float, default=0.7)
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--hsi_bands', type=int, default=224)   
    p.add_argument('--msi_bands', type=int, default=4)  
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--optimizer', type=str, default='Adam')
    p.add_argument('--train_mode', type=bool, default=True)
    return p.parse_args()

# ------------------------- utilities ------------------------- #
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

# ------------------------- main train loop ------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for gt, lr_hsi, hr_msi in loader:
        gt, lr_hsi, hr_msi = gt.to(device), lr_hsi.to(device), hr_msi.to(device)
        pred = model(lr_hsi, hr_msi)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * gt.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    for gt, lr_hsi, hr_msi in loader:
        gt, lr_hsi, hr_msi = gt.to(device), lr_hsi.to(device), hr_msi.to(device)
        pred = model(lr_hsi, hr_msi)
        loss = criterion(pred, gt)
        val_loss += loss.item() * gt.size(0)
    return val_loss / len(loader.dataset)

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & loaders -------------------------------------------------------
    full_dataset = AvirisDataset(args, device=device)
    print(f" Full dataset size {len(full_dataset)}")
    num_tiles      = len(full_dataset.image_list)
    num_training   = full_dataset.training_images_number
    rng = np.random.default_rng(args.seed)
    train_tile_ids = set(rng.choice(num_tiles, num_training, replace=False).tolist())
    # Save image indices selected for training
    with open("selected_for_training.pkl", "wb") as f:
        pickle.dump(train_tile_ids, f)

    train_idx, val_idx = [], []
    for sample_idx, tile_id in enumerate(full_dataset.image_ids):
        if tile_id in train_tile_ids:
            train_idx.append(sample_idx)
        else:
            val_idx.append(sample_idx)
    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          shuffle=True,            # shuffle *within* the tile set
                          drop_last=True,
                          num_workers=args.num_workers,
                          pin_memory=True)

    val_loader   = DataLoader(val_set,
                          batch_size=args.batch_size * 2,
                          shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=True)


    # model, optimiser, scheduler -------------------------------------------
    model = TwoBranchCNN(hsi_bands=args.hsi_bands, msi_bands=args.msi_bands,
                         patch_size=args.patch_size).to(device)
    criterion  = nn.L1Loss()
    if args.optimizer == 'Adam':
        optimizer  = Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters,lr=args.lr,momentum=0.9)
    scheduler  = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # prepare output directory
    os.makedirs(args.results_dir, exist_ok=True)
    best_val = float("inf")

    hist_loss = []
    hist_val_loss = []
    history = {"hist_loss": hist_loss, "hist_val_loss": hist_val_loss}
    # training loop ----------------------------------------------------------
    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)
        history["hist_loss"].append(train_loss)
        history["hist_val_loss"].append(val_loss)
        scheduler.step()

        # logging
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train={train_loss:.4f}, val={val_loss:.4f}, "
              f"lr={lr_now:.2e}, time={elapsed:.1f}s")

        # save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.results_dir,
                                     f"best_epoch{epoch:03d}_{val_loss:.4f}.pt")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": best_val
            }, ckpt_path)

    with open("hist_loss.json", "w") as file:
        json.dump(history, file, indent=4)

if __name__ == "__main__":
    main()
