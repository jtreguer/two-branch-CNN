# train_loop.py
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from sys import getsizeof
from tqdm import tqdm

from model import TwoBranchCNN
from data import AvirisDataset   

DEBUG = True

# ------------------------- CLI arguments ------------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Two-branch CNN training")
    # data / io
    p.add_argument("--image_path", type=str, default="/mnt/c/data/AVIRIS/",help="Root directory that contains AVIRIS scenes")    
    p.add_argument("--srf_path", type=str,default="srf/Landsat8_BGRI_SRF.xls",help="Spectral response file (xls)")
    p.add_argument('--image_number', type=int, default=4)
    p.add_argument("--results_dir", type=str, default="runs",help="Where to store checkpoints & logs")
    # training hyper-params
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-2) # 1e-3 for ADAM, 5e-2 for SGD
    p.add_argument('--optimizer', type=str, default='SGD')
    p.add_argument("--step_size", type=int, default=20, help="LR scheduler step")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="LR decay factor")
    p.add_argument("--seed", type=int, default=3407)
    # data specifics
    p.add_argument("--patch_size", type=int, default=31)
    p.add_argument("--stride", type=int, default=15) # 31 means no overlapping between patches
    p.add_argument("--training_ratio", type=float, default=0.7)
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--hsi_bands', type=int, default=224)   
    p.add_argument('--msi_bands', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--train_mode', type=bool, default=True)
    p.add_argument("--plot_freq", type=int, default=10, help="epochs between live plots")
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
        loss = criterion(pred, gt.squeeze(1)) # input, target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * gt.size(0)
    return epoch_loss / len(loader.dataset), pred

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    for gt, lr_hsi, hr_msi in loader:
        gt, lr_hsi, hr_msi = gt.to(device), lr_hsi.to(device), hr_msi.to(device)
        pred = model(lr_hsi, hr_msi)
        loss = criterion(pred, gt.squeeze(1)) # input, target
        val_loss += loss.item() * gt.size(0)
    return val_loss / len(loader.dataset)


# ---------------- visualisation helper ---------------- #
@torch.no_grad()
def sample_batch(dataset, device, k=32):
    """Return k random (lr_hsi, hr_msi) tensors already on device."""
    idx = torch.randperm(len(dataset))[:k]
    lr  = torch.stack([dataset.LRHSI_tensor_list[i] for i in idx]).to(device)
    hr  = torch.stack([dataset.HRMSI_tensor_list[i] for i in idx]).to(device)
    return lr, hr           # shapes: (k, 1, 224)  and  (k, 4, 31, 31)

def regularization_criterion(pred,gt):
    return 0.1 * torch.acos(F.cosine_similarity(pred, gt, dim=1)).mean()

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    assert device.type == 'cuda', "Warning: CUDA not being used. Check torch installation and availability of GPU."
    torch.backends.cudnn.benchmark = True

    # dataset & loaders -------------------------------------------------------
    full_dataset = AvirisDataset(args, device=device)
    lr_vis, hr_vis = sample_batch(full_dataset, device, k=8)
    print(f" Full dataset size {len(full_dataset)}")
    num_tiles      = len(full_dataset.image_list)
    num_training   = full_dataset.training_images_number
    rng = np.random.default_rng(args.seed) # no seed for randomness
    train_tile_ids = set(rng.choice(num_tiles, num_training, replace=False).tolist())
    print(f"Number of images {num_tiles}, number of images for training {num_training}, training image ids {train_tile_ids}")
    # Save image indices selected for training
    with open("selected_for_training.pkl", "wb") as f:
        pickle.dump(train_tile_ids, f)

    # DEBUG Compute averge spectrum ###################################
    if DEBUG:
        avg = torch.stack([full_dataset.GT_tensor_list[i] for i in train_tile_ids]).mean(0)       # (1, 224)
        avg = avg.detach().cpu().numpy()   
    ###################################################################

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
    criterion  = nn.L1Loss() # + regularization_criterion()
    if args.optimizer == 'Adam':
        optimizer  = Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(),lr=args.lr,momentum=0.9)

    scheduler  = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # prepare output directory
    os.makedirs(args.results_dir, exist_ok=True)
    best_val = float("inf")

    hist_loss = []
    hist_val_loss = []
    history = {"hist_loss": hist_loss, "hist_val_loss": hist_val_loss}

    plt.ion()                            # interactive mode
    fig, ax = plt.subplots(figsize=(7,4))
    lines = [ax.plot(np.zeros(args.hsi_bands))[0] for _ in range(lr_vis.size(0))]
    ax.set_title("Predicted spectra during training")
    ax.set_xlabel("Band (#)")
    ax.set_ylabel("Reflectance (arb.)")
    plt.show(block=False)
    # training loop ----------------------------------------------------------
    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        train_loss, pred = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        # torch.cuda.empty_cache()
        val_loss   = validate(model, val_loader, criterion, device)
        # torch.cuda.empty_cache()
        history["hist_loss"].append(train_loss)
        history["hist_val_loss"].append(val_loss)

        # Visualize a few predicted spectra (could be replaced by sampling from pred from train_one_epoch)
        if DEBUG:
            if epoch % args.plot_freq == 0:
                model.eval()
                with torch.no_grad():
                    pred_lines = model(lr_vis, hr_vis).cpu().numpy()   # shape (k, 224)

                for i, line in enumerate(lines):
                    line.set_ydata(pred_lines[i])                     # reuse SAME line objects
                    line.set_label(f"sample {i} @ ep {epoch}")
                    line.set_alpha(0.6)
                ax.relim()
                ax.autoscale_view()
                ax.legend(loc="upper right", fontsize="x-small")
                fig.canvas.draw()
                fig.canvas.flush_events()
                model.train()
        
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
                                     f"best_epoch{epoch:03d}.pt")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": best_val
            }, ckpt_path)

        if epoch % 20 == 0 or epoch == args.epochs:
            for n,p in model.named_parameters():
                print("paremeters average", n, p.data.abs().mean())
            print({n: p.grad.abs().mean().item() for n,p in model.named_parameters() if 'conv' in n})

    with open("hist_loss.json", "w") as file:
        json.dump(history, file, indent=4)

    if DEBUG:
        with open("DEBUG_average_pred.pkl", "wb") as f:
            pickle.dump([avg, pred.detach().cpu().numpy()], f)

if __name__ == "__main__":
    main()


