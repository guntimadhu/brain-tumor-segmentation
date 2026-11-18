"""Train a 2D U-Net or Attention U-Net on the training patients of experiments/dl_split.json.

Usage: python experiments/train_dl.py --model attention_unet --seed 42 [--epochs 50]
Loss: soft Dice + binary cross-entropy. Optimiser: Adam with cosine learning-rate schedule.
Each epoch uses every training slice with an annotated abnormality plus an equal number of
randomly drawn empty slices. Early stopping on the validation (development) patients'
patient-level mean Dice.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.dl.models import build  # noqa: E402
from src.dl.infer import prepare, predict_probs, postprocess, IN_SIZE  # noqa: E402
from src.evaluation import slice_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["unet", "attention_unet"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

torch.set_num_threads(int(os.environ.get("DL_THREADS", "10")))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

cache = np.load(os.path.join(ROOT, "data", "dl_cache.npz"))
images, masks, patient = cache["images"], cache["masks"], cache["patient"]
split = json.load(open(os.path.join(ROOT, "experiments", "dl_split.json")))
idx = {k: np.where(np.isin(patient, split[k]))[0] for k in ("train", "val", "test")}

x_train = prepare(images[idx["train"]])
y_train = F.interpolate(torch.from_numpy(masks[idx["train"]].astype(np.float32)).unsqueeze(1),
                        size=(IN_SIZE, IN_SIZE), mode="area")
pos = np.where(masks[idx["train"]].any(axis=(1, 2)))[0]
neg = np.where(~masks[idx["train"]].any(axis=(1, 2)))[0]
print(f"train slices {len(idx['train'])} (positive {len(pos)}), val slices {len(idx['val'])}", flush=True)


def augment(x, y):
    n = x.shape[0]
    flip = torch.rand(n) < 0.5
    x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
    ang = (torch.rand(n) * 30 - 15) * math.pi / 180
    sc = 1 + (torch.rand(n) * 0.2 - 0.1)
    tx, ty = (torch.rand(n) * 0.1 - 0.05), (torch.rand(n) * 0.1 - 0.05)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = torch.cos(ang) / sc, -torch.sin(ang) / sc, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = torch.sin(ang) / sc, torch.cos(ang) / sc, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * (1 + (torch.rand(n, 1, 1, 1) * 0.2 - 0.1)) + torch.randn_like(x) * 0.05
    return x, y


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


def val_dice(model):
    probs, _ = predict_probs(model, images[idx["val"]])
    per = {}
    for k, i in enumerate(idx["val"]):
        gt = masks[i]
        if gt.any():
            per.setdefault(patient[i], []).append(slice_metrics(postprocess(probs[k]), gt)["dice"])
    return float(np.mean([np.mean(v) for v in per.values()]))


model = build(args.model).to(memory_format=torch.channels_last)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results", "dl"), exist_ok=True)
ckpt = os.path.join(ROOT, "models", f"{args.model}_seed{args.seed}.pt")
log_path = os.path.join(ROOT, "results", "dl", f"{args.model}_seed{args.seed}_log.csv")
log_file = open(log_path, "w", newline="")
log = csv.writer(log_file)
log.writerow(["epoch", "train_loss", "val_patient_dice", "lr", "seconds"])
best, best_epoch, wait = -1.0, 0, 0
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    sel = np.concatenate([pos, np.random.choice(neg, size=min(len(pos), len(neg)), replace=False)])
    np.random.shuffle(sel)
    losses = []
    for b in range(0, len(sel), args.batch):
        bi = torch.from_numpy(sel[b:b + args.batch])
        x, y = augment(x_train[bi].clone(), y_train[bi].clone())
        logits = model(x.contiguous(memory_format=torch.channels_last))
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    sched.step()
    vd = val_dice(model)
    secs = time.time() - t0
    log.writerow([epoch, f"{np.mean(losses):.4f}", f"{vd:.4f}", f"{sched.get_last_lr()[0]:.6f}", f"{secs:.1f}"])
    log_file.flush()
    print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val patient Dice {vd:.4f}  ({secs:.0f}s)", flush=True)
    if vd > best:
        best, best_epoch, wait = vd, epoch, 0
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_patient_dice": vd,
                    "config": vars(args)}, ckpt)
    else:
        wait += 1
        if wait >= args.patience:
            print("early stop", flush=True)
            break
print(f"best val patient Dice {best:.4f} at epoch {best_epoch}; saved {ckpt}", flush=True)
