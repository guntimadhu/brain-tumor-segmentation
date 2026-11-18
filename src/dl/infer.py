"""Inference helpers shared by training, evaluation and the Streamlit app."""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dl.models import build

IN_SIZE = 128
PROB_THRESHOLD = 0.5
MIN_COMPONENT_PX = 50
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


def prepare(images_u8):
    """uint8 (N,256,256) -> float tensor (N,1,128,128), z-scored per slice."""
    x = torch.from_numpy(np.asarray(images_u8, dtype=np.float32)).unsqueeze(1)
    x = F.interpolate(x, size=(IN_SIZE, IN_SIZE), mode="area")
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def postprocess(prob256):
    """Probability map (256x256) -> binary mask: threshold and drop components < MIN_COMPONENT_PX."""
    mask = (prob256 > PROB_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX:
            out |= labels == i
    return out


@torch.no_grad()
def predict_probs(model, images_u8, batch=32, mc_passes=0):
    """Return (mean probability, uncertainty std or None) at the original resolution."""
    model.eval()
    model.mc_dropout = mc_passes > 0
    h, w = np.asarray(images_u8).shape[1:]
    probs, stds = [], []
    x_all = prepare(images_u8)
    for i in range(0, len(x_all), batch):
        x = x_all[i:i + batch]
        if mc_passes > 0:
            ps = torch.stack([torch.sigmoid(model(x)) for _ in range(mc_passes)])
            p, s = ps.mean(0), ps.std(0)
            stds.append(F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)[:, 0].numpy())
        else:
            p = torch.sigmoid(model(x))
        probs.append(F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[:, 0].numpy())
    model.mc_dropout = False
    return np.concatenate(probs), (np.concatenate(stds) if stds else None)


def gradcam(model, image_u8):
    """Grad-CAM on the bottleneck features for the predicted abnormality (256x256 map in [0, 1])."""
    model.eval()
    model.mc_dropout = False
    x = prepare(image_u8[None]).requires_grad_(False)
    logits, feat = model(x, return_bottleneck=True)
    feat.retain_grad()
    prob = torch.sigmoid(logits)
    region = (prob > PROB_THRESHOLD).float()
    score = (logits * region).sum() if region.sum() > 0 else logits.max()
    model.zero_grad()
    score.backward()
    weights = feat.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image_u8.shape, mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def load(name, seed=42, path=None):
    model = build(name)
    path = path or os.path.join(MODEL_DIR, f"{name}_seed{seed}.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model
