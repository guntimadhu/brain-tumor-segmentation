"""Figures for the deep-learning comparison (run after evaluate_dl.py).

Usage: python experiments/make_dl_figures.py
"""
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import pipeline as P  # noqa: E402
from src.dl.infer import load, gradcam, postprocess  # noqa: E402
from src.evaluation import slice_metrics  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
OUT = os.path.join(ROOT, "results", "dl", "figures")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(ROOT, "results", "dl", "summary_dl.json")))
cache = np.load(os.path.join(ROOT, "data", "dl_cache.npz"))
images, masks, patient, slice_no = cache["images"], cache["masks"], cache["patient"], cache["slice_number"]
split = json.load(open(os.path.join(ROOT, "experiments", "dl_split.json")))
test_idx = np.where(np.isin(patient, split["test"]))[0]
LABEL = {"classical_percentile": "Percen-\ntile", "classical_otsu": "Otsu",
         "classical_adaptive": "Adap-\ntive", "unet_seed42": "U-Net", "attention_unet_seed42": "Att.\nU-Net",
         "attention_unet_mc_seed42": "Att. U-Net\n+ MC"}

# 1. training curves
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
for name, lab, col in [("unet", "U-Net", "#1f77b4"), ("attention_unet", "Attention U-Net", "#d62728")]:
    p = os.path.join(ROOT, "results", "dl", f"{name}_seed42_log.csv")
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p)))
    ep = [int(r["epoch"]) for r in rows]
    axes[0].plot(ep, [float(r["train_loss"]) for r in rows], color=col, lw=1.2, label=lab)
    axes[1].plot(ep, [float(r["val_patient_dice"]) for r in rows], color=col, lw=1.2, label=lab)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Dice"); axes[1].set_ylim(0, 1)
for a in axes:
    a.grid(alpha=0.3); a.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dl_training_curves.png"), dpi=300)
plt.close(fig)

# 2. patient-level Dice distribution on the 28 test patients
per = {}
for r in csv.DictReader(open(os.path.join(ROOT, "results", "dl", "per_slice_dl.csv"))):
    if r["positive"] == "1" and r["dice"] not in ("", "None"):
        per.setdefault(r["method"], {}).setdefault(r["patient"], []).append(float(r["dice"]))
keys = [k for k in LABEL if k in per]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.boxplot([[np.mean(v) for v in per[k].values()] for k in keys], widths=0.5, flierprops=dict(markersize=2))
ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([LABEL[k] for k in keys], fontsize=6.5)
ax.set_ylabel("Patient-level mean Dice"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dl_test_boxplot.png"), dpi=300)
plt.close(fig)


# 3. qualitative comparison + uncertainty + Grad-CAM on three test slices
def ov(img, pred, gt):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if pred.any():
        rgb[pred] = (0.45 * rgb[pred] + 0.55 * np.array([0, 200, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cs, -1, (255, 0, 0), 1)
    return rgb


pa = np.load(os.path.join(ROOT, "results", "dl", "pred_attention_unet_mc_seed42.npz"))
pu = np.load(os.path.join(ROOT, "results", "dl", "pred_unet_seed42.npz"))
pr = np.load(os.path.join(ROOT, "results", "dl", "pred_attention_unet_seed42.npz"))
model = load("attention_unet", 42)
pos = [k for k, i in enumerate(test_idx) if masks[i].sum() > 1500]
rng = np.random.RandomState(0)
dices = [(slice_metrics(postprocess(pr["probs"][k].astype(np.float32)), masks[test_idx[k]])["dice"], k) for k in pos]
dices.sort()
picks = [dices[len(dices) // 2][1], dices[int(len(dices) * 0.85)][1], dices[int(len(dices) * 0.2)][1]]
sources = []
fig1, ax1 = plt.subplots(3, 3, figsize=(3.5, 3.6))
fig2, ax2 = plt.subplots(3, 3, figsize=(3.5, 3.6))
for r, k in enumerate(picks):
    i = test_idx[k]
    img, gt = images[i], masks[i]
    classical = P.run_pipeline(img, "percentile")["final"] > 0
    unet = postprocess(pu["probs"][k].astype(np.float32))
    att = postprocess(pr["probs"][k].astype(np.float32))
    unc = pa["std"][k].astype(np.float32)
    cam = gradcam(model, img)
    d = {n: slice_metrics(m, gt)["dice"] for n, m in [("classical", classical), ("unet", unet), ("att", att)]}
    for c, (im, t) in enumerate([(ov(img, classical, gt), f"Percentile {d['classical']:.2f}"),
                                 (ov(img, unet, gt), f"U-Net {d['unet']:.2f}"), (ov(img, att, gt), f"Att. U-Net {d['att']:.2f}")]):
        ax1[r, c].imshow(im); ax1[r, c].set_title(t, fontsize=7); ax1[r, c].axis("off")
    ax2[r, 0].imshow(ov(img, att, gt)); ax2[r, 0].set_title(f"Att. U-Net {d['att']:.2f}", fontsize=7)
    ax2[r, 1].imshow(img, cmap="gray"); ax2[r, 1].imshow(unc, cmap="turbo", alpha=0.6, vmin=0, vmax=0.25)
    ax2[r, 1].set_title("Uncertainty", fontsize=7)
    ax2[r, 2].imshow(img, cmap="gray"); ax2[r, 2].imshow(cam, cmap="turbo", alpha=0.5, vmin=0, vmax=1)
    ax2[r, 2].set_title("Grad-CAM", fontsize=7)
    for a in ax2[r]:
        a.axis("off")
    sources.append({"patient": str(patient[i]), "slice": int(slice_no[i]), **{k2: float(v) for k2, v in d.items()}})
for f, name in [(fig1, "dl_examples.png"), (fig2, "dl_explain.png")]:
    f.tight_layout(pad=0.2)
    f.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(f)
json.dump(sources, open(os.path.join(OUT, "dl_examples.json"), "w"), indent=1)
print("figures saved to", OUT, sources)
