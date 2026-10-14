"""Regenerate all paper figures from the frozen pipeline and the saved results.

Usage:
    python experiments/make_figures.py --data <path to kaggle_3m> --results results
Each example figure is recomputed with run_pipeline and its Dice is checked
against results/per_slice_main.csv before saving.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import pipeline as P  # noqa: E402
from src.evaluation import slice_metrics  # noqa: E402
from src.morphology import get_morphology_comparison  # noqa: E402
from src.segmentation import segment_tumor_candidate, apply_skull_strip_approximation  # noqa: E402
from evaluate import load_image, load_mask  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9})
METHOD_SHORT = {"percentile": "Percentile (P75)", "otsu": "Otsu", "adaptive": "Adaptive Gaussian"}


def paths(data, patient, n):
    stem = f"{patient}_{n}"
    return os.path.join(data, patient, stem + ".tif"), os.path.join(data, patient, stem + "_mask.tif")


def panel_grid(images, titles, out, header=None, cols=None, size=1.7):
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows + (0.3 if header else 0)))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if header:
        fig.suptitle(header, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def overlay(img, pred, gt):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    out = rgb.copy()
    out[pred] = (0.5 * out[pred] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (255, 0, 0), 1)
    return out


def block_diagram(out):
    row1 = ["LGG TIFF\n(3 channels)", "FLAIR\n(channel 1)", "Min-max\nnormalise", "Gaussian\nσ = 1.0",
            "CLAHE\nclip 2.0"]
    row2 = ["Threshold\nP75 / Otsu /\nadaptive", "Open, then\nclose (5×5)", "Keep 100 px\nto 25%;\nlargest",
            "Mask and\npixel count"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.02", fc="#e8eef8", ec="#2b4c7e", lw=0.7)

    def draw(x, y, s, w=0.88, h=0.8, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, **{**box, **kw}))
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=5.6)

    arrow = dict(arrowstyle="->", lw=0.7)
    for i, s in enumerate(row1):
        draw(i + 0.06, 2.2, s)
        if i < len(row1) - 1:
            ax.annotate("", xy=(i + 1.06, 2.6), xytext=(i + 0.94, 2.6), arrowprops=arrow)
    for i, s in enumerate(row2):
        x = 4.06 - i
        draw(x, 0.1, s)
        if i < len(row2) - 1:
            ax.annotate("", xy=(x - 0.12, 0.5), xytext=(x, 0.5), arrowprops=arrow)
    draw(1.06, 1.15, "Brain mask from the\nnormalised image: threshold\n15, largest component,\nhole fill, 9×9 erosion",
         w=1.9, h=0.8, fc="#fdf1e3", ec="#a0601a")
    ax.annotate("", xy=(4.5, 0.9), xytext=(4.5, 2.2), arrowprops=arrow)
    ax.annotate("", xy=(2.5, 1.95), xytext=(2.5, 2.2), arrowprops=dict(arrowstyle="->", lw=0.7, color="#a0601a"))
    ax.annotate("", xy=(4.06, 0.75), xytext=(2.96, 1.4), arrowprops=dict(arrowstyle="->", lw=0.7, color="#a0601a"))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    summary = json.load(open(os.path.join(args.results, "summary.json")))
    sel = summary["percentile_selection"]["selected"]
    assert sel == P.METHOD_PARAMS["percentile"]["percentile"], "pipeline default differs from selected percentile"
    ver = P.PIPELINE_VERSION
    rows = list(csv.DictReader(open(os.path.join(args.results, "per_slice_main.csv"))))
    main = [r for r in rows if r["method"] == "percentile" and r["morphology"] == "1"]
    captions = {}

    block_diagram(os.path.join(figdir, "fig01_pipeline.png"))

    # Illustration slice: development patient with a clearly annotated abnormality
    ill_patient, ill_n = "TCGA_DU_6401_19831001", None
    cand = [r for r in main if r["patient"] == ill_patient and r["positive"] == "1"]
    ill = sorted(cand, key=lambda r: -int(r["gt_px"]))[0]
    ill_n = int(ill["slice_number"])
    ipath, mpath = paths(args.data, ill_patient, ill_n)
    gt = load_mask(mpath)
    chans = [load_image(ipath, c) for c in ["pre_contrast", "flair", "post_contrast"]]
    chans_ov = [overlay(c, np.zeros_like(gt), gt) for c in chans]
    panel_grid(chans_ov, ["(a) Pre-contrast (ch. 0)", "(b) FLAIR (ch. 1)", "(c) Post-contrast (ch. 2)"],
               os.path.join(figdir, "fig02_channels.png"), cols=3, size=2.0,
               header=f"{ill_patient}, slice {ill_n}; red = reference contour")
    captions["fig02"] = f"{ill_patient} slice {ill_n}"

    img = chans[1]
    res = P.run_pipeline(img, "percentile")
    st = res["pp_steps"]
    panel_grid([st["original"], st["normalized"], st["filtered"], st["enhanced"]],
               ["(a) FLAIR input", "(b) Normalised", "(c) Gaussian", "(d) CLAHE"],
               os.path.join(figdir, "fig03_preprocessing.png"), cols=4, size=1.6)

    brain_only = cv2.bitwise_and(res["enhanced"], res["enhanced"], mask=res["brain_mask"])
    old_mask = apply_skull_strip_approximation(res["enhanced"])
    panel_grid([st["normalized"], res["brain_mask"], brain_only, old_mask],
               ["(a) Normalised", "(b) Brain mask", "(c) CLAHE in mask", "(d) Mask from CLAHE"],
               os.path.join(figdir, "fig04_brain_mask.png"), cols=4, size=1.6)
    captions["fig04_mask_fraction"] = {"normalised": float((res["brain_mask"] > 0).mean()),
                                       "clahe": float((old_mask > 0).mean())}

    cands = {m: segment_tumor_candidate(res["enhanced"], m, P.METHOD_PARAMS[m], brain_mask=res["brain_mask"])
             for m in ["percentile", "otsu", "adaptive"]}
    panel_grid([cands["percentile"]["binary_mask"], cands["otsu"]["binary_mask"],
                cands["adaptive"]["binary_mask"], (gt * 255).astype(np.uint8)],
               ["(a) Percentile P75", "(b) Otsu", "(c) Adaptive Gaussian", "(d) Reference"],
               os.path.join(figdir, "fig05_methods.png"), cols=4, size=1.6,
               header=f"{ill_patient}, slice {ill_n}; candidate masks before morphology")
    captions["fig05_thresholds"] = {m: float(c["threshold_value"]) if c["threshold_value"] is not None else None
                                    for m, c in cands.items()}

    mc = get_morphology_comparison(cands["percentile"]["binary_mask"])
    panel_grid([mc["original"], mc["erosion"], mc["dilation"], mc["opening"], mc["closing"], mc["opening_then_closing"]],
               ["(a) Candidate", "(b) Erosion", "(c) Dilation", "(d) Opening", "(e) Closing", "(f) Open → close"],
               os.path.join(figdir, "fig06_morphology.png"), cols=3, size=1.6)

    panel_grid([res["morph"]["output_mask"], res["filtered"], res["final"], overlay(img, res["final"] > 0, gt)],
               ["(a) After morphology", "(b) Size-filtered", "(c) Final mask", "(d) Overlay"],
               os.path.join(figdir, "fig07_postprocessing.png"), cols=4, size=1.6)

    # Development percentile sweep
    sw = summary["percentile_selection"]["results"]
    qs = sorted(int(q) for q in sw)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.errorbar(qs, [sw[str(q)]["patient_mean_dice"] for q in qs], yerr=[sw[str(q)]["patient_sd_dice"] for q in qs],
                marker="o", ms=3, capsize=2, lw=1, label="Patient-level mean ± SD")
    ax.plot(qs, [sw[str(q)]["pos_slice_mean_dice"] for q in qs], marker="s", ms=3, lw=1, ls="--",
            label="Positive-slice mean")
    ax.axvline(sel, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Percentile threshold")
    ax.set_ylabel("Dice (development cohort)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig08_percentile_sweep.png"), dpi=300)
    plt.close(fig)

    # Held-out patient-level Dice distribution
    pp = list(csv.DictReader(open(os.path.join(args.results, "per_patient_main.csv"))))
    data, labels = [], []
    for m in ["percentile", "otsu", "adaptive"]:
        for mo, tag in [("morph", "+M"), ("nomorph", "−M")]:
            data.append([float(r["mean_dice"]) for r in pp if r["split"] == "test" and r["method"] == m
                         and r["morphology"] == mo and r["mean_dice"]])
            labels.append(f"{ {'percentile': 'P75', 'otsu': 'Otsu', 'adaptive': 'Adapt.'}[m] }\n{tag}")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.boxplot(data, widths=0.5, flierprops=dict(markersize=2))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Patient-level mean Dice")
    ax.set_title("Held-out test cohort (98 patients)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig09_test_boxplot.png"), dpi=300)
    plt.close(fig)

    # Worked examples from the held-out cohort, percentile + morphology
    test_main = [r for r in main if r["split"] == "test"]
    pos = [r for r in test_main if r["positive"] == "1"]
    dices = np.array([float(r["dice"]) for r in pos])
    med = float(np.median(dices))
    examples = {
        "fig10_example_high": max(pos, key=lambda r: float(r["dice"])),
        "fig11_example_typical": min(pos, key=lambda r: abs(float(r["dice"]) - med)),
        "fig12_example_failure": max([r for r in pos if float(r["dice"]) == 0.0], key=lambda r: int(r["gt_px"])),
    }
    neg = [r for r in test_main if r["positive"] == "0" and int(r["pred_px"]) > 0]
    neg_med = float(np.median([int(r["pred_px"]) for r in neg]))
    examples["fig13_example_negative"] = min(neg, key=lambda r: abs(int(r["pred_px"]) - neg_med))

    for name, r in examples.items():
        ipath, mpath = paths(args.data, r["patient"], int(r["slice_number"]))
        im, g = load_image(ipath, "flair"), load_mask(mpath)
        pred = P.run_pipeline(im, "percentile")["final"] > 0
        m = slice_metrics(pred, g)
        if r["positive"] == "1":
            assert abs(m["dice"] - float(r["dice"])) < 1e-9, (name, m["dice"], r["dice"])
            metric = f"Dice {m['dice']:.2f}"
        else:
            assert m["pred_px"] == int(r["pred_px"])
            metric = f"empty reference; {m['pred_px']} predicted px"
        header = f"{r['patient']}, slice {r['slice_number']} | Percentile P75 + morphology | v{ver} | {metric}"
        panel_grid([im, (pred * 255).astype(np.uint8), (g * 255).astype(np.uint8), overlay(im, pred, g)],
                   ["(a) FLAIR input", "(b) Prediction", "(c) Reference", "(d) Overlay"],
                   os.path.join(figdir, name + ".png"), cols=2, size=1.75, header=header)
        captions[name] = {"patient": r["patient"], "slice": int(r["slice_number"]), "metric": metric,
                          "gt_px": m["gt_px"], "pred_px": m["pred_px"]}
    captions["test_median_positive_dice"] = med
    captions["test_negative_median_pred_px"] = neg_med
    json.dump(captions, open(os.path.join(figdir, "figure_sources.json"), "w"), indent=2)
    print(json.dumps(captions, indent=1))


if __name__ == "__main__":
    main()
