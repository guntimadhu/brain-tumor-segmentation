"""Compare classical and deep-learning segmentation on the 28 deep-learning test patients.

Classical results are taken from results/per_slice_main.csv (frozen pipeline v2.0.0; these
patients were never used for its tuning). Deep-learning models are evaluated for every
trained seed; the Attention U-Net is additionally evaluated with Monte-Carlo dropout
(20 stochastic passes averaged), which also yields an uncertainty map.

Usage: python experiments/evaluate_dl.py
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import summarize  # noqa: E402
from src.evaluation import slice_metrics  # noqa: E402
from src.dl.infer import load, predict_probs, postprocess  # noqa: E402

torch.set_num_threads(10)
OUT = os.path.join(ROOT, "results", "dl")
MC_PASSES = 20
SEEDS = [42]

cache = np.load(os.path.join(ROOT, "data", "dl_cache.npz"))
images, masks, patient, slice_no = cache["images"], cache["masks"], cache["patient"], cache["slice_number"]
split = json.load(open(os.path.join(ROOT, "experiments", "dl_split.json")))
test_idx = np.where(np.isin(patient, split["test"]))[0]


def hd95(pred, gt):
    """95th-percentile symmetric surface distance in pixels; None if either mask is empty."""
    if not pred.any() or not gt.any():
        return None
    bp = pred ^ ndimage.binary_erosion(pred)
    bg = gt ^ ndimage.binary_erosion(gt)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    return float(np.percentile(np.concatenate([dt_g[bp], dt_p[bg]]), 95))


def rows_from_preds(preds, method):
    rows = []
    for k, i in enumerate(test_idx):
        m = slice_metrics(preds[k], masks[i])
        m.update({"patient": patient[i], "slice_number": int(slice_no[i]), "method": method,
                  "positive": int(m["positive"]), "hd95": hd95(preds[k], masks[i]) if m["positive"] else None})
        rows.append(m)
    return rows


def classical_rows(method):
    rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, "results", "per_slice_main.csv"))):
        if r["method"] == method and r["morphology"] == "1" and r["patient"] in split["test"]:
            row = {k: (float(r[k]) if r[k] not in ("", None) else None) for k in
                   ["dice", "iou", "precision", "recall", "specificity"]}
            row.update({"patient": r["patient"], "slice_number": int(r["slice_number"]), "method": f"classical_{method}",
                        "positive": int(r["positive"]), "gt_px": int(r["gt_px"]), "pred_px": int(r["pred_px"])})
            rows.append(row)
    return rows


def add_hd95_classical(rows, method):
    """Recompute classical predictions for HD95 (needs masks, which the CSV does not store)."""
    from src import pipeline as P
    lookup = {(patient[i], int(slice_no[i])): i for i in test_idx}
    for r in rows:
        if r["positive"]:
            i = lookup[(r["patient"], r["slice_number"])]
            pred = P.run_pipeline(images[i], method=method)["final"] > 0
            r["hd95"] = hd95(pred, masks[i])
    return rows


def hd_summary(rows):
    v = [r["hd95"] for r in rows if r["positive"] and r.get("hd95") is not None]
    per = {}
    for r in rows:
        if r["positive"] and r.get("hd95") is not None:
            per.setdefault(r["patient"], []).append(r["hd95"])
    pm = [np.mean(x) for x in per.values()]
    return {"hd95_slice_median": float(np.median(v)) if v else None,
            "hd95_patient_mean": float(np.mean(pm)) if pm else None,
            "hd95_patient_sd": float(np.std(pm, ddof=1)) if len(pm) > 1 else None}


results, all_rows, per_patient_dice = {}, [], {}

for method in ["percentile", "otsu", "adaptive"]:
    rows = add_hd95_classical(classical_rows(method), method)
    s, pp = summarize(rows)
    s.update(hd_summary(rows))
    results[f"classical_{method}"] = s
    per_patient_dice[f"classical_{method}"] = {p: d["mean_dice"] for p, d in pp.items() if d["mean_dice"] is not None}
    all_rows += rows
    print(f"classical_{method}: patient Dice {s['patient_mean_dice']:.3f}", flush=True)

timing = {}
for name in ["unet", "attention_unet"]:
    seed_scores = []
    for seed in SEEDS:
        path = os.path.join(ROOT, "models", f"{name}_seed{seed}.pt")
        if not os.path.exists(path):
            continue
        model = load(name, seed)
        t0 = time.perf_counter()
        probs, _ = predict_probs(model, images[test_idx])
        ms = (time.perf_counter() - t0) / len(test_idx) * 1000
        preds = np.stack([postprocess(p) for p in probs])
        key = f"{name}_seed{seed}"
        rows = rows_from_preds(preds, key)
        s, pp = summarize(rows)
        s.update(hd_summary(rows))
        s["ms_per_slice_batched"] = ms
        results[key] = s
        per_patient_dice[key] = {p: d["mean_dice"] for p, d in pp.items() if d["mean_dice"] is not None}
        seed_scores.append(s["patient_mean_dice"])
        if seed == 42:
            all_rows += rows
            np.savez_compressed(os.path.join(OUT, f"pred_{key}.npz"), probs=probs.astype(np.float16))
        print(f"{key}: patient Dice {s['patient_mean_dice']:.3f}  ({ms:.1f} ms/slice)", flush=True)

        if name == "attention_unet":
            t0 = time.perf_counter()
            mprobs, stds = predict_probs(model, images[test_idx], mc_passes=MC_PASSES)
            ms_mc = (time.perf_counter() - t0) / len(test_idx) * 1000
            mpreds = np.stack([postprocess(p) for p in mprobs])
            mkey = f"{name}_mc_seed{seed}"
            mrows = rows_from_preds(mpreds, mkey)
            ms_, mpp = summarize(mrows)
            ms_.update(hd_summary(mrows))
            ms_["ms_per_slice_batched"] = ms_mc
            # uncertainty inside the predicted region: higher on wrong predictions?
            unc_fp, unc_tp = [], []
            for k, i in enumerate(test_idx):
                if mpreds[k].any():
                    u = float(stds[k][mpreds[k]].mean())
                    (unc_tp if masks[i].any() else unc_fp).append(u)
            ms_["uncertainty_mean_on_positive_slices"] = float(np.mean(unc_tp)) if unc_tp else None
            ms_["uncertainty_mean_on_false_alarm_slices"] = float(np.mean(unc_fp)) if unc_fp else None
            results[mkey] = ms_
            per_patient_dice[mkey] = {p: d["mean_dice"] for p, d in mpp.items() if d["mean_dice"] is not None}
            if seed == 42:
                all_rows += mrows
                np.savez_compressed(os.path.join(OUT, f"pred_{mkey}.npz"), probs=mprobs.astype(np.float16),
                                    std=stds.astype(np.float16))
            print(f"{mkey}: patient Dice {ms_['patient_mean_dice']:.3f}  ({ms_mc:.1f} ms/slice)", flush=True)
    if seed_scores:
        results[f"{name}_seed_summary"] = {"seeds": len(seed_scores), "patient_dice_mean": float(np.mean(seed_scores)),
                                           "patient_dice_sd_across_seeds": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else None}

# single-slice CPU latency (batch size 1), as in the base paper
x1 = images[test_idx[:1]]
for name in ["unet", "attention_unet"]:
    if os.path.exists(os.path.join(ROOT, "models", f"{name}_seed42.pt")):
        model = load(name, 42)
        predict_probs(model, x1)
        t = []
        for _ in range(30):
            t0 = time.perf_counter()
            predict_probs(model, x1)
            t.append((time.perf_counter() - t0) * 1000)
        timing[name] = {"ms_mean": float(np.mean(t)), "ms_sd": float(np.std(t, ddof=1))}

tests = {}
for a, b in [("attention_unet_seed42", "classical_percentile"), ("unet_seed42", "classical_percentile"),
             ("attention_unet_seed42", "unet_seed42"), ("attention_unet_mc_seed42", "attention_unet_seed42")]:
    if a in per_patient_dice and b in per_patient_dice:
        common = sorted(set(per_patient_dice[a]) & set(per_patient_dice[b]))
        x = np.array([per_patient_dice[a][p] for p in common])
        y = np.array([per_patient_dice[b][p] for p in common])
        r = wilcoxon(x, y) if np.any(x != y) else None
        tests[f"{a} vs {b}"] = {"n": len(common), "mean_difference": float((x - y).mean()),
                                "p_value": float(r.pvalue) if r else 1.0}

json.dump({"test_patients": split["test"], "n_test_slices": int(len(test_idx)), "results": results,
           "wilcoxon": tests, "latency_batch1_ms": timing, "mc_passes": MC_PASSES},
          open(os.path.join(OUT, "summary_dl.json"), "w"), indent=2)
fields = ["method", "patient", "slice_number", "positive", "gt_px", "pred_px", "dice", "iou", "precision", "recall",
          "specificity", "hd95"]
with open(os.path.join(OUT, "per_slice_dl.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_rows)
print("saved", os.path.join(OUT, "summary_dl.json"))
