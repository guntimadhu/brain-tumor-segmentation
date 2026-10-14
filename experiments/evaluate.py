"""Development-set parameter selection followed by held-out evaluation.

Usage:
    python experiments/evaluate.py --data <path to kaggle_3m> --out results

Stage 1 (development cohort only, experiments/dev_patients.txt):
    - percentile sweep with the FLAIR channel; the value with the highest
      patient-level mean Dice is selected and frozen.
    - channel ablation at the selected percentile (FLAIR vs other channels and
      luminance grayscale), all other settings fixed.
Stage 2 (development and held-out test cohorts, reported separately):
    - percentile (frozen), Otsu and adaptive Gaussian, each with and without
      opening-closing.
Stage 3: timing on the development cohort.
"""
import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import PIL
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import pipeline as P  # noqa: E402
from src.io_utils import pair_images_and_masks  # noqa: E402
from src.evaluation import slice_metrics  # noqa: E402

PERCENTILE_SWEEP = [70, 75, 80, 85, 90, 95]
CHANNELS = ["flair", "luminance", "pre_contrast", "post_contrast"]
CHANNEL_INDEX = {"pre_contrast": 0, "flair": 1, "post_contrast": 2}
TIMING_REPS = 5
FIELDS = ["experiment", "split", "patient", "slice_number", "image_file", "mask_file",
          "method", "param", "channel", "morphology", "positive", "gt_px", "pred_px",
          "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall", "specificity"]


def load_image(path, channel):
    img = Image.open(path)
    if channel == "luminance":
        arr = np.array(img.convert("L"), dtype=np.float64)
    else:
        arr = np.array(img)[:, :, CHANNEL_INDEX[channel]].astype(np.float64)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def load_patient(folder, channel):
    names, masks = pair_images_and_masks(os.listdir(folder))
    slices = []
    for img_name, mask_name in zip(names, masks):
        if mask_name is None:
            continue  # unannotated slices are excluded from reference-based evaluation
        slices.append({
            "slice_number": int(os.path.splitext(img_name)[0].rsplit("_", 1)[-1]),
            "image_file": img_name,
            "mask_file": mask_name,
            "image": load_image(os.path.join(folder, img_name), channel),
            "gt": load_mask(os.path.join(folder, mask_name)),
        })
    return slices


def run_config(slices, method, param, morphology):
    params = dict(P.METHOD_PARAMS[method])
    if method == "percentile":
        params["percentile"] = param
    preds = []
    for s in slices:
        preds.append(P.run_pipeline(s["image"], method=method, params=params,
                                    morphology=morphology)["final"] > 0)
    return preds


def rows_for(experiment, split, patient, slices, preds, method, param, channel, morphology):
    out = []
    for s, pred in zip(slices, preds):
        r = {"experiment": experiment, "split": split, "patient": patient,
             "slice_number": s["slice_number"], "image_file": s["image_file"],
             "mask_file": s["mask_file"], "method": method,
             "param": param if param is not None else "", "channel": channel,
             "morphology": int(morphology)}
        r.update(slice_metrics(pred, s["gt"]))
        r["positive"] = int(r["positive"])
        out.append(r)
    return out


def summarize(rows):
    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if not r["positive"]]
    patients = sorted({r["patient"] for r in rows})
    s = {"n_patients": len(patients), "n_slices": len(rows),
         "n_positive": len(pos), "n_negative": len(neg)}

    for k in ["dice", "iou", "precision", "recall", "specificity"]:
        v = np.array([r[k] for r in pos], dtype=float)
        s[f"pos_slice_mean_{k}"] = float(v.mean()) if len(v) else None
        s[f"pos_slice_sd_{k}"] = float(v.std(ddof=1)) if len(v) > 1 else None
    s["pos_slice_median_dice"] = float(np.median([r["dice"] for r in pos])) if pos else None
    s["pos_empty_predictions"] = sum(1 for r in pos if r["pred_px"] == 0)

    per_patient = {}
    for p in patients:
        pr = [r for r in rows if r["patient"] == p]
        pp = [r for r in pr if r["positive"]]
        pn = [r for r in pr if not r["positive"]]
        g_tot = sum(r["gt_px"] for r in pr)
        p_tot = sum(r["pred_px"] for r in pr)
        d = {"n_slices": len(pr), "n_positive": len(pp), "n_negative": len(pn),
             "gt_total_px": g_tot, "pred_total_px": p_tot,
             "pred_px_on_positive": sum(r["pred_px"] for r in pp),
             "pred_px_on_negative": sum(r["pred_px"] for r in pn),
             "signed_count_error_pct": 100 * (p_tot - g_tot) / g_tot if g_tot else None,
             "neg_nonempty": sum(1 for r in pn if r["pred_px"] > 0)}
        for k in ["dice", "iou", "precision", "recall"]:
            d[f"mean_{k}"] = float(np.mean([r[k] for r in pp])) if pp else None
        per_patient[p] = d

    with_pos = [d for d in per_patient.values() if d["n_positive"]]
    s["n_patients_with_positive"] = len(with_pos)
    for k in ["dice", "iou", "precision", "recall"]:
        v = np.array([d[f"mean_{k}"] for d in with_pos], dtype=float)
        s[f"patient_mean_{k}"] = float(v.mean()) if len(v) else None
        s[f"patient_sd_{k}"] = float(v.std(ddof=1)) if len(v) > 1 else None

    s["neg_nonempty_fraction"] = (sum(1 for r in neg if r["pred_px"] > 0) / len(neg)) if neg else None
    s["neg_mean_pred_px_all"] = float(np.mean([r["pred_px"] for r in neg])) if neg else None
    nz = [r["pred_px"] for r in neg if r["pred_px"] > 0]
    s["neg_mean_pred_px_nonempty"] = float(np.mean(nz)) if nz else None
    s["neg_median_pred_px_nonempty"] = float(np.median(nz)) if nz else None
    s["neg_mean_specificity"] = float(np.mean([r["specificity"] for r in neg])) if neg else None

    errs = [d["signed_count_error_pct"] for d in per_patient.values() if d["signed_count_error_pct"] is not None]
    s["count_error_n_over"] = sum(1 for e in errs if e > 0)
    s["count_error_n_under"] = sum(1 for e in errs if e < 0)
    s["count_error_median_signed_pct"] = float(np.median(errs)) if errs else None
    s["count_error_median_abs_pct"] = float(np.median(np.abs(errs))) if errs else None

    if pos:
        best = max(pos, key=lambda r: r["dice"])
        s["best_slice"] = {"patient": best["patient"], "slice_number": best["slice_number"],
                           "dice": best["dice"]}
    return s, per_patient


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def environment():
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", ROOT, "status", "--porcelain", "--", ".", ":!results"], text=True).strip())
    except Exception:
        commit, dirty = None, None
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            cpu = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True).strip()
        except Exception:
            pass
    import scipy
    return {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
            "scipy": scipy.__version__, "git_commit": commit, "git_dirty": dirty,
            "pipeline_version": P.PIPELINE_VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)

    with open(os.path.join(ROOT, "experiments", "dev_patients.txt")) as f:
        dev = [l.strip() for l in f if l.strip()]
    all_patients = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    test = [p for p in all_patients if p not in dev]
    missing = [p for p in dev if p not in all_patients]
    if missing:
        raise SystemExit(f"Development patients missing from data: {missing}")
    print(f"dev={len(dev)} test={len(test)}")

    # ---- Stage 1a: percentile sweep on development cohort ----
    dev_cache = {p: load_patient(os.path.join(args.data, p), "flair") for p in dev}
    sweep_rows = []
    for pct in PERCENTILE_SWEEP:
        for p in dev:
            preds = run_config(dev_cache[p], "percentile", pct, True)
            sweep_rows += rows_for("dev_percentile_sweep", "dev", p, dev_cache[p], preds,
                                   "percentile", pct, "flair", True)
    sweep_summary = {}
    for pct in PERCENTILE_SWEEP:
        sweep_summary[pct], _ = summarize([r for r in sweep_rows if r["param"] == pct])
    selected = max(PERCENTILE_SWEEP, key=lambda q: sweep_summary[q]["patient_mean_dice"])
    print("sweep:", {q: round(sweep_summary[q]["patient_mean_dice"], 4) for q in PERCENTILE_SWEEP},
          "selected", selected)

    # ---- Stage 1b: channel ablation on development cohort ----
    abl_rows = []
    for ch in CHANNELS:
        for p in dev:
            sl = dev_cache[p] if ch == "flair" else load_patient(os.path.join(args.data, p), ch)
            preds = run_config(sl, "percentile", selected, True)
            abl_rows += rows_for("dev_channel_ablation", "dev", p, sl, preds,
                                 "percentile", selected, ch, True)
    abl_summary = {ch: summarize([r for r in abl_rows if r["channel"] == ch])[0] for ch in CHANNELS}

    # ---- Stage 2: frozen configurations on dev and held-out test ----
    configs = [(m, selected if m == "percentile" else None, morph)
               for m in ["percentile", "otsu", "adaptive"] for morph in (True, False)]
    main_rows, file_rows = [], []
    packed = {f"{m}_{'morph' if mo else 'nomorph'}": {} for m, _, mo in configs}
    for split, plist in [("dev", dev), ("test", test)]:
        for i, p in enumerate(plist):
            sl = dev_cache[p] if split == "dev" else load_patient(os.path.join(args.data, p), "flair")
            for s in sl:
                file_rows.append({"split": split, "patient": p, "slice_number": s["slice_number"],
                                  "image_file": s["image_file"], "mask_file": s["mask_file"],
                                  "positive": int(s["gt"].any()), "gt_px": int(s["gt"].sum())})
            for m, prm, mo in configs:
                preds = run_config(sl, m, prm, mo)
                main_rows += rows_for("main", split, p, sl, preds, m, prm, "flair", mo)
                packed[f"{m}_{'morph' if mo else 'nomorph'}"][p] = np.packbits(np.stack(preds), axis=-1)
            if split == "test" and (i + 1) % 10 == 0:
                print(f"  test patients done: {i + 1}/{len(plist)}")

    main_summary, per_patient = {}, {}
    for split in ["dev", "test"]:
        for m, prm, mo in configs:
            key = f"{split}|{m}|{'morph' if mo else 'nomorph'}"
            sub = [r for r in main_rows if r["split"] == split and r["method"] == m
                   and r["morphology"] == int(mo)]
            main_summary[key], per_patient[key] = summarize(sub)

    # paired Wilcoxon signed-rank tests on held-out patient-level mean Dice
    from scipy.stats import wilcoxon
    def pat_dice(key):
        return {p: d["mean_dice"] for p, d in per_patient[key].items() if d["mean_dice"] is not None}
    tests = {}
    for a, b in [("test|percentile|morph", "test|otsu|morph"),
                 ("test|percentile|morph", "test|adaptive|morph"),
                 ("test|otsu|morph", "test|adaptive|morph"),
                 ("test|percentile|morph", "test|percentile|nomorph"),
                 ("test|otsu|morph", "test|otsu|nomorph"),
                 ("test|adaptive|morph", "test|adaptive|nomorph")]:
        da, db = pat_dice(a), pat_dice(b)
        common = sorted(set(da) & set(db))
        x, y = np.array([da[p] for p in common]), np.array([db[p] for p in common])
        res = wilcoxon(x, y)
        tests[f"{a} vs {b}"] = {"n_patients": len(common), "mean_difference": float((x - y).mean()),
                                "median_difference": float(np.median(x - y)),
                                "statistic": float(res.statistic), "p_value": float(res.pvalue)}

    # ---- Stage 3: timing (development cohort, frozen default configuration) ----
    dev_paths = [os.path.join(args.data, p, s["image_file"]) for p in dev for s in dev_cache[p]]
    t_load = []
    for path in dev_paths:
        t0 = time.perf_counter()
        load_image(path, "flair")
        t_load.append(time.perf_counter() - t0)
    imgs = [s["image"] for p in dev for s in dev_cache[p]]
    params = {"percentile": selected}
    for im in imgs[:20]:
        P.run_pipeline(im, "percentile", params, True)  # warm-up, not timed
    t_proc = []
    for _ in range(TIMING_REPS):
        for im in imgs:
            t0 = time.perf_counter()
            P.run_pipeline(im, "percentile", params, True)
            t_proc.append(time.perf_counter() - t0)
    t_proc, t_load = np.array(t_proc) * 1000, np.array(t_load) * 1000
    timing = {"slices": len(imgs), "repetitions": TIMING_REPS, "measurements": int(t_proc.size),
              "boundary": "run_pipeline(): normalisation, Gaussian filter, CLAHE, brain mask, "
                          "threshold, morphology, component filtering and selection on an in-memory "
                          "uint8 FLAIR slice; excludes disk reading, metric computation and display",
              "processing_ms_mean": float(t_proc.mean()), "processing_ms_sd": float(t_proc.std(ddof=1)),
              "processing_ms_median": float(np.median(t_proc)),
              "loading_ms_mean": float(t_load.mean()), "loading_ms_sd": float(t_load.std(ddof=1)),
              "loading_boundary": "TIFF read, FLAIR channel extraction and min-max scaling, measured once per slice"}

    # ---- write outputs ----
    write_csv(os.path.join(args.out, "per_slice_dev_sweep.csv"), sweep_rows, FIELDS)
    write_csv(os.path.join(args.out, "per_slice_dev_channel_ablation.csv"), abl_rows, FIELDS)
    write_csv(os.path.join(args.out, "per_slice_main.csv"), main_rows, FIELDS)
    write_csv(os.path.join(args.out, "file_list.csv"), file_rows,
              ["split", "patient", "slice_number", "image_file", "mask_file", "positive", "gt_px"])
    pp_rows = []
    for key, d in per_patient.items():
        split, m, mo = key.split("|")
        for p, v in d.items():
            pp_rows.append({"split": split, "method": m, "morphology": mo, "patient": p, **v})
    write_csv(os.path.join(args.out, "per_patient_main.csv"), pp_rows, list(pp_rows[0].keys()))
    for name, d in packed.items():
        np.savez_compressed(os.path.join(args.out, "predictions", f"{name}.npz"), **d)

    config = P.config_dict()
    config["methods"]["percentile"] = {"percentile": selected}
    summary = {"environment": environment(), "config": config,
               "definitions": {
                   "positive_slice": "supplied reference mask contains at least one foreground pixel",
                   "negative_slice": "supplied reference mask is empty",
                   "unannotated_slice": "no mask file; excluded from reference-based metrics (none occurred)",
                   "dice": "2TP/(2TP+FP+FN), positive slices only",
                   "iou": "TP/(TP+FP+FN), positive slices only",
                   "recall": "TP/(TP+FN), positive slices only",
                   "precision": "TP/(TP+FP); set to 0 when the prediction is empty on a positive slice",
                   "specificity": "TN/(TN+FP), pixel-wise per slice",
                   "pos_slice_mean": "mean over all positive slices (weights patients by positive-slice count)",
                   "patient_mean": "mean over patients of each patient's mean over its positive slices",
                   "signed_count_error_pct": "100*(P_total-G_total)/G_total summed over all slices of a patient"},
               "split": {"dev": dev, "n_test": len(test)},
               "percentile_selection": {"criterion": "highest development patient-level mean Dice",
                                        "candidates": PERCENTILE_SWEEP, "selected": selected,
                                        "results": sweep_summary},
               "channel_ablation": abl_summary,
               "main": main_summary,
               "wilcoxon_test_patients": tests,
               "timing": timing}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done; outputs in", args.out)


if __name__ == "__main__":
    main()
