"""Smoke test: python test_pipeline.py"""
import os
import numpy as np
from src.io_utils import load_patient_data, get_patient_list, pair_images_and_masks
from src.pipeline import run_pipeline, METHOD_LABELS
from src.evaluation import patient_metrics
from src.measurements import summarize_foreground

names = ["P_1.tif", "P_10.tif", "P_2.tif", "P_1_mask.tif", "P_10_mask.tif", "P_2_mask.tif"]
imgs, masks = pair_images_and_masks(names)
assert imgs == ["P_1.tif", "P_2.tif", "P_10.tif"], imgs
assert masks == ["P_1_mask.tif", "P_2_mask.tif", "P_10_mask.tif"], masks

patients = get_patient_list("data/raw")
if not patients:
    raise SystemExit("No patients in data/raw")
patient = load_patient_data(patients[0])
for img_name, mask_name in zip(patient["image_files"], patient["mask_files"]):
    assert os.path.splitext(img_name)[0] + "_mask" == os.path.splitext(mask_name)[0]
print(f"{patient['patient_id']}: {patient['slice_count']} slices, pairing OK")

for method in METHOD_LABELS:
    preds = [run_pipeline(im, method=method)["final"] for im in patient["images"]]
    pm = patient_metrics(preds, patient["masks"])
    cs = summarize_foreground(preds, patient["masks"])
    dice = f"{pm['mean_dice']:.3f}" if pm["mean_dice"] is not None else "n/a"
    print(f"{method:10s} mean Dice (positive slices) {dice}  "
          f"negative slices with prediction {pm['neg_nonempty']}/{pm['n_negative']}  "
          f"count error {cs['signed_count_error_pct']:+.1f}%")
print("OK")
