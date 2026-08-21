import numpy as np
from src.io_utils import load_patient_data, get_patient_list
from src.preprocessing import preprocess_image
from src.segmentation import segment_tumor_candidate
from src.morphology import apply_morphological_pipeline
from src.postprocessing import filter_components_by_size, select_tumor_region, create_tumor_overlay
from src.measurements import calculate_tumor_area, estimate_tumor_volume
from src.evaluation import evaluate_segmentation

DATA_DIR = "data/raw"

print("=" * 60)
print("BRAIN TUMOR SEGMENTATION - FULL PIPELINE TEST")
print("=" * 60)

# Step 1: Find patients
patients = get_patient_list(DATA_DIR)
print(f"\n[1] Found {len(patients)} patients in {DATA_DIR}")
for p in patients:
    print(f"    - {p}")

if not patients:
    print("ERROR: No patients found!")
    exit(1)

# Step 2: Load first patient
patient = load_patient_data(patients[0])
print(f"\n[2] Loaded patient: {patient['patient_id']}")
print(f"    Slices: {patient['slice_count']}")
print(f"    Shape: {patient['image_shape']}")
print(f"    Has masks: {patient['has_masks']}")
print(f"    Tumor slices: {patient['tumor_slices']}")
print(f"    Mask count: {len(patient['masks'])}")

if patient['has_masks']:
    for i, m in enumerate(patient['masks']):
        if m.max() > 10:
            print(f"    Mask[{i}]: min={m.min()}, max={m.max()}, nonzero={np.count_nonzero(m)}")
            break

# Step 3: Pick a slice with tumor
if patient['tumor_slices']:
    test_idx = patient['tumor_slices'][0]
else:
    test_idx = patient['slice_count'] // 2
print(f"\n[3] Testing with slice index: {test_idx}")

img = patient['images'][test_idx]
print(f"    Image: dtype={img.dtype}, min={img.min()}, max={img.max()}, shape={img.shape}")

# Step 4: Preprocessing
config = {'normalize': True, 'filter_type': 'gaussian', 'sigma': 1.0, 'clahe': True, 'clip_limit': 2.0}
pp_img, pp_steps = preprocess_image(img, config)
print(f"\n[4] Preprocessing:")
for name, step_img in pp_steps.items():
    print(f"    {name}: dtype={step_img.dtype}, min={step_img.min()}, max={step_img.max()}")

# Step 5: Segmentation
seg = segment_tumor_candidate(pp_img, method='otsu')
print(f"\n[5] Segmentation (Otsu):")
print(f"    Candidate pixels: {seg['candidate_region_count']}")
print(f"    Brain mask nonzero: {np.count_nonzero(seg['brain_mask'])}")
print(f"    Binary mask nonzero: {np.count_nonzero(seg['binary_mask'])}")
print(f"    Threshold: {seg['threshold_value']}")

# Step 6: Morphology
morph = apply_morphological_pipeline(seg['binary_mask'], {'operation': 'opening_then_closing', 'kernel_size': 5, 'iterations': 1})
print(f"\n[6] Morphology:")
print(f"    Noise removed: {morph['noise_removed_pixels']} pixels")
print(f"    Output nonzero: {np.count_nonzero(morph['output_mask'])}")

# Step 7: Post-processing
filtered = filter_components_by_size(morph['output_mask'], min_area=100)
tumor = select_tumor_region(filtered, strategy='largest', min_area=50)
overlay = create_tumor_overlay(img, tumor)
print(f"\n[7] Post-processing:")
print(f"    Filtered nonzero: {np.count_nonzero(filtered)}")
print(f"    Tumor mask nonzero: {np.count_nonzero(tumor)}")
print(f"    Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")

# Step 8: Measurements
area = calculate_tumor_area(tumor, pixel_spacing_mm=1.0)
print(f"\n[8] Measurements:")
print(f"    Tumor pixels: {area['tumor_pixels']}")
print(f"    Area: {area['tumor_area_mm2']:.1f} mm2")
print(f"    Percentage: {area['tumor_percentage']:.4f}%")

# Step 9: Volume (all slices)
pred_masks = []
for s in patient['images']:
    pp, _ = preprocess_image(s, config)
    sg = segment_tumor_candidate(pp, method='otsu')
    mp = apply_morphological_pipeline(sg['binary_mask'], {'operation': 'opening_then_closing', 'kernel_size': 5, 'iterations': 1})
    ft = filter_components_by_size(mp['output_mask'], min_area=100)
    tm = select_tumor_region(ft, strategy='largest', min_area=50)
    pred_masks.append(tm)

vol = estimate_tumor_volume(pred_masks, 1.0, 5.0)
print(f"\n[9] Volume (all {len(pred_masks)} slices):")
print(f"    Volume: {vol['tumor_volume_cm3']:.2f} cm3")
print(f"    Affected slices: {vol['affected_slices']}/{vol['total_slices']}")

# Step 10: Evaluation
if patient['has_masks'] and test_idx < len(patient['masks']):
    gt_mask = patient['masks'][test_idx]
    ev = evaluate_segmentation(tumor, gt_mask)
    print(f"\n[10] Evaluation (slice {test_idx}):")
    print(f"    GT mask: min={gt_mask.min()}, max={gt_mask.max()}, nonzero={np.count_nonzero(gt_mask)}")
    print(f"    Pred mask: nonzero={np.count_nonzero(tumor)}")
    print(f"    Dice:        {ev['dice']}")
    print(f"    IoU:         {ev['iou']}")
    print(f"    Precision:   {ev['precision']}")
    print(f"    Recall:      {ev['recall']}")
    print(f"    Specificity: {ev['specificity']}")
    print(f"    TP={ev['TP']}, FP={ev['FP']}, TN={ev['TN']}, FN={ev['FN']}")
    print(f"    {ev['interpretation']}")
else:
    print("\n[10] Evaluation: skipped (no ground truth for this slice)")

print("\n" + "=" * 60)
print("ALL PIPELINE STEPS PASSED")
print("=" * 60)
