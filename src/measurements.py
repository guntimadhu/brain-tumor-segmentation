import numpy as np
import pandas as pd


def calculate_tumor_area(mask, pixel_spacing_mm=1.0):
    """Calculate tumor area in a single 2D slice."""
    tumor_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    area_mm2 = tumor_pixels * (pixel_spacing_mm ** 2)
    return {
        "tumor_pixels": tumor_pixels,
        "tumor_area_mm2": area_mm2,
        "tumor_area_cm2": area_mm2 / 100.0,
        "total_pixels": total_pixels,
        "tumor_percentage": (tumor_pixels / total_pixels * 100) if total_pixels > 0 else 0.0,
    }


def calculate_slice_areas(masks_list, pixel_spacing_mm=1.0):
    """Calculate tumor area for every slice."""
    results = []
    for i, mask in enumerate(masks_list):
        area = calculate_tumor_area(mask, pixel_spacing_mm)
        area["slice_index"] = i
        results.append(area)
    return results


def estimate_tumor_volume(masks_list, pixel_spacing_mm=1.0, slice_thickness_mm=5.0):
    """Estimate tumor volume using voxel-count (primary) and slice-summation (verification).

    Pixel spacing and slice thickness are assumed values — the Kaggle 2D dataset
    does not include spatial metadata. Results are pseudo-3D estimates.
    """
    voxel_volume = pixel_spacing_mm * pixel_spacing_mm * slice_thickness_mm
    total_voxels = sum(int(np.count_nonzero(m)) for m in masks_list)
    volume_mm3 = total_voxels * voxel_volume

    slice_areas = calculate_slice_areas(masks_list, pixel_spacing_mm)
    slice_sum_mm3 = sum(s["tumor_area_mm2"] for s in slice_areas) * slice_thickness_mm

    affected = sum(1 for s in slice_areas if s["tumor_pixels"] > 0)

    return {
        "total_tumor_voxels": total_voxels,
        "voxel_volume_mm3": voxel_volume,
        "tumor_volume_mm3": volume_mm3,
        "tumor_volume_cm3": volume_mm3 / 1000.0,
        "tumor_volume_ml": volume_mm3 / 1000.0,
        "affected_slices": affected,
        "total_slices": len(masks_list),
        "slice_summation_mm3": slice_sum_mm3,
        "pixel_spacing_mm": pixel_spacing_mm,
        "slice_thickness_mm": slice_thickness_mm,
        "note": (
            "Volume estimated from 2D MRI slices using assumed spatial parameters "
            f"(pixel spacing {pixel_spacing_mm} mm, slice thickness {slice_thickness_mm} mm). "
            "These are pseudo-3D estimates, not derived from DICOM/NIfTI metadata."
        ),
    }


def get_tumor_statistics(masks_list, pixel_spacing_mm=1.0, slice_thickness_mm=5.0):
    """Complete measurement summary combining per-slice areas and volume."""
    slice_areas = calculate_slice_areas(masks_list, pixel_spacing_mm)
    volume = estimate_tumor_volume(masks_list, pixel_spacing_mm, slice_thickness_mm)

    area_values = [s["tumor_area_mm2"] for s in slice_areas]
    tumor_areas = [a for a in area_values if a > 0]

    stats = {
        "volume": volume,
        "slice_areas": slice_areas,
        "min_area_mm2": min(tumor_areas) if tumor_areas else 0.0,
        "max_area_mm2": max(tumor_areas) if tumor_areas else 0.0,
        "mean_area_mm2": float(np.mean(tumor_areas)) if tumor_areas else 0.0,
        "std_area_mm2": float(np.std(tumor_areas)) if tumor_areas else 0.0,
    }
    return stats


def generate_slice_area_dataframe(masks_list, pixel_spacing_mm=1.0):
    """Create a pandas DataFrame summarizing per-slice tumor data."""
    rows = []
    for i, mask in enumerate(masks_list):
        area = calculate_tumor_area(mask, pixel_spacing_mm)
        rows.append({
            "Slice": i,
            "Has Tumor": area["tumor_pixels"] > 0,
            "Tumor Pixels": area["tumor_pixels"],
            "Area mm2": round(area["tumor_area_mm2"], 2),
            "Percentage": round(area["tumor_percentage"], 4),
        })
    return pd.DataFrame(rows)
