"""Mask measurement in pixels. The LGG TIFF data carry no verified spatial
calibration, so no physical areas or volumes are reported."""
import numpy as np
import pandas as pd


def foreground_pixels(mask):
    n = int(np.count_nonzero(mask))
    return {"foreground_pixels": n, "total_pixels": int(mask.size),
            "foreground_percentage": 100.0 * n / mask.size if mask.size else 0.0}


def summarize_foreground(pred_masks, ref_masks=None):
    pred_total = sum(int(np.count_nonzero(m)) for m in pred_masks)
    out = {"pred_total_px": pred_total,
           "pred_affected_slices": sum(1 for m in pred_masks if np.count_nonzero(m)),
           "total_slices": len(pred_masks)}
    refs = [r for r in (ref_masks or []) if r is not None]
    if refs and len(refs) == len(pred_masks):
        ref_total = sum(int(np.count_nonzero(r > 127)) for r in refs)
        out["ref_total_px"] = ref_total
        out["ref_affected_slices"] = sum(1 for r in refs if np.count_nonzero(r > 127))
        out["signed_count_error_pct"] = (100.0 * (pred_total - ref_total) / ref_total) if ref_total else None
    return out


def slice_dataframe(pred_masks, slice_labels, ref_masks=None):
    rows = []
    for i, m in enumerate(pred_masks):
        row = {"Slice": slice_labels[i], "Predicted px": int(np.count_nonzero(m))}
        if ref_masks and i < len(ref_masks) and ref_masks[i] is not None:
            row["Reference px"] = int(np.count_nonzero(ref_masks[i] > 127))
        rows.append(row)
    return pd.DataFrame(rows)
