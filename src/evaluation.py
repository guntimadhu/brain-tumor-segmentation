import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_binary(mask):
    """Convert any mask to binary 0/1 using threshold 127."""
    return (mask > 127).astype(np.uint8)


def compute_confusion_matrix_values(pred_mask, true_mask):
    """Compute TP, FP, TN, FN from masks."""
    pred = _to_binary(pred_mask).ravel()
    true = _to_binary(true_mask).ravel()
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    tn = int(np.sum((pred == 0) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def dice_coefficient(pred_mask, true_mask):
    """Dice = 2|P∩T| / (|P|+|T|). Returns 1.0 when both masks are empty."""
    pred = _to_binary(pred_mask)
    true = _to_binary(true_mask)
    if pred.sum() == 0 and true.sum() == 0:
        return 1.0
    tp = np.sum((pred == 1) & (true == 1))
    return float(2 * tp) / (pred.sum() + true.sum() + 1e-8)


def iou_score(pred_mask, true_mask):
    """Intersection over Union. Returns 1.0 when both masks are empty."""
    pred = _to_binary(pred_mask)
    true = _to_binary(true_mask)
    if pred.sum() == 0 and true.sum() == 0:
        return 1.0
    intersection = np.sum((pred == 1) & (true == 1))
    union = np.sum((pred == 1) | (true == 1))
    return float(intersection) / (union + 1e-8)


def precision_score(pred_mask, true_mask):
    """Precision = TP / (TP + FP)."""
    cm = compute_confusion_matrix_values(pred_mask, true_mask)
    return cm["TP"] / (cm["TP"] + cm["FP"] + 1e-8)


def recall_score(pred_mask, true_mask):
    """Recall (sensitivity) = TP / (TP + FN)."""
    cm = compute_confusion_matrix_values(pred_mask, true_mask)
    return cm["TP"] / (cm["TP"] + cm["FN"] + 1e-8)


def specificity_score(pred_mask, true_mask):
    """Specificity = TN / (TN + FP)."""
    cm = compute_confusion_matrix_values(pred_mask, true_mask)
    return cm["TN"] / (cm["TN"] + cm["FP"] + 1e-8)


def evaluate_segmentation(pred_mask, true_mask):
    """Complete single-slice evaluation with all metrics and interpretation."""
    pred = _to_binary(pred_mask)
    true = _to_binary(true_mask)

    if pred.sum() == 0 and true.sum() == 0:
        return {
            "dice": 1.0, "iou": 1.0,
            "precision": 1.0, "recall": 1.0, "specificity": 1.0,
            "TP": 0, "FP": 0, "TN": int((pred == 0).sum()), "FN": 0,
            "no_tumor_slice": True,
            "interpretation": "No tumor in either mask",
        }

    d = dice_coefficient(pred_mask, true_mask)
    iou = iou_score(pred_mask, true_mask)
    prec = precision_score(pred_mask, true_mask)
    rec = recall_score(pred_mask, true_mask)
    spec = specificity_score(pred_mask, true_mask)
    cm = compute_confusion_matrix_values(pred_mask, true_mask)

    if d > 0.7:
        interp = "Good overlap — segmentation closely matches ground truth."
    elif d > 0.5:
        interp = "Moderate overlap — some regions missed or over-segmented."
    else:
        interp = "Poor overlap — classical methods have limitations on this slice."

    return {
        "dice": round(d, 4),
        "iou": round(iou, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        **cm,
        "no_tumor_slice": False,
        "interpretation": interp,
    }


def evaluate_volume_set(pred_masks_list, true_masks_list):
    """Evaluate segmentation across all slices of a patient volume."""
    per_slice = [dice_coefficient(p, t) for p, t in zip(pred_masks_list, true_masks_list)]
    per_slice_iou = [iou_score(p, t) for p, t in zip(pred_masks_list, true_masks_list)]

    pred_flat = np.concatenate([_to_binary(m).ravel() for m in pred_masks_list])
    true_flat = np.concatenate([_to_binary(m).ravel() for m in true_masks_list])

    overall = evaluate_segmentation(
        (pred_flat * 255).astype(np.uint8),
        (true_flat * 255).astype(np.uint8),
    )

    return {
        "per_slice_dice": per_slice,
        "mean_dice": float(np.mean(per_slice)),
        "std_dice": float(np.std(per_slice)),
        "mean_iou": float(np.mean(per_slice_iou)),
        "overall_dice": overall["dice"],
        "overall_iou": overall["iou"],
        "summary": overall,
    }


def plot_evaluation_charts(eval_results):
    """Create a figure with a metrics bar chart and confusion matrix heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["dice", "iou", "precision", "recall", "specificity"]
    values = [eval_results.get(m, 0) for m in metrics]
    colors = ["#2ecc71" if v > 0.7 else "#f39c12" if v > 0.5 else "#e74c3c" for v in values]
    ax1.bar(metrics, values, color=colors)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Segmentation Metrics")
    for i, v in enumerate(values):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    cm = np.array([[eval_results.get("TP", 0), eval_results.get("FN", 0)],
                    [eval_results.get("FP", 0), eval_results.get("TN", 0)]])
    ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Positive", "Negative"])
    ax2.set_yticklabels(["Positive", "Negative"])
    ax2.set_xlabel("Ground Truth")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    return fig
