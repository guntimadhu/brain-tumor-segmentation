"""Single source of truth for the segmentation pipeline used by the app and the evaluation."""
import numpy as np

from src.preprocessing import preprocess_image
from src.segmentation import segment_tumor_candidate, apply_skull_strip_approximation
from src.morphology import apply_morphological_pipeline
from src.postprocessing import filter_components_by_size, select_tumor_region

PIPELINE_VERSION = "2.0.0"

PREPROCESS = {"normalize": True, "filter_type": "gaussian", "sigma": 1.0,
              "clahe": True, "clip_limit": 2.0}
MORPHOLOGY = {"operation": "opening_then_closing", "kernel_size": 5,
              "kernel_shape": "ellipse", "iterations": 1}
MIN_COMPONENT_AREA = 100
MAX_COMPONENT_FRACTION = 0.25

METHOD_PARAMS = {
    "percentile": {"percentile": 75},  # selected on the development cohort (experiments/evaluate.py)
    "otsu": {},
    "adaptive": {"block_size": 11, "C": 2},
}
METHOD_LABELS = {
    "percentile": "Percentile threshold (brain-masked)",
    "otsu": "Otsu threshold (brain-masked)",
    "adaptive": "Adaptive Gaussian threshold",
}
DEFAULT_METHOD = "percentile"


def run_pipeline(image, method=DEFAULT_METHOD, params=None, morphology=True):
    """image: uint8 FLAIR slice. Returns the final mask and all intermediate steps."""
    if params is None:
        params = METHOD_PARAMS[method]
    enhanced, pp_steps = preprocess_image(image, PREPROCESS)
    brain_mask = apply_skull_strip_approximation(pp_steps["normalized"])
    seg = segment_tumor_candidate(enhanced, method=method, params=params, brain_mask=brain_mask)
    morph = apply_morphological_pipeline(seg["binary_mask"], MORPHOLOGY) if morphology else None
    refined = morph["output_mask"] if morph else seg["binary_mask"]
    filtered = filter_components_by_size(refined, min_area=MIN_COMPONENT_AREA)
    h, w = filtered.shape
    final = select_tumor_region(filtered, strategy="largest", min_area=MIN_COMPONENT_AREA,
                                max_area=int(MAX_COMPONENT_FRACTION * h * w))
    return {
        "enhanced": enhanced,
        "pp_steps": pp_steps,
        "brain_mask": brain_mask,
        "seg": seg,
        "morph": morph,
        "filtered": filtered,
        "final": final,
    }


def config_dict():
    return {
        "pipeline_version": PIPELINE_VERSION,
        "preprocess": PREPROCESS,
        "skull_strip": {"source": "normalized image (before CLAHE)", "tissue_threshold": 15,
                        "component": "largest", "closing_kernel": 7, "hole_fill": "external contours",
                        "erosion_kernel": 9},
        "methods": METHOD_PARAMS,
        "default_method": DEFAULT_METHOD,
        "morphology": MORPHOLOGY,
        "min_component_area_px": MIN_COMPONENT_AREA,
        "max_component_fraction": MAX_COMPONENT_FRACTION,
        "selection": "largest remaining component",
    }


def count_foreground(mask):
    return int(np.count_nonzero(mask))
