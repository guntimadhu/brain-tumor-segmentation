import numpy as np
import cv2


_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


def get_kernel(size=5, shape="ellipse"):
    """Create a structuring element for morphological operations."""
    morph_shape = _SHAPES.get(shape, cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(morph_shape, (size, size))


def apply_erosion(mask, kernel_size=5, iterations=1):
    """Erosion shrinks bright regions — removes small bright noise."""
    kernel = get_kernel(kernel_size)
    return cv2.erode(mask, kernel, iterations=iterations)


def apply_dilation(mask, kernel_size=5, iterations=1):
    """Dilation expands bright regions — fills small dark gaps."""
    kernel = get_kernel(kernel_size)
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_opening(mask, kernel_size=5, iterations=1):
    """Opening (erosion then dilation) removes small bright noise while preserving shape."""
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(mask, kernel_size=5, iterations=1):
    """Closing (dilation then erosion) fills small dark holes while preserving shape."""
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_morphological_pipeline(mask, config=None):
    """Master morphological processing pipeline.

    Returns input/output masks, intermediate steps, and pixel-change stats.
    """
    if config is None:
        config = {}

    operation = config.get("operation", "opening_then_closing")
    kernel_size = config.get("kernel_size", 5)
    kernel_shape = config.get("kernel_shape", "ellipse")
    iterations = config.get("iterations", 1)

    kernel = get_kernel(kernel_size, kernel_shape)
    steps = []
    current = mask.copy()

    if operation == "erosion":
        current = cv2.erode(current, kernel, iterations=iterations)
        steps.append(("Erosion", current.copy()))
    elif operation == "dilation":
        current = cv2.dilate(current, kernel, iterations=iterations)
        steps.append(("Dilation", current.copy()))
    elif operation == "opening":
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel, iterations=iterations)
        steps.append(("Opening", current.copy()))
    elif operation == "closing":
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        steps.append(("Closing", current.copy()))
    elif operation == "opening_then_closing":
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel, iterations=iterations)
        steps.append(("Opening", current.copy()))
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        steps.append(("Closing", current.copy()))

    input_pixels = int(np.count_nonzero(mask))
    output_pixels = int(np.count_nonzero(current))

    return {
        "input_mask": mask,
        "output_mask": current,
        "intermediate_steps": steps,
        "noise_removed_pixels": input_pixels - output_pixels,
        "config_used": {
            "operation": operation,
            "kernel_size": kernel_size,
            "kernel_shape": kernel_shape,
            "iterations": iterations,
        },
    }


def get_morphology_comparison(mask):
    """Apply all standard operations for UI comparison."""
    return {
        "original": mask.copy(),
        "erosion": apply_erosion(mask),
        "dilation": apply_dilation(mask),
        "opening": apply_opening(mask),
        "closing": apply_closing(mask),
        "opening_then_closing": apply_closing(apply_opening(mask)),
    }
