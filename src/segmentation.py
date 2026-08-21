import numpy as np
import cv2


def otsu_threshold(image):
    """Otsu's automatic thresholding."""
    threshold_val, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, threshold_val


def adaptive_threshold(image, block_size=11, C=2):
    """Adaptive thresholding for handling uneven illumination."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)


def intensity_threshold(image, low_percentile=75, high_percentile=100):
    """Keep only pixels within a given intensity percentile range."""
    low_val = np.percentile(image, low_percentile)
    high_val = np.percentile(image, high_percentile)
    mask = np.zeros_like(image)
    mask[(image >= low_val) & (image <= high_val)] = 255
    return mask


def apply_skull_strip_approximation(image):
    """Approximate skull stripping by removing dark background and bright skull edges.

    Skull stripping isolates the brain parenchyma from the surrounding skull
    and background, which reduces false positives in tumor segmentation.
    """
    _, brain_region = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_region = cv2.morphologyEx(brain_region, cv2.MORPH_CLOSE, kernel, iterations=3)
    brain_region = cv2.morphologyEx(brain_region, cv2.MORPH_OPEN, kernel, iterations=2)

    h, w = image.shape
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.42)
    circular_mask = np.zeros_like(image)
    cv2.ellipse(circular_mask, center, (radius, int(radius * 0.95)), 0, 0, 360, 255, -1)

    combined = cv2.bitwise_and(brain_region, circular_mask)
    eroded = cv2.erode(combined, kernel, iterations=1)
    return eroded


def segment_tumor_candidate(image, method="otsu", params=None):
    """Master segmentation function.

    Applies skull stripping approximation first, then thresholding.
    """
    if params is None:
        params = {}

    brain_mask = apply_skull_strip_approximation(image)
    brain_only = cv2.bitwise_and(image, image, mask=brain_mask)

    if method == "otsu":
        binary, thresh_val = otsu_threshold(brain_only)
    elif method == "adaptive":
        block_size = params.get("block_size", 11)
        C = params.get("C", 2)
        binary = adaptive_threshold(brain_only, block_size, C)
        thresh_val = None
    elif method == "intensity":
        low_p = params.get("low_percentile", 75)
        high_p = params.get("high_percentile", 100)
        binary = intensity_threshold(brain_only, low_p, high_p)
        thresh_val = None
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    binary = cv2.bitwise_and(binary, brain_mask)

    num_labels, _ = cv2.connectedComponents(binary)

    return {
        "binary_mask": binary,
        "brain_mask": brain_mask,
        "method_used": method,
        "threshold_value": thresh_val,
        "candidate_region_count": max(0, num_labels - 1),
    }
