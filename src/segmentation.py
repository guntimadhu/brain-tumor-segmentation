import numpy as np
import cv2


def apply_skull_strip_approximation(image):
    """Approximate skull stripping by removing dark background and bright skull edges."""
    img = image.astype(np.uint8)

    _, brain_mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)

    return brain_mask


def otsu_threshold(image):
    """Otsu's automatic thresholding."""
    img = image.astype(np.uint8)
    threshold_val, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, threshold_val


def adaptive_threshold(image, block_size=11, C=2):
    """Adaptive thresholding for handling uneven illumination."""
    img = image.astype(np.uint8)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)


def intensity_threshold(image, brain_mask, low_percentile=75, high_percentile=100):
    """Keep only pixels within a given intensity percentile range of the brain region."""
    brain_pixels = image[brain_mask > 0]
    if len(brain_pixels) == 0:
        return np.zeros_like(image)
    low_val = np.percentile(brain_pixels, low_percentile)
    high_val = np.percentile(brain_pixels, high_percentile)
    mask = np.zeros_like(image)
    mask[(image >= low_val) & (image <= high_val)] = 255
    return mask


def segment_tumor_candidate(image, method="otsu", params=None):
    """Master segmentation function.

    Applies skull stripping approximation first, then thresholding.
    """
    if params is None:
        params = {}

    img = image.astype(np.uint8)

    brain_mask = apply_skull_strip_approximation(img)
    brain_only = cv2.bitwise_and(img, img, mask=brain_mask)

    thresh_val = None

    if method == "otsu":
        binary, thresh_val = otsu_threshold(brain_only)
        binary = cv2.bitwise_and(binary, binary, mask=brain_mask)

    elif method == "adaptive":
        block_size = params.get("block_size", 11)
        C = params.get("C", 2)
        if block_size % 2 == 0:
            block_size += 1
        binary = adaptive_threshold(brain_only, block_size, C)
        binary = cv2.bitwise_and(binary, binary, mask=brain_mask)

    elif method == "intensity":
        low_p = params.get("low_percentile", 75)
        high_p = params.get("high_percentile", 100)
        binary = intensity_threshold(brain_only, brain_mask, low_p, high_p)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    candidate_count = int(np.sum(binary > 0))

    return {
        "binary_mask": binary,
        "brain_mask": brain_mask,
        "method_used": method,
        "threshold_value": thresh_val,
        "candidate_region_count": candidate_count,
    }
