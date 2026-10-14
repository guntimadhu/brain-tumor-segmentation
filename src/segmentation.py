import numpy as np
import cv2


def apply_skull_strip_approximation(image):
    """Approximate skull stripping by extracting the brain region.

    Uses thresholding to separate tissue from background, then selects the
    largest connected component (the brain parenchyma) and fills internal
    holes.  The resulting mask excludes both the dark background and bright
    skull/scalp pixels at the periphery.
    """
    img = image.astype(np.uint8)

    _, binary = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    brain_mask = np.zeros_like(img)
    brain_mask[labels == largest_label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(brain_mask, contours, -1, 255, -1)

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    brain_mask = cv2.erode(brain_mask, erode_k, iterations=1)

    return brain_mask


def otsu_threshold(image):
    """Otsu's automatic thresholding."""
    img = image.astype(np.uint8)
    threshold_val, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, threshold_val


def adaptive_threshold(image, block_size=11, C=2):
    """Adaptive thresholding for handling uneven illumination."""
    img = image.astype(np.uint8)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, C,
    )


def intensity_threshold(image, brain_mask, low_percentile=75, high_percentile=100):
    """Keep only pixels within a given intensity percentile range of the brain region."""
    brain_pixels = image[brain_mask > 0]
    if len(brain_pixels) == 0:
        return np.zeros_like(image)
    low_val = np.percentile(brain_pixels, low_percentile)
    high_val = np.percentile(brain_pixels, high_percentile)
    mask = np.zeros_like(image)
    mask[(image >= low_val) & (image <= high_val) & (brain_mask > 0)] = 255
    return mask


def _refine_with_brain_mask(binary, brain_mask):
    """Intersect the thresholded result with the brain mask."""
    return cv2.bitwise_and(binary, binary, mask=brain_mask)


def segment_tumor_candidate(image, method="percentile", params=None, brain_mask=None):
    """Brain-mask the image, then threshold with one of:
    percentile (global, brain-masked), otsu (brain-masked), adaptive (Gaussian),
    intensity (percentile band; same family as percentile).

    brain_mask should be computed from the normalized image before CLAHE:
    CLAHE raises background noise above the tissue threshold."""
    if params is None:
        params = {}

    img = image.astype(np.uint8)
    if brain_mask is None:
        brain_mask = apply_skull_strip_approximation(img)
    brain_only = cv2.bitwise_and(img, img, mask=brain_mask)

    thresh_val = None

    brain_pixels = img[brain_mask > 0]

    if method == "percentile":
        binary = np.zeros_like(img)
        if len(brain_pixels) > 0:
            thresh_val = max(int(np.percentile(brain_pixels, params.get("percentile", 80))), 1)
            binary[(img > thresh_val) & (brain_mask > 0)] = 255

    elif method == "otsu":
        binary = np.zeros_like(img)
        if len(brain_pixels) > 0:
            thresh_val, _ = cv2.threshold(brain_pixels.reshape(-1, 1), 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary[(img > thresh_val) & (brain_mask > 0)] = 255

    elif method == "adaptive":
        block_size = params.get("block_size", 11)
        C = params.get("C", 2)
        if block_size % 2 == 0:
            block_size += 1
        binary = adaptive_threshold(brain_only, block_size, C)
        binary = _refine_with_brain_mask(binary, brain_mask)

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
