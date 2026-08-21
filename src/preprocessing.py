import numpy as np
import cv2


def normalize_intensity(image):
    """Min-max normalization to 0-255 range."""
    img = image.astype(np.float64)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255
    return img.astype(np.uint8)


def apply_gaussian_filter(image, sigma=1.0):
    """Gaussian blur for noise reduction."""
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median_filter(image, kernel_size=3):
    """Median filter for salt-and-pepper noise reduction."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def preprocess_image(image, config=None):
    """Master preprocessing pipeline: normalize -> filter -> CLAHE.

    Returns the final image and a dict of intermediate results.
    """
    if config is None:
        config = {}

    do_normalize = config.get("normalize", True)
    filter_type = config.get("filter_type", "gaussian")
    sigma = config.get("sigma", 1.0)
    kernel_size = config.get("kernel_size", 3)
    do_clahe = config.get("clahe", True)
    clip_limit = config.get("clip_limit", 2.0)

    steps = {"original": image.copy()}

    current = image.copy()
    if do_normalize:
        current = normalize_intensity(current)
    steps["normalized"] = current.copy()

    if filter_type == "gaussian":
        current = apply_gaussian_filter(current, sigma)
    elif filter_type == "median":
        current = apply_median_filter(current, kernel_size)
    steps["filtered"] = current.copy()

    if do_clahe:
        current = apply_clahe(current, clip_limit)
    steps["enhanced"] = current.copy()

    return current, steps


def compare_filters(image):
    """Apply gaussian, median, and no filter for side-by-side comparison."""
    return {
        "none": image.copy(),
        "gaussian": apply_gaussian_filter(image, sigma=1.0),
        "median": apply_median_filter(image, kernel_size=3),
    }
