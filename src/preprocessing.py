import numpy as np
import cv2


def normalize_intensity(image):
    """Min-max normalization to 0-255 range."""
    img = image.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-8:
        img = (img - mn) / (mx - mn) * 255.0
    return img.astype(np.uint8)


def apply_gaussian_filter(image, sigma=1.0):
    """Gaussian blur for noise reduction."""
    img = image.astype(np.uint8)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_median_filter(image, kernel_size=3):
    """Median filter for salt-and-pepper noise reduction."""
    img = image.astype(np.uint8)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE contrast enhancement."""
    img = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def preprocess_image(image, config=None):
    """Master preprocessing pipeline: normalize -> filter -> CLAHE.

    Returns the final image and a dict of intermediate results.
    """
    if config is None:
        config = {
            "normalize": True,
            "filter_type": "gaussian",
            "sigma": 1.0,
            "kernel_size": 3,
            "clahe": True,
            "clip_limit": 2.0,
        }

    results = {"original": image.copy()}

    img = image.copy()

    if config.get("normalize", True):
        img = normalize_intensity(img)
    results["normalized"] = img.copy()

    filter_type = config.get("filter_type", "gaussian")
    if filter_type == "gaussian":
        img = apply_gaussian_filter(img, config.get("sigma", 1.0))
    elif filter_type == "median":
        img = apply_median_filter(img, config.get("kernel_size", 3))
    results["filtered"] = img.copy()

    if config.get("clahe", True):
        img = apply_clahe(img, config.get("clip_limit", 2.0))
    results["enhanced"] = img.copy()

    return img, results


def compare_filters(image):
    """Apply gaussian, median, and no filter for side-by-side comparison."""
    return {
        "none": image.copy(),
        "gaussian": apply_gaussian_filter(image, sigma=1.0),
        "median": apply_median_filter(image, kernel_size=3),
    }
