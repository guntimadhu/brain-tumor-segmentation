import numpy as np
import cv2


def find_connected_components(mask):
    """Find all connected components in a binary mask with statistics."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_sizes = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(num_labels)]
    return {
        "num_components": num_labels - 1,
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
        "component_sizes": component_sizes[1:],
    }


def filter_components_by_size(mask, min_area=100, max_area=None):
    """Remove connected components outside the given area range."""
    cc = find_connected_components(mask)
    filtered = np.zeros_like(mask)
    for i in range(1, cc["num_components"] + 1):
        area = cc["stats"][i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered[cc["labels"] == i] = 255
    return filtered


def select_tumor_region(mask, strategy="largest", min_area=50, original_image=None):
    """Select the most likely tumor region from candidate components.

    Strategies:
      largest  — pick the component with the most pixels
      brightest — pick the component with highest mean intensity in the original image
      central  — pick the component whose centroid is closest to the image center
    """
    cc = find_connected_components(mask)
    if cc["num_components"] == 0:
        return np.zeros_like(mask)

    valid = []
    for i in range(1, cc["num_components"] + 1):
        area = cc["stats"][i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid.append(i)

    if not valid:
        return np.zeros_like(mask)

    if strategy == "largest":
        chosen = max(valid, key=lambda i: cc["stats"][i, cv2.CC_STAT_AREA])
    elif strategy == "brightest" and original_image is not None:
        chosen = max(valid, key=lambda i: float(original_image[cc["labels"] == i].mean()))
    elif strategy == "central":
        h, w = mask.shape
        cx, cy = w / 2, h / 2
        chosen = min(valid, key=lambda i: (cc["centroids"][i][0] - cx) ** 2 + (cc["centroids"][i][1] - cy) ** 2)
    else:
        chosen = max(valid, key=lambda i: cc["stats"][i, cv2.CC_STAT_AREA])

    result = np.zeros_like(mask)
    result[cc["labels"] == chosen] = 255
    return result


def extract_tumor_contour(mask):
    """Find contours and bounding box of the tumor mask."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"contours": [], "bounding_box": (0, 0, 0, 0), "perimeter": 0.0, "num_contours": 0}

    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    return {
        "contours": contours,
        "bounding_box": (x, y, w, h),
        "perimeter": perimeter,
        "num_contours": len(contours),
    }


def create_tumor_overlay(original_image, tumor_mask, color=(0, 255, 0), alpha=0.4):
    """Create a colored semi-transparent overlay of the tumor on the original MRI."""
    if len(original_image.shape) == 2:
        rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = original_image.copy()

    overlay = rgb.copy()
    binary = tumor_mask > 0
    overlay[binary] = color

    blended = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)

    contour_info = extract_tumor_contour(tumor_mask)
    if contour_info["contours"]:
        cv2.drawContours(blended, contour_info["contours"], -1, color, 2)

    return blended


def create_visualization_panel(original, preprocessed, mask, overlay):
    """Combine four images into a labeled 2x2 grid."""
    def to_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    imgs = [to_rgb(original), to_rgb(preprocessed), to_rgb(mask), overlay if len(overlay.shape) == 3 else to_rgb(overlay)]
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]

    labels = ["Original", "Preprocessed", "Mask", "Overlay"]
    for img, label in zip(imgs, labels):
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    top = np.hstack([imgs[0], imgs[1]])
    bottom = np.hstack([imgs[2], imgs[3]])
    return np.vstack([top, bottom])
