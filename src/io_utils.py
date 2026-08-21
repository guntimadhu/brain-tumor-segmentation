import os
import io
import zipfile
import tempfile
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS) and not filename.startswith("__")


def load_single_image(image_path):
    """Load a TIF/PNG image as a grayscale numpy array normalized to 0-255 uint8."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)


def load_single_mask(mask_path):
    """Load a TIF/PNG mask as a uint8 array with values 0 or 255."""
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    img = Image.open(mask_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.max() <= 1:
        arr = arr * 255
    return arr


def _image_from_bytes(data):
    img = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(img, dtype=np.float64)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)


def _mask_from_bytes(data):
    img = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if arr.max() <= 1:
        arr = arr * 255
    return arr


def load_from_uploaded_file(uploaded_file):
    """Handle a single uploaded image from st.file_uploader."""
    uploaded_file.seek(0)
    data = uploaded_file.read()
    if len(data) == 0:
        raise ValueError(f"Uploaded file '{uploaded_file.name}' is empty (0 bytes).")
    name = uploaded_file.name
    img = _image_from_bytes(data)
    return {
        "images": [img],
        "masks": [],
        "slice_count": 1,
        "patient_id": os.path.splitext(name)[0],
        "has_masks": False,
        "image_shape": img.shape,
        "image_files": [name],
        "mask_files": [],
        "tumor_slices": [],
    }


def load_from_zip(zip_file):
    """Extract a ZIP file containing MRI slices (and optional _mask files)."""
    zip_file.seek(0)
    data = zip_file.read()
    image_names = []
    mask_names = []
    file_data = {}

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = os.path.basename(info.filename)
            if not _is_image_file(basename):
                continue
            raw = zf.read(info.filename)
            file_data[basename] = raw
            if "_mask" in basename.lower():
                mask_names.append(basename)
            else:
                image_names.append(basename)

    image_names.sort()
    mask_names.sort()

    if not image_names:
        raise ValueError("No image files found inside the ZIP archive.")

    images = [_image_from_bytes(file_data[n]) for n in image_names]
    masks = [_mask_from_bytes(file_data[n]) for n in mask_names] if mask_names else []

    tumor_slices = [i for i, m in enumerate(masks) if np.any(m > 0)]
    zip_name = os.path.splitext(zip_file.name)[0]
    return {
        "images": images,
        "masks": masks,
        "slice_count": len(images),
        "patient_id": zip_name,
        "has_masks": len(masks) > 0,
        "image_shape": images[0].shape,
        "image_files": image_names,
        "mask_files": mask_names,
        "tumor_slices": tumor_slices,
    }


def handle_upload(uploaded_file):
    """Detect file type and route to the correct loader."""
    name = uploaded_file.name.lower()
    if name.endswith(".zip"):
        return load_from_zip(uploaded_file)
    elif name.endswith(IMAGE_EXTENSIONS):
        return load_from_uploaded_file(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def load_patient_data(patient_folder_path):
    """Load all slices and masks for a patient folder."""
    if not os.path.isdir(patient_folder_path):
        raise NotADirectoryError(f"Patient folder not found: {patient_folder_path}")

    all_files = sorted(os.listdir(patient_folder_path))
    image_files = []
    mask_files = []

    for f in all_files:
        if not _is_image_file(f):
            continue
        if "_mask" in f:
            mask_files.append(f)
        else:
            image_files.append(f)

    if not image_files:
        raise ValueError(f"No image files found in {patient_folder_path}")

    images = [load_single_image(os.path.join(patient_folder_path, f)) for f in image_files]
    masks = [load_single_mask(os.path.join(patient_folder_path, f)) for f in mask_files] if mask_files else []

    tumor_slices = [i for i, m in enumerate(masks) if np.any(m > 0)]
    patient_id = os.path.basename(patient_folder_path)
    return {
        "images": images,
        "masks": masks,
        "slice_count": len(images),
        "patient_id": patient_id,
        "has_masks": len(masks) > 0,
        "image_shape": images[0].shape,
        "image_files": image_files,
        "mask_files": mask_files,
        "tumor_slices": tumor_slices,
    }


def get_patient_list(data_dir):
    """Return list of patient folder paths that contain image files."""
    if not os.path.isdir(data_dir):
        return []
    patients = []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        has_images = any(
            _is_image_file(f) and "_mask" not in f
            for f in os.listdir(folder)
        )
        if has_images:
            patients.append(folder)
    return patients


def get_pixel_spacing():
    """Return assumed pixel spacing for the 2D Kaggle dataset.

    Since this dataset provides 2D slices without DICOM/NIfTI metadata,
    we assume standard MRI spacing values for pseudo-3D volume estimation.
    """
    return {"pixel_spacing_mm": 1.0, "slice_thickness_mm": 5.0}


def stack_slices_to_volume(images_list):
    """Stack a list of 2D arrays into a 3D numpy array (depth, height, width)."""
    if not images_list:
        raise ValueError("Cannot stack empty list of images")
    return np.stack(images_list, axis=0)
