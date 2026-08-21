import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import os
from src.io_utils import handle_upload, get_patient_list, load_patient_data
from src.preprocessing import preprocess_image
from src.segmentation import segment_tumor_candidate
from src.morphology import apply_morphological_pipeline
from src.postprocessing import (
    filter_components_by_size,
    select_tumor_region,
    create_tumor_overlay,
)
from src.measurements import (
    estimate_tumor_volume,
    get_tumor_statistics,
    generate_slice_area_dataframe,
)
from src.evaluation import (
    evaluate_segmentation,
    evaluate_volume_set,
    plot_evaluation_charts,
)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BrainScan AI — Brain Tumor Segmentation",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --accent: #00d4aa;
    --accent-blue: #3b82f6;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --border: #1f2937;
    --bg-hover: #1a2332;
}

/* Global */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary) !important;
}
.stApp > header { background-color: transparent !important; }
section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--accent) !important;
    color: #0a0e1a !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--accent) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 14px rgba(0,212,170,0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0,212,170,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: var(--bg-card) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] label { color: var(--text-primary) !important; }

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Selectbox / Slider / Radio */
.stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
    color: var(--text-primary) !important;
}

/* DataFrame */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* Info / Warning / Error / Success */
.stAlert { border-radius: 10px !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Custom card class */
.card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    margin-bottom: 16px;
}
.card-accent {
    border-left: 4px solid #00d4aa;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #111827 0%, #0f172a 50%, #111827 100%);
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.header-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f9fafb;
    letter-spacing: -0.5px;
    margin: 0;
}
.header-subtitle {
    font-size: 0.95rem;
    color: #9ca3af;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-green { background: rgba(0,212,170,0.15); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.badge-blue  { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-grey  { background: rgba(156,163,175,0.15); color: #9ca3af; border: 1px solid rgba(156,163,175,0.3); }

/* Upload area placeholder */
.upload-placeholder {
    text-align: center;
    padding: 40px 20px;
    color: #9ca3af;
}
.upload-placeholder .icon { font-size: 3rem; margin-bottom: 12px; }

/* Status items */
.status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.85rem;
    color: #9ca3af;
}
.status-done { color: #00d4aa; }
.status-pending { color: #4b5563; }

/* Image caption fix */
[data-testid="stImage"] { border-radius: 10px !important; overflow: hidden !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "patient": None,
    "slice_idx": 0,
    "pp_result": None,
    "pp_steps": None,
    "pp_config": None,
    "seg_result": None,
    "morph_result": None,
    "tumor_mask": None,
    "overlay": None,
    "pred_masks": None,
    "meas_stats": None,
    "meas_df": None,
    "eval_result": None,
    "single_eval": None,
    "report_text": None,
    "step_preprocessing": False,
    "step_segmentation": False,
    "step_morphology": False,
    "step_measurements": False,
    "step_evaluation": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <div>
        <div class="header-title">\U0001f9e0 BrainScan AI</div>
        <div class="header-subtitle">Automated Brain Tumor Segmentation & Volume Estimation</div>
    </div>
    <div>
        <span class="badge badge-green">SDG 3</span>
        <span class="badge badge-blue">MRI Analysis</span>
        <span class="badge badge-grey">Research Prototype</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## \U0001f9e0 BrainScan AI")
    st.divider()

    # Upload status
    st.markdown("#### Upload Status")
    patient = st.session_state["patient"]
    if patient:
        st.markdown('<div class="status-item status-done">✅ File loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-item status-pending">❌ No file loaded</div>', unsafe_allow_html=True)
    st.divider()

    # Patient info
    if patient:
        st.markdown("#### Patient Info")
        st.markdown(f"**ID:** `{patient['patient_id']}`")
        st.markdown(f"**Slices:** {patient['slice_count']}")
        st.markdown(f"**Size:** {patient['image_shape'][0]} x {patient['image_shape'][1]}")
        st.markdown(f"**Ground truth:** {'Yes' if patient['has_masks'] else 'No'}")
        st.divider()

    # Processing status
    st.markdown("#### Processing Status")
    steps = [
        ("Preprocessing", st.session_state["step_preprocessing"]),
        ("Segmentation", st.session_state["step_segmentation"]),
        ("Morphology", st.session_state["step_morphology"]),
        ("Measurements", st.session_state["step_measurements"]),
        ("Evaluation", st.session_state["step_evaluation"]),
    ]
    for name, done in steps:
        icon = "✅" if done else "⬜"
        css = "status-done" if done else "status-pending"
        st.markdown(f'<div class="status-item {css}">{icon} {name}</div>', unsafe_allow_html=True)
    st.divider()

    # Settings
    st.markdown("#### Settings")
    pixel_spacing = st.number_input("Pixel spacing (mm)", 0.1, 10.0, 1.0, 0.1, key="cfg_px")
    slice_thickness = st.number_input("Slice thickness (mm)", 0.5, 20.0, 5.0, 0.5, key="cfg_st")
    st.divider()

    # How it works
    with st.expander("How it works"):
        st.markdown(
            "**1. Upload** — Single MRI image or ZIP of slices.\n\n"
            "**2. Preprocess** — Normalize, filter, enhance contrast.\n\n"
            "**3. Segment** — Otsu / Adaptive / Intensity thresholding + skull stripping.\n\n"
            "**4. Morphology** — Refine mask with opening/closing/erosion/dilation.\n\n"
            "**5. Measure** — Per-slice area + pseudo-3D volume estimation.\n\n"
            "**6. Evaluate** — Dice, IoU, Precision, Recall vs ground truth.\n\n"
            "---\n"
            "*No ML models or external APIs. Classical image processing only.*"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def to_display(img):
    """Ensure image is properly normalized uint8 for st.image()."""
    if img is None:
        return None
    out = img.copy().astype(np.float64)
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-8:
        out = (out - mn) / (mx - mn) * 255.0
    elif mx > 0:
        out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def show_img(img, caption="", use_container_width=True):
    if img is not None:
        st.image(to_display(img), caption=caption, use_container_width=use_container_width, clamp=True)


def metric_color(value):
    if value > 0.7:
        return "#00d4aa"
    elif value > 0.5:
        return "#f59e0b"
    return "#ef4444"


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_upload, tab_preprocess, tab_segment, tab_measure, tab_eval, tab_report = st.tabs([
    "\U0001f4c2 Upload & Preview",
    "\U0001f527 Preprocessing",
    "\U0001f9e9 Segmentation & Morphology",
    "\U0001f4cf Measurements",
    "\U0001f4ca Evaluation",
    "\U0001f4dd Report",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & PREVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload Brain MRI Data")

    uploaded = st.file_uploader(
        "Upload a single MRI image or a ZIP file containing multiple slices",
        type=["tif", "tiff", "jpg", "jpeg", "png", "zip"],
        help="For evaluation, include _mask files alongside slices in the ZIP.",
        key="uploader",
    )

    def _reset_downstream():
        for k in ["pp_result", "pp_steps", "pp_config", "seg_result", "morph_result",
                   "tumor_mask", "overlay", "pred_masks", "meas_stats", "meas_df",
                   "eval_result", "single_eval", "report_text"]:
            st.session_state[k] = None
        for k in ["step_preprocessing", "step_segmentation", "step_morphology",
                   "step_measurements", "step_evaluation"]:
            st.session_state[k] = False
        st.session_state["slice_idx"] = 0

    if uploaded is not None:
        st.write(f"Debug: file type = {uploaded.type}, size = {uploaded.size} bytes")

        upload_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_last_upload_key") != upload_key:
            try:
                with st.spinner("Loading MRI data..."):
                    patient_data = handle_upload(uploaded)
                st.session_state["patient"] = patient_data
                st.session_state["_last_upload_key"] = upload_key
                st.session_state["_last_local_key"] = None
                _reset_downstream()
                st.success(f"Loaded **{patient_data['slice_count']}** slice(s) from **{patient_data['patient_id']}**")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

    # ── Local dataset browser ────────────────────────────────────────────
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    local_patients = get_patient_list(DATA_DIR)

    if local_patients:
        st.divider()
        st.subheader("Or load from local dataset")

        patient_names = [os.path.basename(p) for p in local_patients]
        selected_name = st.selectbox(
            f"Found **{len(local_patients)}** patients in `data/raw/`",
            patient_names,
            index=None,
            placeholder="Select a patient...",
            key="local_patient_select",
        )

        if selected_name is not None:
            local_key = selected_name
            if st.session_state.get("_last_local_key") != local_key:
                try:
                    folder_path = local_patients[patient_names.index(selected_name)]
                    with st.spinner(f"Loading {selected_name}..."):
                        patient_data = load_patient_data(folder_path)
                    st.session_state["patient"] = patient_data
                    st.session_state["_last_local_key"] = local_key
                    st.session_state["_last_upload_key"] = None
                    _reset_downstream()
                    st.success(f"Loaded **{patient_data['slice_count']}** slices from **{selected_name}** "
                               f"({'with' if patient_data['has_masks'] else 'no'} ground truth masks)")
                except Exception as e:
                    st.error(f"Failed to load patient: {e}")
    elif st.session_state["patient"] is None and uploaded is None:
        st.markdown("""
        <div class="card" style="text-align:center; padding:50px 20px;">
            <div style="font-size:4rem; margin-bottom:16px;">\U0001f9e0</div>
            <div style="font-size:1.1rem; color:#f9fafb; font-weight:600; margin-bottom:8px;">
                No MRI data uploaded yet
            </div>
            <div style="color:#9ca3af; font-size:0.85rem; max-width:500px; margin:auto;">
                Upload a <b>.tif</b>, <b>.png</b>, or <b>.jpg</b> brain MRI image, or a <b>.zip</b>
                file containing multiple slices. Or place patient folders in <code>data/raw/</code>
                to browse locally.
            </div>
        </div>
        """, unsafe_allow_html=True)

    patient = st.session_state["patient"]
    if patient:
        # Image info bar
        info_cols = st.columns(4)
        info_cols[0].markdown(f"**Filename:** `{patient['image_files'][0]}`")
        info_cols[1].markdown(f"**Dimensions:** {patient['image_shape'][0]} x {patient['image_shape'][1]}")
        info_cols[2].markdown(f"**Slices:** {patient['slice_count']}")
        ext = patient['image_files'][0].rsplit('.', 1)[-1].upper() if patient['image_files'] else "N/A"
        info_cols[3].markdown(f"**Format:** {ext}")

        if patient["slice_count"] > 1:
            st.session_state["slice_idx"] = st.slider(
                "Select Slice",
                0, patient["slice_count"] - 1,
                st.session_state["slice_idx"],
                key="slice_slider",
            )
            st.caption(f"Slice {st.session_state['slice_idx'] + 1} of {patient['slice_count']}")

        idx = st.session_state["slice_idx"]
        col_img, col_mask = st.columns(2)
        with col_img:
            st.markdown("""<div class="card card-accent"><p style="color:#9ca3af;font-size:0.8rem;margin:0 0 8px;">ORIGINAL MRI</p></div>""", unsafe_allow_html=True)
            show_img(patient["images"][idx], f"Slice {idx + 1}")
        with col_mask:
            if patient["has_masks"] and idx < len(patient["masks"]):
                st.markdown("""<div class="card card-accent"><p style="color:#9ca3af;font-size:0.8rem;margin:0 0 8px;">GROUND TRUTH MASK</p></div>""", unsafe_allow_html=True)
                show_img(patient["masks"][idx] * 255, f"Mask {idx + 1}")
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:40px 20px;">
                    <div style="font-size:1.5rem;margin-bottom:8px;">\U0001f4cb</div>
                    <div style="color:#9ca3af;font-size:0.85rem;">No ground truth mask available for this slice</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
with tab_preprocess:
    st.markdown("### Image Preprocessing")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload & Preview** tab first.")
    else:
        idx = st.session_state["slice_idx"]
        img = patient["images"][idx]

        col_ctrl, col_result = st.columns([1, 2])

        with col_ctrl:
            st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
            st.markdown("**Filter Settings**")
            filter_type = st.radio("Filter type", ["gaussian", "median", "none"], key="pp_filter", horizontal=True)

            sigma = 1.0
            kernel_size = 3
            if filter_type == "gaussian":
                sigma = st.slider("Sigma", 0.5, 3.0, 1.0, 0.1, key="pp_sigma")
            elif filter_type == "median":
                kernel_size = st.select_slider("Kernel size", [3, 5, 7], value=3, key="pp_kern")

            st.markdown("**Contrast Enhancement**")
            do_clahe = st.checkbox("Enable CLAHE", True, key="pp_clahe")
            clip_limit = 2.0
            if do_clahe:
                clip_limit = st.slider("Clip limit", 1.0, 4.0, 2.0, 0.1, key="pp_clip")
            st.markdown('</div>', unsafe_allow_html=True)

            run_pp = st.button("▶ Run Preprocessing", key="btn_pp", use_container_width=True)

        config = {
            "normalize": True,
            "filter_type": filter_type,
            "sigma": sigma,
            "kernel_size": kernel_size,
            "clahe": do_clahe,
            "clip_limit": clip_limit,
        }

        if run_pp:
            try:
                with st.spinner("Applying preprocessing pipeline..."):
                    result, steps = preprocess_image(img, config)
                st.session_state["pp_result"] = result
                st.session_state["pp_steps"] = steps
                st.session_state["pp_config"] = config
                st.session_state["step_preprocessing"] = True
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")

        with col_result:
            if st.session_state["pp_steps"] is not None:
                steps = st.session_state["pp_steps"]
                r1c1, r1c2 = st.columns(2)
                r2c1, r2c2 = st.columns(2)
                with r1c1:
                    show_img(steps["original"], "Original")
                with r1c2:
                    show_img(steps["normalized"], "Normalized")
                with r2c1:
                    show_img(steps["filtered"], "Filtered")
                with r2c2:
                    show_img(steps["enhanced"], "Enhanced (CLAHE)")
                st.success("Preprocessing complete")
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:60px 20px;">
                    <div style="color:#9ca3af;">Configure settings and click <b>Run Preprocessing</b> to see results.</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTATION & MORPHOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with tab_segment:
    st.markdown("### Tumor Segmentation")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload & Preview** tab first.")
    elif not st.session_state["step_preprocessing"]:
        st.warning("Run **Preprocessing** (Tab 2) first.")
    else:
        idx = st.session_state["slice_idx"]
        img = st.session_state["pp_result"]

        # Segmentation method cards
        st.markdown("**Select Method**")
        seg_cols = st.columns(3)
        with seg_cols[0]:
            st.markdown("""<div class="card"><b>Otsu Thresholding</b><br><span style="color:#9ca3af;font-size:0.8rem;">
            Automatic global threshold based on image histogram bimodality.</span></div>""", unsafe_allow_html=True)
        with seg_cols[1]:
            st.markdown("""<div class="card"><b>Adaptive Thresholding</b><br><span style="color:#9ca3af;font-size:0.8rem;">
            Local region-based threshold for uneven illumination.</span></div>""", unsafe_allow_html=True)
        with seg_cols[2]:
            st.markdown("""<div class="card"><b>Intensity Based</b><br><span style="color:#9ca3af;font-size:0.8rem;">
            Percentile-based intensity range selection for bright tumors.</span></div>""", unsafe_allow_html=True)

        sc1, sc2 = st.columns([1, 1])
        with sc1:
            seg_method = st.radio("Method", ["otsu", "adaptive", "intensity"], key="seg_method", horizontal=True, label_visibility="collapsed")
        with sc2:
            params = {}
            if seg_method == "intensity":
                params["low_percentile"] = st.slider("Low percentile", 50, 95, 75, key="seg_low")
                params["high_percentile"] = st.slider("High percentile", 80, 100, 100, key="seg_high")

        run_seg = st.button("▶ Run Segmentation", key="btn_seg", use_container_width=False)

        if run_seg:
            try:
                with st.spinner("Running tumor segmentation..."):
                    seg_result = segment_tumor_candidate(img, method=seg_method, params=params)
                st.session_state["seg_result"] = seg_result
                st.session_state["step_segmentation"] = True
                st.success(f"Found **{seg_result['candidate_region_count']}** candidate region(s)")
            except Exception as e:
                st.error(f"Segmentation failed: {e}")

        if st.session_state["seg_result"] is not None:
            seg = st.session_state["seg_result"]
            cols = st.columns(3)
            with cols[0]:
                show_img(img, "Preprocessed Input")
            with cols[1]:
                show_img(seg["binary_mask"], "Candidate Mask")
            with cols[2]:
                show_img(seg["brain_mask"], "Skull-Stripped Brain")

        # Morphology section
        st.divider()
        st.markdown("### Morphological Refinement")

        if not st.session_state["step_segmentation"]:
            st.warning("Run **Segmentation** above first.")
        else:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                morph_op = st.selectbox("Operation", [
                    "opening_then_closing", "opening", "closing", "erosion", "dilation"
                ], key="morph_op")
            with mc2:
                morph_kern = st.select_slider("Kernel Size", [3, 5, 7], value=5, key="morph_kern",
                                              format_func=lambda x: f"{x}×{x}")
            with mc3:
                morph_iter = st.select_slider("Iterations", [1, 2, 3], value=1, key="morph_iter")

            run_morph = st.button("▶ Run Morphology", key="btn_morph", use_container_width=False)

            if run_morph:
                try:
                    mask_in = st.session_state["seg_result"]["binary_mask"]
                    config = {"operation": morph_op, "kernel_size": morph_kern, "iterations": morph_iter}
                    with st.spinner("Applying morphological operations..."):
                        morph_result = apply_morphological_pipeline(mask_in, config)
                        filtered = filter_components_by_size(morph_result["output_mask"], min_area=100)
                        tumor = select_tumor_region(filtered, strategy="largest", min_area=50)
                        overlay = create_tumor_overlay(patient["images"][idx], tumor)
                    st.session_state["morph_result"] = morph_result
                    st.session_state["tumor_mask"] = tumor
                    st.session_state["overlay"] = overlay
                    st.session_state["step_morphology"] = True
                    st.success(f"Noise removed: **{morph_result['noise_removed_pixels']}** pixels")
                except Exception as e:
                    st.error(f"Morphology failed: {e}")

            if st.session_state["morph_result"] is not None:
                morph = st.session_state["morph_result"]

                # Step-by-step pipeline
                st.markdown("**Pipeline Steps**")
                n_steps = len(morph["intermediate_steps"])
                step_cols = st.columns(n_steps + 1)
                with step_cols[0]:
                    show_img(morph["input_mask"], "Input Mask")
                for i, (label, step_img) in enumerate(morph["intermediate_steps"]):
                    with step_cols[i + 1]:
                        show_img(step_img, f"→ {label}")

                st.markdown("**Final Result**")
                cols = st.columns(3)
                with cols[0]:
                    show_img(patient["images"][idx], "Original MRI")
                with cols[1]:
                    show_img(st.session_state["tumor_mask"], "Tumor Mask")
                with cols[2]:
                    show_img(st.session_state["overlay"], "Overlay")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_measure:
    st.markdown("### Tumor Measurements")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload & Preview** tab first.")
    elif not st.session_state["step_morphology"]:
        st.warning("Complete **Segmentation & Morphology** (Tab 3) first.")
    else:
        run_meas = st.button("▶ Calculate Measurements", key="btn_meas", use_container_width=False)

        if run_meas:
            try:
                with st.spinner("Processing all slices — this may take a moment..."):
                    pp_config = st.session_state.get("pp_config", {
                        "normalize": True, "filter_type": "gaussian",
                        "sigma": 1.0, "clahe": True, "clip_limit": 2.0
                    })
                    morph_cfg = {
                        "operation": st.session_state.get("morph_op", "opening_then_closing"),
                        "kernel_size": st.session_state.get("morph_kern", 5),
                        "iterations": st.session_state.get("morph_iter", 1),
                    }
                    seg_method = st.session_state.get("seg_method", "otsu")

                    pred_masks = []
                    for img_slice in patient["images"]:
                        pp_img, _ = preprocess_image(img_slice, pp_config)
                        seg = segment_tumor_candidate(pp_img, method=seg_method)
                        morph = apply_morphological_pipeline(seg["binary_mask"], morph_cfg)
                        filtered = filter_components_by_size(morph["output_mask"], min_area=100)
                        tumor = select_tumor_region(filtered, strategy="largest", min_area=50)
                        pred_masks.append(tumor)

                    px_sp = st.session_state.get("cfg_px", 1.0)
                    sl_th = st.session_state.get("cfg_st", 5.0)
                    stats = get_tumor_statistics(pred_masks, px_sp, sl_th)
                    df = generate_slice_area_dataframe(pred_masks, px_sp)

                st.session_state["pred_masks"] = pred_masks
                st.session_state["meas_stats"] = stats
                st.session_state["meas_df"] = df
                st.session_state["step_measurements"] = True
                st.success("Measurements complete")
            except Exception as e:
                st.error(f"Measurement failed: {e}")

        if st.session_state["meas_stats"] is not None:
            stats = st.session_state["meas_stats"]
            vol = stats["volume"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tumor Volume", f"{vol['tumor_volume_cm3']:.2f} cm³")
            m2.metric("Tumor Volume", f"{vol['tumor_volume_ml']:.2f} mL")
            m3.metric("Affected Slices", f"{vol['affected_slices']} / {vol['total_slices']}")
            m4.metric("Mean Area", f"{stats['mean_area_mm2']:.1f} mm²")

            st.caption(vol["note"])

            # Chart
            df = st.session_state["meas_df"]
            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#111827")
            ax.bar(df["Slice"], df["Area mm2"], color="#00d4aa", edgecolor="#00b894", linewidth=0.5)
            ax.set_xlabel("Slice Index", color="#9ca3af")
            ax.set_ylabel("Tumor Area (mm²)", color="#9ca3af")
            ax.set_title("Tumor Area per Slice", color="#f9fafb", fontweight="bold")
            ax.tick_params(colors="#9ca3af")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#1f2937")
            ax.spines["left"].set_color("#1f2937")
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Slice Details"):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button("\U0001f4be Download CSV", csv, "slice_measurements.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Segmentation Evaluation")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload & Preview** tab first.")
    elif not patient["has_masks"]:
        st.markdown("""
        <div class="card" style="text-align:center;padding:40px 20px;border-left:4px solid #f59e0b;">
            <div style="font-size:1.5rem;margin-bottom:8px;">⚠️</div>
            <div style="color:#f59e0b;font-weight:600;font-size:1rem;margin-bottom:8px;">
                Ground Truth Masks Required
            </div>
            <div style="color:#9ca3af;font-size:0.85rem;max-width:500px;margin:auto;">
                Upload a ZIP file containing both MRI slices and corresponding
                <code>_mask</code> files to enable evaluation metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state["pred_masks"] is None:
        st.warning("Run **Measurements** (Tab 4) first to generate predicted masks.")
    else:
        run_eval = st.button("▶ Run Evaluation", key="btn_eval", use_container_width=False)

        if run_eval:
            try:
                pred_masks = st.session_state["pred_masks"]
                true_masks = [m * 255 for m in patient["masks"]]
                n = min(len(pred_masks), len(true_masks))
                with st.spinner("Computing evaluation metrics..."):
                    vol_eval = evaluate_volume_set(pred_masks[:n], true_masks[:n])
                    single_eval = vol_eval["summary"]
                st.session_state["eval_result"] = vol_eval
                st.session_state["single_eval"] = single_eval
                st.session_state["step_evaluation"] = True
                st.success("Evaluation complete")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

        if st.session_state["single_eval"] is not None:
            ev = st.session_state["single_eval"]
            vol_eval = st.session_state["eval_result"]

            # Metric cards with color
            e1, e2, e3, e4, e5 = st.columns(5)
            for col, name, key in [
                (e1, "Dice", "dice"), (e2, "IoU", "iou"), (e3, "Precision", "precision"),
                (e4, "Recall", "recall"), (e5, "Specificity", "specificity")
            ]:
                val = ev[key]
                color = metric_color(val)
                with col:
                    st.metric(name, f"{val:.4f}")
                    st.markdown(f'<div style="height:3px;background:{color};border-radius:2px;margin-top:-8px;"></div>', unsafe_allow_html=True)

            # Interpretation banner
            d = ev["dice"]
            if d > 0.7:
                st.success(f"✅ {ev['interpretation']}")
            elif d > 0.5:
                st.warning(f"⚠️ {ev['interpretation']}")
            else:
                st.error(f"❌ {ev['interpretation']}")

            # Charts
            col_bar, col_cm = st.columns(2)
            with col_bar:
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                fig_bar.patch.set_facecolor("#0a0e1a")
                ax_bar.set_facecolor("#111827")
                per_dice = vol_eval["per_slice_dice"]
                colors = [metric_color(v) for v in per_dice]
                ax_bar.bar(range(len(per_dice)), per_dice, color=colors)
                ax_bar.set_xlabel("Slice", color="#9ca3af")
                ax_bar.set_ylabel("Dice", color="#9ca3af")
                ax_bar.set_title("Per-Slice Dice Coefficient", color="#f9fafb", fontweight="bold")
                ax_bar.set_ylim(0, 1.05)
                ax_bar.tick_params(colors="#9ca3af")
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)
                ax_bar.spines["bottom"].set_color("#1f2937")
                ax_bar.spines["left"].set_color("#1f2937")
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col_cm:
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                fig_cm.patch.set_facecolor("#0a0e1a")
                ax_cm.set_facecolor("#111827")
                cm = np.array([[ev["TP"], ev["FN"]], [ev["FP"], ev["TN"]]])
                im = ax_cm.imshow(cm, cmap="YlGnBu")
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(["Positive", "Negative"], color="#9ca3af")
                ax_cm.set_yticklabels(["Positive", "Negative"], color="#9ca3af")
                ax_cm.set_xlabel("Ground Truth", color="#9ca3af")
                ax_cm.set_ylabel("Predicted", color="#9ca3af")
                ax_cm.set_title("Confusion Matrix", color="#f9fafb", fontweight="bold")
                for i in range(2):
                    for j in range(2):
                        ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold", color="#0a0e1a")
                ax_cm.tick_params(colors="#9ca3af")
                st.pyplot(fig_cm)
                plt.close(fig_cm)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("### Analysis Report")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload & Preview** tab first.")
    else:
        # Completion checklist
        st.markdown("**Completion Status**")
        checks = [
            ("Upload", True),
            ("Preprocessing", st.session_state["step_preprocessing"]),
            ("Segmentation", st.session_state["step_segmentation"]),
            ("Morphology", st.session_state["step_morphology"]),
            ("Measurements", st.session_state["step_measurements"]),
            ("Evaluation", st.session_state["step_evaluation"]),
        ]
        cols_check = st.columns(len(checks))
        for col, (name, done) in zip(cols_check, checks):
            with col:
                icon = "✅" if done else "⬜"
                st.markdown(f"{icon} **{name}**")

        st.divider()

        gen_report = st.button("▶ Generate Report", key="btn_report", use_container_width=False)

        if gen_report:
            lines = []
            lines.append("=" * 60)
            lines.append("  BRAINSCAN AI — ANALYSIS REPORT")
            lines.append("=" * 60)
            lines.append(f"  Patient ID     : {patient['patient_id']}")
            lines.append(f"  Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Total Slices   : {patient['slice_count']}")
            lines.append(f"  Image Size     : {patient['image_shape'][0]} x {patient['image_shape'][1]}")
            lines.append(f"  Ground Truth   : {'Available' if patient['has_masks'] else 'Not available'}")
            lines.append("")

            lines.append("-" * 60)
            lines.append("  PREPROCESSING")
            lines.append("-" * 60)
            pp_cfg = st.session_state.get("pp_config")
            if pp_cfg:
                lines.append(f"  Filter         : {pp_cfg.get('filter_type', 'N/A')}")
                if pp_cfg.get("filter_type") == "gaussian":
                    lines.append(f"  Sigma          : {pp_cfg.get('sigma', 'N/A')}")
                elif pp_cfg.get("filter_type") == "median":
                    lines.append(f"  Kernel Size    : {pp_cfg.get('kernel_size', 'N/A')}")
                lines.append(f"  CLAHE          : {'Enabled' if pp_cfg.get('clahe') else 'Disabled'}")
                if pp_cfg.get("clahe"):
                    lines.append(f"  Clip Limit     : {pp_cfg.get('clip_limit', 'N/A')}")
            else:
                lines.append("  (not yet run)")
            lines.append("")

            lines.append("-" * 60)
            lines.append("  SEGMENTATION")
            lines.append("-" * 60)
            seg = st.session_state.get("seg_result")
            if seg:
                lines.append(f"  Method         : {seg['method_used']}")
                lines.append(f"  Threshold      : {seg.get('threshold_value', 'N/A')}")
                lines.append(f"  Candidates     : {seg['candidate_region_count']}")
            else:
                lines.append("  (not yet run)")
            lines.append("")

            lines.append("-" * 60)
            lines.append("  MORPHOLOGY")
            lines.append("-" * 60)
            morph = st.session_state.get("morph_result")
            if morph:
                cfg = morph["config_used"]
                lines.append(f"  Operation      : {cfg['operation']}")
                lines.append(f"  Kernel         : {cfg['kernel_size']}x{cfg['kernel_size']} {cfg['kernel_shape']}")
                lines.append(f"  Iterations     : {cfg['iterations']}")
                lines.append(f"  Noise Removed  : {morph['noise_removed_pixels']} pixels")
            else:
                lines.append("  (not yet run)")
            lines.append("")

            lines.append("-" * 60)
            lines.append("  MEASUREMENTS")
            lines.append("-" * 60)
            if st.session_state["meas_stats"]:
                stats = st.session_state["meas_stats"]
                vol = stats["volume"]
                lines.append(f"  Tumor Volume   : {vol['tumor_volume_cm3']:.2f} cm3  ({vol['tumor_volume_ml']:.2f} mL)")
                lines.append(f"  Affected       : {vol['affected_slices']} / {vol['total_slices']} slices")
                lines.append(f"  Mean Area      : {stats['mean_area_mm2']:.1f} mm2")
                lines.append(f"  Max Area       : {stats['max_area_mm2']:.1f} mm2")
                lines.append(f"  Min Area       : {stats['min_area_mm2']:.1f} mm2")
                lines.append(f"  Std Dev        : {stats['std_area_mm2']:.1f} mm2")
                lines.append(f"  Pixel Spacing  : {vol['pixel_spacing_mm']} mm")
                lines.append(f"  Slice Thickness: {vol['slice_thickness_mm']} mm")
            else:
                lines.append("  (not yet run)")
            lines.append("")

            lines.append("-" * 60)
            lines.append("  EVALUATION")
            lines.append("-" * 60)
            ev = st.session_state.get("single_eval")
            if ev:
                lines.append(f"  Dice           : {ev['dice']:.4f}")
                lines.append(f"  IoU            : {ev['iou']:.4f}")
                lines.append(f"  Precision      : {ev['precision']:.4f}")
                lines.append(f"  Recall         : {ev['recall']:.4f}")
                lines.append(f"  Specificity    : {ev['specificity']:.4f}")
                lines.append(f"  Assessment     : {ev['interpretation']}")
            else:
                lines.append("  (not available — no ground truth or not yet run)")
            lines.append("")

            lines.append("-" * 60)
            lines.append("  LIMITATIONS")
            lines.append("-" * 60)
            lines.append("  - Volume estimated from 2D slices with assumed spatial parameters.")
            lines.append("  - Classical thresholding — no deep learning model used.")
            lines.append("  - Skull stripping is approximate (elliptical mask + threshold).")
            lines.append("  - Dice scores for classical methods typically range 0.4-0.7.")
            lines.append("  - Best accuracy on high-contrast (high-grade glioma) tumors.")
            lines.append("")
            lines.append("=" * 60)
            lines.append("  Generated by BrainScan AI — Research Prototype")
            lines.append("=" * 60)

            report_text = "\n".join(lines)
            st.session_state["report_text"] = report_text
            st.success("Report generated successfully")

        if st.session_state["report_text"] is not None:
            st.code(st.session_state["report_text"], language=None)

            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button(
                    "\U0001f4c4 Download TXT Report",
                    st.session_state["report_text"],
                    f"brainscan_report_{patient['patient_id']}.txt",
                    "text/plain",
                    use_container_width=True,
                )
            with dl2:
                if st.session_state["meas_df"] is not None:
                    csv = st.session_state["meas_df"].to_csv(index=False)
                    st.download_button(
                        "\U0001f4ca Download CSV Data",
                        csv,
                        f"brainscan_data_{patient['patient_id']}.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.button("\U0001f4ca Download CSV Data", disabled=True, use_container_width=True, key="csv_disabled")
            with dl3:
                st.button("\U0001f4cb Download PDF", disabled=True, use_container_width=True, help="Coming soon", key="pdf_placeholder")
