import datetime
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
from src.io_utils import handle_upload, load_mri_and_mask, get_patient_list, load_patient_data
from src.preprocessing import preprocess_image
from src.segmentation import segment_tumor_candidate
from src.morphology import apply_morphological_pipeline
from src.postprocessing import (
    filter_components_by_size,
    select_tumor_region,
    create_tumor_overlay,
)
from src.measurements import calculate_tumor_area
from src.evaluation import evaluate_segmentation, plot_evaluation_charts

st.set_page_config(
    page_title="BrainScan AI",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg-primary: #0a0e1a; --bg-card: #111827; --accent: #00d4aa;
    --accent-blue: #3b82f6; --warning: #f59e0b; --danger: #ef4444;
    --text-primary: #f9fafb; --text-secondary: #9ca3af; --border: #1f2937;
}
.stApp { background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important; }
.stApp > header { background-color: transparent !important; }

.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-card) !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important; border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important; color: var(--text-secondary) !important;
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 0.85rem !important; padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--accent) !important; color: #0a0e1a !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

[data-testid="stMetric"] {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-left: 4px solid var(--accent) !important; border-radius: 10px !important;
    padding: 16px 20px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    color: #0a0e1a !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 1rem !important; padding: 12px 32px !important;
    box-shadow: 0 4px 14px rgba(0,212,170,0.25) !important;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(0,212,170,0.4) !important; }

.stDownloadButton > button {
    background: var(--bg-card) !important; color: var(--accent) !important;
    border: 1px solid var(--accent) !important; border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
    background-color: var(--bg-card) !important; border: 2px dashed var(--accent) !important;
    border-radius: 16px !important; padding: 20px !important;
}
.stAlert { border-radius: 10px !important; }
hr { border-color: var(--border) !important; }
[data-testid="stImage"] { border-radius: 10px !important; overflow: hidden !important; }

.header-banner {
    background: linear-gradient(135deg, #111827 0%, #0f172a 50%, #111827 100%);
    border: 1px solid #1f2937; border-radius: 16px; padding: 28px 36px;
    margin-bottom: 24px; display: flex; justify-content: space-between;
    align-items: center; box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.header-title { font-size: 2rem; font-weight: 800; color: #f9fafb; margin: 0; }
.header-subtitle { font-size: 0.95rem; color: #9ca3af; margin-top: 4px; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-left: 8px; }
.badge-green { background: rgba(0,212,170,0.15); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.badge-blue  { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "patient": None, "slice_idx": 0,
    "pp_result": None, "pp_steps": None,
    "seg_result": None, "morph_result": None,
    "tumor_mask": None, "overlay": None,
    "meas_stats": None,
    "eval_single": None, "eval_chart": None,
    "report_text": None, "processed": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <div>
        <div class="header-title">\U0001f9e0 BrainScan AI</div>
        <div class="header-subtitle">Brain Tumor Segmentation & Volume Estimation</div>
    </div>
    <div>
        <span class="badge badge-green">SDG 3</span>
        <span class="badge badge-blue">Classical IP</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def to_display(img):
    if img is None:
        return None
    out = img.copy().astype(np.float64)
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-8:
        out = (out - mn) / (mx - mn) * 255.0
    elif mx > 0:
        out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def show_img(img, caption=""):
    if img is not None:
        st.image(to_display(img), caption=caption, use_container_width=True, clamp=True)


def show_mask_green(mask, caption=""):
    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[:, :, 1] = m
    st.image(rgb, caption=caption, use_container_width=True, clamp=True)


def metric_color(v):
    return "#00d4aa" if v > 0.7 else "#f59e0b" if v > 0.5 else "#ef4444"


def _reset():
    for k, v in DEFAULTS.items():
        if k != "slice_idx":
            st.session_state[k] = v
    st.session_state["slice_idx"] = 0


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001f4c2 Upload",
    "⚙️ Process",
    "\U0001f4cf Measurements",
    "\U0001f4ca Evaluation",
    "\U0001f4dd Report",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload Brain MRI Data")

    up1, up2 = st.columns(2)
    with up1:
        mri_file = st.file_uploader("Upload MRI Image",
            type=["tif", "tiff", "jpg", "jpeg", "png"], key="mri_up")
    with up2:
        mask_file = st.file_uploader("Ground Truth Mask (optional)",
            type=["tif", "tiff", "jpg", "jpeg", "png"], key="mask_up")

    if mri_file is not None:
        ukey = f"mri_{mri_file.name}_{mri_file.size}"
        mkey = f"mask_{mask_file.name}_{mask_file.size}" if mask_file else "none"
        ckey = f"{ukey}_{mkey}"
        if st.session_state.get("_last_key") != ckey:
            try:
                with st.spinner("Loading..."):
                    pd_ = load_mri_and_mask(mri_file, mask_file)
                st.session_state["patient"] = pd_
                st.session_state["_last_key"] = ckey
                st.session_state["_last_local"] = None
                _reset()
                st.session_state["patient"] = pd_
                st.success(f"Loaded **{pd_['patient_id']}**")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()
    zip_file = st.file_uploader("Or upload ZIP with multiple slices", type=["zip"], key="zip_up")
    if zip_file is not None:
        zkey = f"zip_{zip_file.name}_{zip_file.size}"
        if st.session_state.get("_last_key") != zkey:
            try:
                with st.spinner("Extracting..."):
                    pd_ = handle_upload(zip_file)
                st.session_state["patient"] = pd_
                st.session_state["_last_key"] = zkey
                st.session_state["_last_local"] = None
                _reset()
                st.session_state["patient"] = pd_
                st.success(f"Loaded **{pd_['slice_count']}** slices")
            except Exception as e:
                st.error(f"Failed: {e}")

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    local_patients = get_patient_list(DATA_DIR)
    if local_patients:
        st.divider()
        st.subheader("Or select from local dataset")
        names = [os.path.basename(p) for p in local_patients]
        sel = st.selectbox(f"**{len(names)}** patients available", names,
                           index=None, placeholder="Select a patient...", key="local_sel")
        if sel is not None:
            if st.session_state.get("_last_local") != sel:
                try:
                    fp = local_patients[names.index(sel)]
                    with st.spinner(f"Loading {sel}..."):
                        pd_ = load_patient_data(fp)
                    st.session_state["patient"] = pd_
                    st.session_state["_last_local"] = sel
                    st.session_state["_last_key"] = None
                    _reset()
                    st.session_state["patient"] = pd_
                    st.success(f"Loaded **{pd_['slice_count']}** slices from **{sel}**")
                except Exception as e:
                    st.error(f"Failed: {e}")

    patient = st.session_state["patient"]
    if patient is None:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; background:#111827;
            border:1px solid #1f2937; border-radius:12px; margin-top:20px;">
            <div style="font-size:4rem;">\U0001f9e0</div>
            <div style="font-size:1.1rem; font-weight:600; margin:12px 0 8px;">No MRI data loaded</div>
            <div style="color:#9ca3af; font-size:0.85rem;">Upload an image, ZIP, or select from local dataset.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Patient", patient['patient_id'])
        c2.metric("Slices", patient['slice_count'])
        c3.metric("Size", f"{patient['image_shape'][0]}x{patient['image_shape'][1]}")

        if patient["slice_count"] > 1:
            st.session_state["slice_idx"] = st.slider("Select Slice",
                0, patient["slice_count"] - 1, st.session_state["slice_idx"], key="sl_slider")

        idx = st.session_state["slice_idx"]
        ci, cm = st.columns(2)
        with ci:
            show_img(patient["images"][idx], f"MRI Slice {idx+1}/{patient['slice_count']}")
        with cm:
            if patient["has_masks"] and idx < len(patient["masks"]):
                has_t = idx in patient.get("tumor_slices", [])
                show_mask_green(patient["masks"][idx],
                    f"Ground Truth ({'Tumor' if has_t else 'No Tumor'})")
            else:
                st.info("No ground truth mask")

        if patient["has_masks"]:
            ts = patient.get("tumor_slices", [])
            st.caption(f"Tumor in **{len(ts)}** of **{patient['slice_count']}** slices")

        st.success("Data loaded! Go to **Process** tab and click **Run Processing**.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PROCESS (single slice)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Process Current Slice")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload** tab first.")
    else:
        idx = st.session_state["slice_idx"]
        st.markdown(f"Processing **Slice {idx+1}** of {patient['slice_count']}")

        run = st.button("▶ Run Processing", key="btn_run", use_container_width=True)

        if run:
            try:
                with st.spinner("Processing..."):
                    img = patient["images"][idx]

                    # 1. Preprocessing
                    pp_config = {
                        "normalize": True, "filter_type": "gaussian",
                        "sigma": 1.0, "kernel_size": 3,
                        "clahe": True, "clip_limit": 2.0,
                    }
                    pp_result, pp_steps = preprocess_image(img, pp_config)
                    st.session_state["pp_result"] = pp_result
                    st.session_state["pp_steps"] = pp_steps

                    # 2. Segmentation
                    seg_result = segment_tumor_candidate(pp_result, method="otsu", params={})
                    st.session_state["seg_result"] = seg_result

                    # 3. Morphology
                    morph_config = {"operation": "opening_then_closing", "kernel_size": 5, "iterations": 1}
                    morph_result = apply_morphological_pipeline(seg_result["binary_mask"], morph_config)
                    st.session_state["morph_result"] = morph_result

                    # 4. Post-processing
                    filtered = filter_components_by_size(morph_result["output_mask"], min_area=100)
                    tumor = select_tumor_region(filtered, strategy="largest", min_area=50)
                    overlay = create_tumor_overlay(img, tumor)
                    st.session_state["tumor_mask"] = tumor
                    st.session_state["overlay"] = overlay

                    # 5. Measurements
                    area = calculate_tumor_area(tumor, pixel_spacing_mm=1.0)
                    st.session_state["meas_stats"] = area

                    # 6. Evaluation
                    if patient["has_masks"] and idx < len(patient["masks"]):
                        gt = patient["masks"][idx].copy()
                        if gt.max() <= 1:
                            gt = (gt * 255).astype(np.uint8)
                        ev = evaluate_segmentation(tumor, gt)
                        st.session_state["eval_single"] = ev
                        fig = plot_evaluation_charts(ev)
                        st.session_state["eval_chart"] = fig

                    st.session_state["processed"] = True
                    st.session_state["report_text"] = None

                st.success("Processing complete! Check **Measurements**, **Evaluation**, and **Report** tabs.")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Show pipeline results
        if st.session_state["processed"] and st.session_state["pp_steps"] is not None:
            steps = st.session_state["pp_steps"]

            # Preprocessing
            st.divider()
            st.markdown("#### 1. Preprocessing Pipeline")
            cols = st.columns(4)
            with cols[0]: show_img(steps["original"], "Original")
            with cols[1]: show_img(steps["normalized"], "Normalized")
            with cols[2]: show_img(steps["filtered"], "Gaussian Filtered")
            with cols[3]: show_img(steps["enhanced"], "CLAHE Enhanced")

            # Segmentation
            st.divider()
            st.markdown("#### 2. Segmentation (Otsu Thresholding)")
            seg = st.session_state["seg_result"]
            sg1, sg2 = st.columns(2)
            with sg1:
                show_img(seg["binary_mask"], "Otsu Threshold Result")
                if seg.get("threshold_value") is not None:
                    st.caption(f"Threshold value: {seg['threshold_value']}")
            with sg2:
                show_img(seg["brain_mask"], "Brain Mask (skull stripped)")

            # Morphology
            st.divider()
            st.markdown("#### 3. Morphological Operations")
            morph_r = st.session_state["morph_result"]
            mo1, mo2 = st.columns(2)
            with mo1: show_img(morph_r["input_mask"], "Before Morphology")
            with mo2: show_img(morph_r["output_mask"], "After Opening → Closing")
            st.caption(f"Noise removed: **{morph_r['noise_removed_pixels']:,}** pixels | "
                       f"Kernel: {morph_r['config_used']['kernel_size']}x{morph_r['config_used']['kernel_size']} ellipse")

            # Final Result
            st.divider()
            st.markdown("#### 4. Final Result")
            fr1, fr2, fr3 = st.columns(3)
            with fr1: show_img(patient["images"][idx], "Original MRI")
            with fr2: show_img(st.session_state["tumor_mask"], "Tumor Mask")
            with fr3: show_img(st.session_state["overlay"], "Overlay")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MEASUREMENTS (single slice area only)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Tumor Measurements")
    patient = st.session_state["patient"]

    if not st.session_state["processed"]:
        st.warning("Click **Run Processing** in the **Process** tab first.")
    else:
        idx = st.session_state["slice_idx"]

        st.markdown(f"#### Slice {idx+1} Tumor Area")
        area = st.session_state["meas_stats"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Tumor Pixels", f"{area['tumor_pixels']:,}")
        m2.metric("Tumor Area", f"{area['tumor_area_mm2']:.1f} mm²")
        m3.metric("Tumor %", f"{area['tumor_percentage']:.4f}%")

        st.divider()
        cols = st.columns(3)
        with cols[0]: show_img(patient["images"][idx], "Original MRI")
        with cols[1]: show_img(st.session_state["tumor_mask"], "Detected Tumor")
        with cols[2]: show_img(st.session_state["overlay"], "Overlay")

        st.divider()
        st.markdown("#### Measurement Details")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Image Pixels | {area['total_pixels']:,} |
        | Tumor Pixels | {area['tumor_pixels']:,} |
        | Tumor Area | {area['tumor_area_mm2']:.2f} mm² |
        | Tumor Area | {area['tumor_area_cm2']:.4f} cm² |
        | Tumor Percentage | {area['tumor_percentage']:.4f}% |
        | Pixel Spacing | 1.0 mm (assumed) |
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION (basic metrics)
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Segmentation Evaluation")
    patient = st.session_state["patient"]

    if not st.session_state["processed"]:
        st.warning("Click **Run Processing** in the **Process** tab first.")
    elif patient is not None:
        idx = st.session_state["slice_idx"]

        if patient["has_masks"] and st.session_state["eval_single"] is not None:
            ev = st.session_state["eval_single"]

            st.markdown(f"#### Slice {idx+1} Metrics")
            e1, e2, e3, e4, e5 = st.columns(5)
            for col, name, key in [
                (e1, "Dice", "dice"), (e2, "IoU", "iou"),
                (e3, "Precision", "precision"), (e4, "Recall", "recall"),
                (e5, "Specificity", "specificity"),
            ]:
                val = ev[key]
                with col:
                    st.metric(name, f"{val:.4f}")
                    color = metric_color(val)
                    st.markdown(f'<div style="height:3px;background:{color};border-radius:2px;margin-top:-8px;"></div>',
                                unsafe_allow_html=True)

            st.markdown(f"**{ev['interpretation']}**")

            # Confusion matrix
            st.divider()
            st.markdown("#### Confusion Matrix")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("True Positive", f"{ev['TP']:,}")
            cm2.metric("False Positive", f"{ev['FP']:,}")
            cm3.metric("True Negative", f"{ev['TN']:,}")
            cm4.metric("False Negative", f"{ev['FN']:,}")

            # Charts
            if st.session_state["eval_chart"] is not None:
                st.divider()
                st.markdown("#### Evaluation Charts")
                st.pyplot(st.session_state["eval_chart"])

            # Visual comparison
            st.divider()
            st.markdown("#### Visual Comparison")
            vc1, vc2, vc3 = st.columns(3)
            with vc1: show_img(patient["images"][idx], "Original MRI")
            with vc2: show_img(st.session_state["tumor_mask"], "Predicted Mask")
            with vc3:
                if idx < len(patient["masks"]):
                    show_mask_green(patient["masks"][idx], "Ground Truth Mask")

        else:
            st.info("Upload a ground truth mask for evaluation metrics (Dice, IoU, Precision, Recall, Specificity).")

            st.divider()
            st.markdown("#### Segmentation Result")
            area = st.session_state.get("meas_stats")
            if area:
                s1, s2 = st.columns(2)
                s1.metric("Tumor Pixels Detected", f"{area['tumor_pixels']:,}")
                s2.metric("Tumor Coverage", f"{area['tumor_percentage']:.4f}%")

            vr1, vr2 = st.columns(2)
            with vr1: show_img(patient["images"][idx], "Original MRI")
            with vr2:
                if st.session_state["overlay"] is not None:
                    show_img(st.session_state["overlay"], "Segmentation Overlay")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Analysis Report")
    patient = st.session_state["patient"]

    if not st.session_state["processed"]:
        st.warning("Click **Run Processing** in the **Process** tab first.")
    else:
        if st.session_state["report_text"] is None:
            idx = st.session_state["slice_idx"]
            L = []
            L.append("=" * 65)
            L.append("  BRAINSCAN AI — ANALYSIS REPORT (Phase 1)")
            L.append("=" * 65)
            L.append(f"  Patient ID      : {patient['patient_id']}")
            L.append(f"  Generated       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            L.append(f"  Total Slices    : {patient['slice_count']}")
            L.append(f"  Analyzed Slice  : {idx + 1}")
            L.append(f"  Image Size      : {patient['image_shape'][0]} x {patient['image_shape'][1]}")
            L.append(f"  Ground Truth    : {'Available' if patient['has_masks'] else 'Not available'}")
            L.append("")

            L.append("-" * 65)
            L.append("  PROCESSING PIPELINE")
            L.append("-" * 65)
            L.append("  Preprocessing   : Normalize + Gaussian (s=1.0) + CLAHE (c=2.0)")
            L.append("  Segmentation    : Otsu thresholding + skull stripping")
            seg = st.session_state.get("seg_result")
            if seg:
                L.append(f"  Threshold       : {seg.get('threshold_value', 'N/A')}")
            L.append("  Morphology      : Opening then closing (5x5 ellipse, 1 iter)")
            L.append("  Post-processing : Component filtering (>=100px) + largest region")
            L.append("")

            L.append("-" * 65)
            L.append("  MEASUREMENTS")
            L.append("-" * 65)
            area = st.session_state.get("meas_stats")
            if area:
                L.append(f"  Tumor Pixels    : {area['tumor_pixels']:,}")
                L.append(f"  Tumor Area      : {area['tumor_area_mm2']:.1f} mm2 ({area['tumor_area_cm2']:.4f} cm2)")
                L.append(f"  Tumor %         : {area['tumor_percentage']:.4f}%")
            L.append("")

            L.append("-" * 65)
            L.append("  EVALUATION")
            L.append("-" * 65)
            ev = st.session_state.get("eval_single")
            if ev:
                L.append(f"  Dice            : {ev['dice']:.4f}")
                L.append(f"  IoU             : {ev['iou']:.4f}")
                L.append(f"  Precision       : {ev['precision']:.4f}")
                L.append(f"  Recall          : {ev['recall']:.4f}")
                L.append(f"  Specificity     : {ev['specificity']:.4f}")
                L.append(f"  TP={ev['TP']:,}  FP={ev['FP']:,}  TN={ev['TN']:,}  FN={ev['FN']:,}")
                L.append(f"  {ev['interpretation']}")
            else:
                L.append("  (no ground truth mask provided)")
            L.append("")

            L.append("-" * 65)
            L.append("  METHOD")
            L.append("-" * 65)
            L.append("  Classical image processing (no ML/DL)")
            L.append("  Otsu thresholding for segmentation")
            L.append("  Morphological operations for noise removal")
            L.append("  Connected component analysis for region selection")
            L.append("")
            L.append("=" * 65)
            L.append("  Generated by BrainScan AI — Phase 1")
            L.append("=" * 65)

            st.session_state["report_text"] = "\n".join(L)

        if st.session_state["report_text"]:
            st.code(st.session_state["report_text"], language=None)

            st.download_button("\U0001f4c4 Download Report",
                st.session_state["report_text"],
                f"brainscan_report_{patient['patient_id']}.txt",
                "text/plain", use_container_width=True)
