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
from src.measurements import calculate_tumor_area, estimate_tumor_volume, generate_slice_area_dataframe
from src.evaluation import (
    dice_coefficient, iou_score, evaluate_segmentation,
    evaluate_volume_set, plot_evaluation_charts,
)

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
    "all_tumor_masks": None, "all_overlays": None,
    "meas_stats": None, "volume_stats": None, "slice_df": None,
    "eval_single": None, "eval_volume": None, "eval_chart": None,
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

        st.success("Data loaded! Go to **Process** tab and click **Run All**.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PROCESS (one button does everything)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Process All Slices")
    patient = st.session_state["patient"]

    if patient is None:
        st.warning("Upload MRI data in the **Upload** tab first.")
    else:
        st.markdown(f"**{patient['slice_count']}** slices will be processed automatically.")

        run = st.button("▶ Run All Processing", key="btn_run", use_container_width=True)

        if run:
            n = patient["slice_count"]
            progress = st.progress(0, text="Starting...")
            all_masks = []
            all_overlays = []

            try:
                for i in range(n):
                    pct = int((i / n) * 100)
                    progress.progress(pct, text=f"Processing slice {i+1}/{n}...")

                    img = patient["images"][i]

                    pp_config = {
                        "normalize": True, "filter_type": "gaussian",
                        "sigma": 1.0, "kernel_size": 3,
                        "clahe": True, "clip_limit": 2.0,
                    }
                    pp_result, pp_steps = preprocess_image(img, pp_config)

                    seg_result = segment_tumor_candidate(pp_result, method="otsu", params={})

                    morph_config = {"operation": "opening_then_closing", "kernel_size": 5, "iterations": 1}
                    morph_result = apply_morphological_pipeline(seg_result["binary_mask"], morph_config)
                    filtered = filter_components_by_size(morph_result["output_mask"], min_area=100)
                    tumor = select_tumor_region(filtered, strategy="largest", min_area=50)
                    overlay = create_tumor_overlay(img, tumor)

                    all_masks.append(tumor)
                    all_overlays.append(overlay)

                    if i == st.session_state["slice_idx"]:
                        st.session_state["pp_result"] = pp_result
                        st.session_state["pp_steps"] = pp_steps
                        st.session_state["seg_result"] = seg_result
                        st.session_state["morph_result"] = morph_result
                        st.session_state["tumor_mask"] = tumor
                        st.session_state["overlay"] = overlay

                st.session_state["all_tumor_masks"] = all_masks
                st.session_state["all_overlays"] = all_overlays

                # Measurements
                progress.progress(85, text="Calculating measurements...")
                idx = st.session_state["slice_idx"]
                area = calculate_tumor_area(all_masks[idx], pixel_spacing_mm=1.0)
                st.session_state["meas_stats"] = area

                vol = estimate_tumor_volume(all_masks, pixel_spacing_mm=1.0, slice_thickness_mm=5.0)
                st.session_state["volume_stats"] = vol

                sdf = generate_slice_area_dataframe(all_masks, pixel_spacing_mm=1.0)
                st.session_state["slice_df"] = sdf

                # Evaluation
                if patient["has_masks"]:
                    progress.progress(92, text="Evaluating...")
                    gt_masks_scaled = []
                    for m in patient["masks"]:
                        gm = m.copy()
                        if gm.max() <= 1:
                            gm = (gm * 255).astype(np.uint8)
                        gt_masks_scaled.append(gm)

                    ev_single = evaluate_segmentation(all_masks[idx], gt_masks_scaled[idx])
                    st.session_state["eval_single"] = ev_single

                    ev_vol = evaluate_volume_set(all_masks, gt_masks_scaled)
                    st.session_state["eval_volume"] = ev_vol

                    fig = plot_evaluation_charts(ev_single)
                    st.session_state["eval_chart"] = fig

                st.session_state["processed"] = True
                progress.progress(100, text="Done!")
                st.success(f"All **{n}** slices processed! Check **Measurements**, **Evaluation**, and **Report** tabs.")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Show pipeline for current slice if processed
        if st.session_state["processed"] and st.session_state["pp_steps"] is not None:
            st.divider()
            st.markdown("#### Preprocessing Pipeline (current slice)")
            steps = st.session_state["pp_steps"]
            cols = st.columns(4)
            with cols[0]: show_img(steps["original"], "1. Original")
            with cols[1]: show_img(steps["normalized"], "2. Normalized")
            with cols[2]: show_img(steps["filtered"], "3. Filtered")
            with cols[3]: show_img(steps["enhanced"], "4. Enhanced")

        if st.session_state["processed"] and st.session_state["tumor_mask"] is not None:
            st.divider()
            st.markdown("#### Segmentation Result (current slice)")
            cols = st.columns(3)
            idx = st.session_state["slice_idx"]
            with cols[0]: show_img(st.session_state["seg_result"]["binary_mask"], "Otsu Threshold")
            with cols[1]: show_img(st.session_state["tumor_mask"], "Tumor Mask")
            with cols[2]: show_img(st.session_state["overlay"], "Overlay")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Tumor Measurements")
    patient = st.session_state["patient"]

    if not st.session_state["processed"]:
        st.warning("Click **Run All Processing** in the **Process** tab first.")
    else:
        idx = st.session_state["slice_idx"]

        # Single slice area
        st.markdown("#### Current Slice Area")
        area = st.session_state["meas_stats"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Tumor Pixels", f"{area['tumor_pixels']:,}")
        m2.metric("Tumor Area", f"{area['tumor_area_mm2']:.1f} mm²")
        m3.metric("Tumor %", f"{area['tumor_percentage']:.4f}%")

        cols = st.columns(3)
        with cols[0]: show_img(patient["images"][idx], "Original MRI")
        with cols[1]: show_img(st.session_state["tumor_mask"], "Detected Tumor")
        with cols[2]: show_img(st.session_state["overlay"], "Overlay")

        # Volume estimation
        st.divider()
        st.markdown("#### Volume Estimation (All Slices)")
        vol = st.session_state["volume_stats"]
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Volume", f"{vol['tumor_volume_cm3']:.2f} cm³")
        v2.metric("Total Voxels", f"{vol['total_tumor_voxels']:,}")
        v3.metric("Affected Slices", f"{vol['affected_slices']}/{vol['total_slices']}")
        v4.metric("Volume (mm³)", f"{vol['tumor_volume_mm3']:.0f}")

        st.caption(vol["note"])

        # Per-slice table
        st.divider()
        st.markdown("#### Per-Slice Tumor Data")
        sdf = st.session_state["slice_df"]
        st.dataframe(sdf, use_container_width=True, hide_index=True)

        # Area chart
        fig_area, ax_area = plt.subplots(figsize=(10, 3))
        ax_area.bar(sdf["Slice"], sdf["Area mm2"], color="#00d4aa", alpha=0.8)
        ax_area.set_xlabel("Slice Index")
        ax_area.set_ylabel("Tumor Area (mm²)")
        ax_area.set_title("Tumor Area per Slice")
        ax_area.set_facecolor("#0a0e1a")
        fig_area.patch.set_facecolor("#0a0e1a")
        ax_area.tick_params(colors="#9ca3af")
        ax_area.xaxis.label.set_color("#9ca3af")
        ax_area.yaxis.label.set_color("#9ca3af")
        ax_area.title.set_color("#f9fafb")
        for spine in ax_area.spines.values():
            spine.set_color("#1f2937")
        plt.tight_layout()
        st.pyplot(fig_area)

        # Browse all slices
        st.divider()
        st.markdown("#### Browse All Slice Results")
        if patient["slice_count"] > 1:
            browse_idx = st.slider("Slice", 0, patient["slice_count"] - 1, idx, key="browse_sl")
        else:
            browse_idx = 0
        bc1, bc2, bc3 = st.columns(3)
        with bc1: show_img(patient["images"][browse_idx], f"MRI Slice {browse_idx+1}")
        with bc2:
            if st.session_state["all_tumor_masks"]:
                show_img(st.session_state["all_tumor_masks"][browse_idx], "Tumor Mask")
        with bc3:
            if st.session_state["all_overlays"]:
                show_img(st.session_state["all_overlays"][browse_idx], "Overlay")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Segmentation Evaluation")
    patient = st.session_state["patient"]

    if not st.session_state["processed"]:
        st.warning("Click **Run All Processing** in the **Process** tab first.")
    elif patient is not None:
        idx = st.session_state["slice_idx"]
        has_gt = patient["has_masks"]

        if has_gt and st.session_state["eval_single"] is not None:
            ev = st.session_state["eval_single"]

            st.markdown(f"#### Current Slice Metrics (Slice {idx+1})")
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

            st.divider()
            st.markdown("#### Confusion Matrix")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("True Positive", f"{ev['TP']:,}")
            cm2.metric("False Positive", f"{ev['FP']:,}")
            cm3.metric("True Negative", f"{ev['TN']:,}")
            cm4.metric("False Negative", f"{ev['FN']:,}")

            if st.session_state["eval_chart"] is not None:
                st.divider()
                st.markdown("#### Evaluation Charts")
                st.pyplot(st.session_state["eval_chart"])

            st.divider()
            st.markdown("#### Visual Comparison")
            vc1, vc2, vc3 = st.columns(3)
            with vc1: show_img(patient["images"][idx], "Original MRI")
            with vc2: show_img(st.session_state["tumor_mask"], "Predicted Mask")
            with vc3:
                if idx < len(patient["masks"]):
                    show_mask_green(patient["masks"][idx], "Ground Truth Mask")

            if st.session_state["eval_volume"] is not None:
                st.divider()
                st.markdown("#### Volume-Level Evaluation (All Slices)")
                ev_v = st.session_state["eval_volume"]
                vv1, vv2, vv3, vv4 = st.columns(4)
                vv1.metric("Mean Dice", f"{ev_v['mean_dice']:.4f}")
                vv2.metric("Std Dice", f"{ev_v['std_dice']:.4f}")
                vv3.metric("Mean IoU", f"{ev_v['mean_iou']:.4f}")
                vv4.metric("Overall Dice", f"{ev_v['overall_dice']:.4f}")

                fig_dice, ax_dice = plt.subplots(figsize=(10, 3))
                slices_range = list(range(len(ev_v["per_slice_dice"])))
                ax_dice.bar(slices_range, ev_v["per_slice_dice"], color="#3b82f6", alpha=0.8)
                ax_dice.axhline(y=ev_v["mean_dice"], color="#00d4aa", linestyle="--", label=f"Mean={ev_v['mean_dice']:.3f}")
                ax_dice.set_xlabel("Slice Index")
                ax_dice.set_ylabel("Dice Score")
                ax_dice.set_title("Per-Slice Dice Coefficient")
                ax_dice.set_ylim(0, 1.05)
                ax_dice.legend()
                ax_dice.set_facecolor("#0a0e1a")
                fig_dice.patch.set_facecolor("#0a0e1a")
                ax_dice.tick_params(colors="#9ca3af")
                ax_dice.xaxis.label.set_color("#9ca3af")
                ax_dice.yaxis.label.set_color("#9ca3af")
                ax_dice.title.set_color("#f9fafb")
                ax_dice.legend(facecolor="#111827", edgecolor="#1f2937", labelcolor="#f9fafb")
                for spine in ax_dice.spines.values():
                    spine.set_color("#1f2937")
                plt.tight_layout()
                st.pyplot(fig_dice)

        else:
            st.markdown("#### Segmentation Results (No Ground Truth)")
            st.info("Upload a ground truth mask for full evaluation with Dice, IoU, Precision, Recall, and Specificity metrics.")

            st.divider()
            st.markdown("#### Detected Tumor Summary")
            all_masks = st.session_state["all_tumor_masks"]
            if all_masks:
                total_px = sum(int(np.count_nonzero(m)) for m in all_masks)
                affected = sum(1 for m in all_masks if np.count_nonzero(m) > 0)
                s1, s2, s3 = st.columns(3)
                s1.metric("Total Tumor Pixels", f"{total_px:,}")
                s2.metric("Affected Slices", f"{affected}/{len(all_masks)}")
                s3.metric("Detection Rate", f"{affected/len(all_masks)*100:.1f}%")

            st.divider()
            st.markdown("#### Visual Result")
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
        st.warning("Click **Run All Processing** in the **Process** tab first.")
    else:
        if st.session_state["report_text"] is None:
            L = []
            L.append("=" * 65)
            L.append("  BRAINSCAN AI — ANALYSIS REPORT")
            L.append("=" * 65)
            L.append(f"  Patient ID      : {patient['patient_id']}")
            L.append(f"  Generated       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            L.append(f"  Total Slices    : {patient['slice_count']}")
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
            L.append("  Morphology      : Opening then closing (5x5, 1 iter)")
            L.append("  Post-processing : Component filtering + largest region")
            L.append("")

            L.append("-" * 65)
            L.append("  MEASUREMENTS")
            L.append("-" * 65)
            area = st.session_state.get("meas_stats")
            if area:
                L.append(f"  Slice {st.session_state['slice_idx']+1} Area : {area['tumor_area_mm2']:.1f} mm2 "
                         f"({area['tumor_pixels']:,} px, {area['tumor_percentage']:.4f}%)")
            vol = st.session_state.get("volume_stats")
            if vol:
                L.append(f"  Total Volume    : {vol['tumor_volume_cm3']:.2f} cm3 ({vol['tumor_volume_mm3']:.0f} mm3)")
                L.append(f"  Total Voxels    : {vol['total_tumor_voxels']:,}")
                L.append(f"  Affected Slices : {vol['affected_slices']}/{vol['total_slices']}")
                L.append(f"  Spacing         : {vol['pixel_spacing_mm']} mm px, {vol['slice_thickness_mm']} mm slice")
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
            ev_v = st.session_state.get("eval_volume")
            if ev_v:
                L.append(f"  Mean Dice (vol) : {ev_v['mean_dice']:.4f} +/- {ev_v['std_dice']:.4f}")
                L.append(f"  Mean IoU (vol)  : {ev_v['mean_iou']:.4f}")
                L.append(f"  Overall Dice    : {ev_v['overall_dice']:.4f}")
            if not ev and not ev_v:
                L.append("  (no ground truth masks)")
            L.append("")

            L.append("-" * 65)
            L.append("  METHOD")
            L.append("-" * 65)
            L.append("  Classical image processing (no ML/DL)")
            L.append("  Otsu thresholding for segmentation")
            L.append("  Morphological operations for refinement")
            L.append("  Voxel-count volume estimation (pseudo-3D)")
            L.append("")
            L.append("=" * 65)
            L.append("  Generated by BrainScan AI")
            L.append("=" * 65)

            st.session_state["report_text"] = "\n".join(L)
            st.success("Report generated!")

        if st.session_state["report_text"]:
            st.code(st.session_state["report_text"], language=None)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("\U0001f4c4 Download TXT Report",
                    st.session_state["report_text"],
                    f"brainscan_report_{patient['patient_id']}.txt",
                    "text/plain", use_container_width=True)
            with c2:
                csv_data = st.session_state["slice_df"].to_csv(index=False) if st.session_state["slice_df"] is not None else ""
                if csv_data:
                    st.download_button("\U0001f4ca Download Slice Data CSV",
                        csv_data,
                        f"brainscan_slices_{patient['patient_id']}.csv",
                        "text/csv", use_container_width=True)
