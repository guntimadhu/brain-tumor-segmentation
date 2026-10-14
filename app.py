import datetime
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
from src.io_utils import handle_upload, load_mri_and_mask, get_patient_list, load_patient_data
from src.preprocessing import compare_filters, apply_clahe, normalize_intensity
from src.segmentation import segment_tumor_candidate
from src.morphology import get_morphology_comparison
from src.postprocessing import create_tumor_overlay, find_connected_components
from src.measurements import foreground_pixels, summarize_foreground, slice_dataframe
from src.evaluation import slice_metrics, patient_metrics
from src.pipeline import (run_pipeline, METHOD_LABELS, METHOD_PARAMS, DEFAULT_METHOD,
                          PIPELINE_VERSION, MORPHOLOGY, MIN_COMPONENT_AREA, MAX_COMPONENT_FRACTION)

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
    "filter_comparison": None, "clahe_before": None, "clahe_after": None,
    "seg_result": None, "seg_compare": None,
    "skull_mask": None,
    "morph_result": None, "morph_comparison": None,
    "cc_info": None, "filtered_mask": None,
    "tumor_mask": None, "overlay": None,
    "all_tumor_masks": None, "all_overlays": None,
    "meas_stats": None, "count_stats": None, "slice_df": None,
    "eval_single": None, "eval_patient": None,
    "method_used": DEFAULT_METHOD,
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
        <div class="header-subtitle">Classical FLAIR Abnormality Segmentation &amp; Mask Measurement</div>
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
            show_img(patient["images"][idx], f"FLAIR {patient['image_files'][idx]} ({idx+1}/{patient['slice_count']})")
        with cm:
            if patient["has_masks"] and idx < len(patient["masks"]) and patient["masks"][idx] is not None:
                has_t = idx in patient.get("tumor_slices", [])
                show_mask_green(patient["masks"][idx],
                    f"Reference mask ({'annotated abnormality' if has_t else 'empty'})")
            else:
                st.info("No reference mask for this slice")

        if patient["has_masks"]:
            ts = patient.get("tumor_slices", [])
            st.caption(f"Annotated abnormality in **{len(ts)}** of **{patient['slice_count']}** slices")

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
        st.markdown(f"**{patient['slice_count']}** slices will be processed.")
        method = st.selectbox("Segmentation method", list(METHOD_LABELS),
                              index=list(METHOD_LABELS).index(DEFAULT_METHOD),
                              format_func=lambda m: METHOD_LABELS[m], key="method_sel")
        st.caption(f"Pipeline version {PIPELINE_VERSION}. Parameters: {METHOD_PARAMS[method] or 'automatic threshold'}; "
                   f"opening then closing, {MORPHOLOGY['kernel_size']}x{MORPHOLOGY['kernel_size']} ellipse; "
                   f"components {MIN_COMPONENT_AREA} px to {int(MAX_COMPONENT_FRACTION*100)}% of image; largest kept.")

        run = st.button("▶ Run All Processing", key="btn_run", use_container_width=True)

        if run:
            n = patient["slice_count"]
            progress = st.progress(0, text="Starting...")
            all_masks = []
            all_overlays = []
            st.session_state["method_used"] = method
            st.session_state["report_text"] = None

            try:
                for i in range(n):
                    progress.progress(int((i / n) * 100), text=f"Processing slice {i+1}/{n}...")

                    img = patient["images"][i]
                    res = run_pipeline(img, method=method)
                    tumor = res["final"]
                    overlay = create_tumor_overlay(img, tumor)

                    all_masks.append(tumor)
                    all_overlays.append(overlay)

                    if i == st.session_state["slice_idx"]:
                        st.session_state["pp_result"] = res["enhanced"]
                        st.session_state["pp_steps"] = res["pp_steps"]
                        st.session_state["seg_result"] = res["seg"]
                        st.session_state["morph_result"] = res["morph"]
                        st.session_state["tumor_mask"] = tumor
                        st.session_state["overlay"] = overlay

                        st.session_state["filter_comparison"] = compare_filters(normalize_intensity(img))
                        normalized = normalize_intensity(img)
                        st.session_state["clahe_before"] = normalized
                        st.session_state["clahe_after"] = apply_clahe(normalized, clip_limit=2.0)

                        st.session_state["skull_mask"] = res["brain_mask"]
                        st.session_state["seg_compare"] = {
                            m: segment_tumor_candidate(res["enhanced"], method=m, params=METHOD_PARAMS[m],
                                                       brain_mask=res["brain_mask"])
                            for m in METHOD_LABELS}

                        st.session_state["morph_comparison"] = get_morphology_comparison(res["seg"]["binary_mask"])
                        st.session_state["cc_info"] = find_connected_components(res["morph"]["output_mask"])
                        st.session_state["filtered_mask"] = res["filtered"]

                st.session_state["all_tumor_masks"] = all_masks
                st.session_state["all_overlays"] = all_overlays

                progress.progress(85, text="Counting foreground pixels...")
                idx = st.session_state["slice_idx"]
                refs = patient["masks"] if patient["has_masks"] else None
                st.session_state["meas_stats"] = foreground_pixels(all_masks[idx])
                st.session_state["count_stats"] = summarize_foreground(all_masks, refs)
                st.session_state["slice_df"] = slice_dataframe(all_masks, patient["image_files"], refs)

                if patient["has_masks"]:
                    progress.progress(92, text="Evaluating against reference masks...")
                    ref = patient["masks"][idx]
                    st.session_state["eval_single"] = (slice_metrics(all_masks[idx] > 0, ref > 127)
                                                       if ref is not None else None)
                    st.session_state["eval_patient"] = patient_metrics(all_masks, patient["masks"])

                st.session_state["processed"] = True
                progress.progress(100, text="Done!")
                st.success(f"All **{n}** slices processed! Check **Measurements**, **Evaluation**, and **Report** tabs.")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Show detailed pipeline for current slice if processed
        if st.session_state["processed"] and st.session_state["pp_steps"] is not None:
            idx = st.session_state["slice_idx"]
            steps = st.session_state["pp_steps"]

            # --- PREPROCESSING ---
            st.divider()
            st.markdown("#### 1. Preprocessing Pipeline")
            cols = st.columns(4)
            with cols[0]: show_img(steps["original"], "Original")
            with cols[1]: show_img(steps["normalized"], "Normalized")
            with cols[2]: show_img(steps["filtered"], "Gaussian Filtered")
            with cols[3]: show_img(steps["enhanced"], "CLAHE Enhanced")

            # Filter Comparison
            if st.session_state["filter_comparison"] is not None:
                st.markdown("##### Filter Comparison")
                fc = st.session_state["filter_comparison"]
                fc1, fc2, fc3 = st.columns(3)
                with fc1: show_img(fc["none"], "No Filter")
                with fc2: show_img(fc["gaussian"], "Gaussian (σ=1.0)")
                with fc3: show_img(fc["median"], "Median (k=3)")

            # CLAHE Before/After
            if st.session_state["clahe_before"] is not None:
                st.markdown("##### CLAHE Contrast Enhancement")
                cl1, cl2 = st.columns(2)
                with cl1: show_img(st.session_state["clahe_before"], "Before CLAHE")
                with cl2: show_img(st.session_state["clahe_after"], "After CLAHE (clip=2.0)")

            # --- SKULL STRIPPING ---
            st.divider()
            st.markdown("#### 2. Brain Mask (computed on the normalized image, before CLAHE)")
            if st.session_state["skull_mask"] is not None:
                sk1, sk2, sk3 = st.columns(3)
                with sk1: show_img(steps["normalized"], "Normalized FLAIR")
                with sk2: show_img(st.session_state["skull_mask"], "Brain mask")
                with sk3:
                    import cv2 as _cv2
                    brain_only = _cv2.bitwise_and(
                        steps["enhanced"].astype(np.uint8),
                        steps["enhanced"].astype(np.uint8),
                        mask=st.session_state["skull_mask"],
                    )
                    show_img(brain_only, "CLAHE image inside brain mask")

            # --- SEGMENTATION ---
            st.divider()
            st.markdown("#### 3. Threshold Methods on This Slice (before morphology)")
            seg_cmp = st.session_state.get("seg_compare") or {}
            for col, m in zip(st.columns(len(seg_cmp) or 1), seg_cmp):
                with col:
                    used = " (selected)" if m == st.session_state["method_used"] else ""
                    show_img(seg_cmp[m]["binary_mask"], METHOD_LABELS[m] + used)
                    if seg_cmp[m].get("threshold_value") is not None:
                        st.caption(f"Threshold: {float(seg_cmp[m]['threshold_value']):.0f}")

            # --- MORPHOLOGY ---
            st.divider()
            st.markdown("#### 4. Morphological Operations")
            morph_comp = st.session_state.get("morph_comparison")
            if morph_comp is not None:
                mr1, mr2, mr3 = st.columns(3)
                with mr1: show_img(morph_comp["original"], "Original Mask")
                with mr2: show_img(morph_comp["erosion"], "Erosion")
                with mr3: show_img(morph_comp["dilation"], "Dilation")

                mr4, mr5, mr6 = st.columns(3)
                with mr4: show_img(morph_comp["opening"], "Opening")
                with mr5: show_img(morph_comp["closing"], "Closing")
                with mr6: show_img(morph_comp["opening_then_closing"], "Opening → Closing (used)")

            morph_r = st.session_state.get("morph_result")
            if morph_r is not None:
                st.caption(f"Noise removed: **{morph_r['noise_removed_pixels']:,}** pixels | "
                           f"Kernel: {morph_r['config_used']['kernel_size']}x{morph_r['config_used']['kernel_size']} ellipse | "
                           f"Iterations: {morph_r['config_used']['iterations']}")

            # --- POST-PROCESSING ---
            st.divider()
            st.markdown("#### 5. Post-Processing & Region Selection")
            cc_info = st.session_state.get("cc_info")
            if cc_info is not None:
                st.markdown(f"Connected components found: **{cc_info['num_components']}**")
                if cc_info["component_sizes"]:
                    sizes = sorted(cc_info["component_sizes"], reverse=True)[:5]
                    st.caption(f"Top component sizes (px): {', '.join(str(s) for s in sizes)}")

            pp1, pp2, pp3 = st.columns(3)
            with pp1:
                if morph_r is not None:
                    show_img(morph_r["output_mask"], "After Morphology")
            with pp2:
                if st.session_state["filtered_mask"] is not None:
                    show_img(st.session_state["filtered_mask"], "After Size Filter (≥100px)")
            with pp3:
                show_img(st.session_state["tumor_mask"], "Final mask (largest component)")

            # --- FINAL RESULT ---
            st.divider()
            st.markdown("#### 6. Final Result")
            fr1, fr2, fr3 = st.columns(3)
            with fr1: show_img(patient["images"][idx], "FLAIR slice")
            with fr2: show_img(st.session_state["tumor_mask"], "Predicted mask")
            with fr3: show_img(st.session_state["overlay"], "Overlay")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Mask Measurements")
    patient = st.session_state["patient"]

    if not st.session_state["processed"]:
        st.warning("Click **Run All Processing** in the **Process** tab first.")
    else:
        idx = st.session_state["slice_idx"]

        st.info("Measurements are foreground pixel counts. Verified spatial calibration is not "
                "available for these TIFF images, so no physical areas or volumes are reported.")

        st.markdown("#### Current Slice")
        area = st.session_state["meas_stats"]
        m1, m2 = st.columns(2)
        m1.metric("Predicted foreground pixels", f"{area['foreground_pixels']:,}")
        m2.metric("Share of image", f"{area['foreground_percentage']:.2f}%")

        cols = st.columns(3)
        with cols[0]: show_img(patient["images"][idx], "FLAIR slice")
        with cols[1]: show_img(st.session_state["tumor_mask"], "Predicted mask")
        with cols[2]: show_img(st.session_state["overlay"], "Overlay")

        st.divider()
        st.markdown("#### All Slices")
        cs = st.session_state["count_stats"]
        v1, v2, v3 = st.columns(3)
        v1.metric("Predicted foreground (total px)", f"{cs['pred_total_px']:,}")
        v2.metric("Slices with a prediction", f"{cs['pred_affected_slices']}/{cs['total_slices']}")
        if "ref_total_px" in cs:
            v3.metric("Reference foreground (total px)", f"{cs['ref_total_px']:,}")
            if cs["signed_count_error_pct"] is not None:
                st.caption(f"Signed count error 100×(P−G)/G: **{cs['signed_count_error_pct']:+.1f}%** "
                           f"(reference abnormality on {cs['ref_affected_slices']} slices)")

        st.divider()
        st.markdown("#### Per-Slice Foreground Counts")
        sdf = st.session_state["slice_df"]
        st.dataframe(sdf, use_container_width=True, hide_index=True)

        fig_area, ax_area = plt.subplots(figsize=(10, 3))
        xs = list(range(len(sdf)))
        ax_area.bar(xs, sdf["Predicted px"], color="#00d4aa", alpha=0.8, label="Predicted")
        if "Reference px" in sdf:
            ax_area.plot(xs, sdf["Reference px"], color="#f59e0b", marker=".", label="Reference")
            ax_area.legend(facecolor="#111827", edgecolor="#1f2937", labelcolor="#f9fafb")
        ax_area.set_xlabel("Slice (anatomical order)")
        ax_area.set_ylabel("Foreground pixels")
        ax_area.set_title("Foreground Pixels per Slice")
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
        with bc1: show_img(patient["images"][browse_idx], patient["image_files"][browse_idx])
        with bc2:
            if st.session_state["all_tumor_masks"]:
                show_img(st.session_state["all_tumor_masks"][browse_idx], "Predicted mask")
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

            st.markdown(f"#### Current Slice ({patient['image_files'][idx]})")
            if ev["positive"]:
                e1, e2, e3, e4, e5 = st.columns(5)
                for col, name, key in [
                    (e1, "Dice", "dice"), (e2, "IoU", "iou"),
                    (e3, "Precision", "precision"), (e4, "Recall", "recall"),
                    (e5, "Specificity", "specificity"),
                ]:
                    val = ev[key]
                    with col:
                        st.metric(name, f"{val:.4f}")
                        st.markdown(f'<div style="height:3px;background:{metric_color(val)};border-radius:2px;margin-top:-8px;"></div>',
                                    unsafe_allow_html=True)
                if ev["pred_px"] == 0:
                    st.caption("Empty prediction on an annotated slice: precision is set to 0.")
            else:
                st.markdown("Reference mask is **empty** (negative slice): Dice, IoU, precision and recall "
                            "are undefined.")
                n1, n2 = st.columns(2)
                n1.metric("Predicted foreground px (all false positive)", f"{ev['pred_px']:,}")
                n2.metric("Specificity", f"{ev['specificity']:.4f}")

            st.divider()
            st.markdown("#### Confusion Counts (pixels)")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("True Positive", f"{ev['tp']:,}")
            cm2.metric("False Positive", f"{ev['fp']:,}")
            cm3.metric("True Negative", f"{ev['tn']:,}")
            cm4.metric("False Negative", f"{ev['fn']:,}")

            st.divider()
            st.markdown("#### Visual Comparison")
            vc1, vc2, vc3 = st.columns(3)
            with vc1: show_img(patient["images"][idx], "FLAIR slice")
            with vc2: show_img(st.session_state["tumor_mask"], "Predicted mask")
            with vc3:
                if idx < len(patient["masks"]) and patient["masks"][idx] is not None:
                    show_mask_green(patient["masks"][idx], "Reference mask")

            ev_p = st.session_state["eval_patient"]
            if ev_p is not None:
                st.divider()
                st.markdown("#### This Patient (all annotated slices)")
                vv1, vv2, vv3, vv4 = st.columns(4)
                vv1.metric("Positive slices", ev_p["n_positive"])
                if ev_p["mean_dice"] is not None:
                    sd = f" ± {ev_p['sd_dice']:.3f}" if ev_p["sd_dice"] is not None else ""
                    vv2.metric("Mean Dice (positive slices)", f"{ev_p['mean_dice']:.3f}{sd}")
                vv3.metric("Negative slices", ev_p["n_negative"])
                vv4.metric("Negative slices with a prediction", ev_p["neg_nonempty"])

                pos_rows = [(i, r["dice"]) for i, r in enumerate(ev_p["per_slice"]) if r["positive"]]
                fig_dice, ax_dice = plt.subplots(figsize=(10, 3))
                ax_dice.bar([i for i, _ in pos_rows], [d for _, d in pos_rows], color="#3b82f6", alpha=0.8)
                if ev_p["mean_dice"] is not None:
                    ax_dice.axhline(y=ev_p["mean_dice"], color="#00d4aa", linestyle="--",
                                    label=f"Mean={ev_p['mean_dice']:.3f}")
                ax_dice.set_xlabel("Slice (anatomical order); positive slices only")
                ax_dice.set_ylabel("Dice Score")
                ax_dice.set_title("Per-Slice Dice on Annotated Slices")
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
            st.markdown("#### Mask Measurement Only (no reference mask)")
            st.info("Accuracy cannot be assessed without a reference mask. The outputs below are "
                    "unvalidated mask measurements, not detections.")

            st.divider()
            st.markdown("#### Predicted Mask Summary")
            all_masks = st.session_state["all_tumor_masks"]
            if all_masks:
                total_px = sum(int(np.count_nonzero(m)) for m in all_masks)
                affected = sum(1 for m in all_masks if np.count_nonzero(m) > 0)
                s1, s2 = st.columns(2)
                s1.metric("Predicted foreground pixels", f"{total_px:,}")
                s2.metric("Slices with a prediction", f"{affected}/{len(all_masks)}")

            st.divider()
            st.markdown("#### Visual Result")
            vr1, vr2 = st.columns(2)
            with vr1: show_img(patient["images"][idx], "FLAIR slice")
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
            m_used = st.session_state["method_used"]
            L.append(f"  Pipeline        : version {PIPELINE_VERSION}")
            L.append("  Input channel   : FLAIR (channel 1 of the LGG TIFF)")
            L.append("  Preprocessing   : min-max normalization, Gaussian (sigma=1.0), CLAHE (clip 2.0, 8x8)")
            L.append("  Brain mask      : threshold 15 on normalized image, largest component, hole fill, 9x9 erosion")
            L.append(f"  Segmentation    : {METHOD_LABELS[m_used]} {METHOD_PARAMS[m_used] or ''}")
            L.append(f"  Morphology      : opening then closing ({MORPHOLOGY['kernel_size']}x{MORPHOLOGY['kernel_size']} ellipse, 1 iteration)")
            L.append(f"  Post-processing : keep components {MIN_COMPONENT_AREA} px to {int(MAX_COMPONENT_FRACTION*100)}% of image; largest kept")
            L.append("")

            L.append("-" * 65)
            L.append("  MASK MEASUREMENT (pixels; no spatial calibration available)")
            L.append("-" * 65)
            cs = st.session_state.get("count_stats")
            if cs:
                L.append(f"  Predicted foreground : {cs['pred_total_px']:,} px on {cs['pred_affected_slices']}/{cs['total_slices']} slices")
                if "ref_total_px" in cs:
                    L.append(f"  Reference foreground : {cs['ref_total_px']:,} px on {cs['ref_affected_slices']} slices")
                    if cs["signed_count_error_pct"] is not None:
                        L.append(f"  Signed count error   : {cs['signed_count_error_pct']:+.1f}%  (100*(P-G)/G)")
            L.append("")

            L.append("-" * 65)
            L.append("  EVALUATION AGAINST REFERENCE MASKS")
            L.append("-" * 65)
            ev_p = st.session_state.get("eval_patient")
            if ev_p:
                L.append(f"  Positive slices      : {ev_p['n_positive']}")
                if ev_p["mean_dice"] is not None:
                    L.append(f"  Mean Dice / IoU      : {ev_p['mean_dice']:.4f} / {ev_p['mean_iou']:.4f}")
                    L.append(f"  Mean Prec / Recall   : {ev_p['mean_precision']:.4f} / {ev_p['mean_recall']:.4f}")
                L.append(f"  Negative slices      : {ev_p['n_negative']} ({ev_p['neg_nonempty']} with a non-empty prediction)")
            else:
                L.append("  (no reference masks: outputs are unvalidated measurements)")
            L.append("")
            L.append("  Research prototype; not for clinical use.")
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
