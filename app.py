import datetime
import io
import os
import zipfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.io_utils import handle_upload, load_mri_and_mask, get_patient_list, load_patient_data  # noqa: E402
from src.preprocessing import compare_filters, apply_clahe, normalize_intensity  # noqa: E402
from src.segmentation import segment_tumor_candidate  # noqa: E402
from src.morphology import get_morphology_comparison  # noqa: E402
from src.postprocessing import find_connected_components  # noqa: E402
from src.measurements import foreground_pixels, summarize_foreground, slice_dataframe  # noqa: E402
from src.evaluation import slice_metrics, patient_metrics  # noqa: E402
from src.pipeline import (run_pipeline, METHOD_LABELS, METHOD_PARAMS, DEFAULT_METHOD,  # noqa: E402
                          PIPELINE_VERSION, MORPHOLOGY, MIN_COMPONENT_AREA, MAX_COMPONENT_FRACTION)

GITHUB = "https://github.com/guntimadhu/brain-tumor-segmentation"

st.set_page_config(page_title="BrainScan · FLAIR segmentation", page_icon=":material/neurology:",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Design tokens (UI/UX Pro Max: accessible clinical style, teal primary,
# Figtree headings / Noto Sans body, visible focus, reduced motion)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@500;600;700&family=Noto+Sans:wght@400;500;600&display=swap');
:root {
  --primary: #0E7490; --primary-dark: #155E75; --primary-soft: #ECFEFF;
  --text: #0F172A; --muted: #475569; --border: #CBD5E1; --surface: #FFFFFF; --bg: #F8FAFC;
  --success: #15803D; --warning: #B45309; --danger: #B91C1C;
}
html, body, .stApp, p, li, label, input, textarea, select, button p, [data-testid="stMarkdownContainer"] { font-family: 'Noto Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, h4, h5 { font-family: 'Figtree', 'Noto Sans', sans-serif !important; color: var(--text) !important; letter-spacing: -0.01em; }
.block-container { max-width: 1200px; padding-top: 1.2rem; padding-bottom: 3rem; }
p, li, label, .stMarkdown { font-size: 1rem; line-height: 1.6; }

/* app bar */
.appbar { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px 22px; margin-bottom:14px; }
.brand { display:flex; align-items:center; gap:14px; }
.brand-icon { width:44px; height:44px; border-radius:12px; background: var(--primary); display:flex; align-items:center; justify-content:center; }
.brand-title { font-family:'Figtree',sans-serif; font-weight:700; font-size:1.35rem; color:var(--text); line-height:1.2; }
.brand-sub { color:var(--muted); font-size:0.92rem; }
.chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip { font-size:0.8rem; font-weight:600; padding:4px 10px; border-radius:999px; border:1px solid var(--border); color:var(--muted); background:#F1F5F9; }
.chip-warn { color: var(--warning); background:#FFFBEB; border-color:#FCD34D; }

/* stepper */
.stepper { display:flex !important; width:100%; align-items:flex-start; gap:0; list-style:none; padding:14px 8px; margin:0 0 6px 0;
  background:var(--surface); border:1px solid var(--border); border-radius:14px; }
.step { flex:1 1 0; display:flex !important; flex-direction:column; align-items:center; position:relative; text-align:center; min-width:0; }
.step:not(:last-child)::after { content:""; position:absolute; top:18px; left:calc(50% + 22px); right:calc(-50% + 22px); height:2px; background:var(--border); }
.step.done:not(:last-child)::after { background: var(--primary); }
.dot { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-weight:700; font-family:'Figtree',sans-serif; border:2px solid var(--border); background:var(--surface); color:var(--muted); }
.step.done .dot { background:var(--primary); border-color:var(--primary); color:#fff; }
.step.current .dot { border-color:var(--primary); color:var(--primary); box-shadow:0 0 0 4px #CFFAFE; }
.step-label { margin-top:6px; font-size:0.85rem; font-weight:600; color:var(--muted); }
.step.current .step-label, .step.done .step-label { color:var(--text); }
.step-hint { font-size:0.75rem; color:var(--muted); }
@media (max-width: 640px) { .step-hint { display:none; } .step-label { font-size:0.75rem; } }

/* cards, metrics, images */
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
[data-testid="stMetricLabel"] p { color:var(--muted) !important; font-size:0.85rem !important; }
[data-testid="stMetricValue"] { color:var(--text) !important; font-family:'Figtree',sans-serif; font-weight:700; }
[data-testid="stImage"] img { border-radius:10px; border:1px solid var(--border); background:#000; }
[data-testid="stImageCaption"], [data-testid="caption"] { color:var(--muted) !important; font-size:0.85rem !important; }
[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border) !important; border-radius:12px !important; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed var(--primary) !important; background:var(--primary-soft) !important; border-radius:12px !important; }

[data-testid="stButtonGroup"] button { min-height:44px; padding-left:16px; padding-right:16px; cursor:pointer; }
/* buttons: 44px targets, visible focus, short transitions */
.stButton > button, .stDownloadButton > button { min-height:44px; border-radius:10px; font-weight:600; cursor:pointer;
  transition: background-color .2s ease, box-shadow .2s ease, border-color .2s ease; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [role="tab"]:focus-visible,
input:focus-visible, [role="slider"]:focus-visible { outline:3px solid var(--primary) !important; outline-offset:2px !important; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover { background:var(--primary-dark); border-color:var(--primary-dark); }

.section-title { font-family:'Figtree',sans-serif; font-weight:700; font-size:1.5rem; margin:4px 0 2px 0; color:var(--text); }
.section-lead { color:var(--muted); margin-bottom:12px; }
.legend { display:inline-flex; align-items:center; gap:6px; margin-right:14px; font-size:0.85rem; color:var(--muted); }
.swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
.footer { color:var(--muted); font-size:0.85rem; text-align:center; margin-top:28px; }
.footer a { color: var(--primary); }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
""", unsafe_allow_html=True)

BRAIN_SVG = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
             '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>'
             '<path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>'
             '<path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>'
             '<path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>'
             '<path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/>'
             '<path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>')
CHECK_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" '
             'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>')

PRED_RGB = (8, 145, 178)    # prediction fill (teal)
REF_RGB = (234, 88, 12)     # reference contour (orange); colour is not the only cue (fill vs. outline)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
STEPS = [
    ("Load data", "MRI slices and optional masks"),
    ("Segment", "Run the pipeline"),
    ("Measure", "Foreground pixel counts"),
    ("Evaluate", "Compare with reference"),
    ("Report", "Summary and downloads"),
]
DEFAULTS = {"step": 0, "patient": None, "slice_idx": 0, "method_used": DEFAULT_METHOD,
            "all_masks": None, "count_stats": None, "slice_df": None, "eval_patient": None,
            "processed": False}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_results():
    for k in ["all_masks", "count_stats", "slice_df", "eval_patient"]:
        st.session_state[k] = None
    st.session_state["processed"] = False


def set_patient(p):
    reset_results()
    st.session_state["patient"] = p
    st.session_state["slice_idx"] = 0


def start_over():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for k in ["_last_key", "_last_local"]:
        st.session_state.pop(k, None)


def go(step):
    st.session_state["step"] = step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_display(img):
    out = img.astype(np.float64)
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-8:
        out = (out - mn) / (mx - mn) * 255.0
    return out.astype(np.uint8)


def show(img, caption):
    st.image(to_display(img) if img.ndim == 2 else img, caption=caption, width="stretch", clamp=True)


def overlay(img, pred=None, ref=None):
    rgb = cv2.cvtColor(to_display(img), cv2.COLOR_GRAY2RGB)
    if pred is not None and pred.any():
        rgb[pred] = (0.45 * rgb[pred] + 0.55 * np.array(PRED_RGB)).astype(np.uint8)
    if ref is not None and ref.any():
        cs, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, REF_RGB, 2)
    return rgb


def legend(with_ref=True):
    html = (f'<span class="legend"><span class="swatch" style="background:rgb{PRED_RGB}"></span>Prediction (filled)</span>')
    if with_ref:
        html += f'<span class="legend"><span class="swatch" style="border:2px solid rgb{REF_RGB}"></span>Reference (outline)</span>'
    st.markdown(html, unsafe_allow_html=True)


def ref_mask(patient, idx):
    masks = patient["masks"]
    if patient["has_masks"] and idx < len(masks) and masks[idx] is not None:
        return masks[idx] > 127
    return None


def slice_picker(patient, key):
    n = patient["slice_count"]
    if n > 1:
        idx = st.slider("Slice", 0, n - 1, st.session_state["slice_idx"], key=key,
                        help="Slices are shown in anatomical order (by slice number in the file name).")
        st.session_state["slice_idx"] = idx
    idx = st.session_state["slice_idx"]
    st.caption(f"Slice {idx + 1} of {n} · `{patient['image_files'][idx]}`")
    return idx


def header(title, lead):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-lead">{lead}</div>',
                unsafe_allow_html=True)


def stepper():
    cur, done = st.session_state["step"], unlocked_until()
    items = []
    for i, (label, hint) in enumerate(STEPS):
        state = "current" if i == cur else ("done" if i < done and i < cur else "")
        dot = CHECK_SVG if state == "done" else str(i + 1)
        aria = ' aria-current="step"' if i == cur else ""
        items.append(f'<div class="step {state}" role="listitem"{aria}><div class="dot">{dot}</div>'
                     f'<div class="step-label">{label}</div><div class="step-hint">{hint}</div></div>')
    st.markdown(f'<nav aria-label="Progress"><div class="stepper" role="list">{"".join(items)}</div></nav>',
                unsafe_allow_html=True)


def unlocked_until():
    """Highest step index the user may open."""
    if st.session_state["patient"] is None:
        return 0
    if not st.session_state["processed"]:
        return 1
    return len(STEPS) - 1


BLOCKED_REASON = {1: "Load MRI data first.", 2: "Run the segmentation first."}


def nav(where="bottom"):
    cur = st.session_state["step"]
    limit = unlocked_until()
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if cur > 0:
            st.button(f"Back: {STEPS[cur - 1][0]}", icon=":material/arrow_back:", on_click=go, args=(cur - 1,),
                      key=f"back_{where}", width="stretch")
    with c3:
        if cur < len(STEPS) - 1:
            blocked = cur + 1 > limit
            st.button(f"Next: {STEPS[cur + 1][0]}", icon=":material/arrow_forward:", icon_position="right",
                      type="primary", on_click=go, args=(cur + 1,), disabled=blocked,
                      key=f"next_{where}", width="stretch")
            if blocked:
                st.caption(BLOCKED_REASON.get(cur + 1, ""))
        else:
            st.button("Start over", icon=":material/restart_alt:", on_click=start_over,
                      key=f"restart_{where}", width="stretch")


def styled_axes(ax, fig):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#334155")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# App bar + stepper
# ---------------------------------------------------------------------------
st.markdown(f"""
<header class="appbar">
  <div class="brand">
    <div class="brand-icon">{BRAIN_SVG}</div>
    <div>
      <div class="brand-title">BrainScan</div>
      <div class="brand-sub">Classical FLAIR abnormality segmentation and mask measurement</div>
    </div>
  </div>
  <div class="chips">
    <span class="chip">Pipeline v{PIPELINE_VERSION}</span>
    <span class="chip">No machine learning</span>
    <span class="chip chip-warn">Research prototype · not for clinical use</span>
  </div>
</header>
""", unsafe_allow_html=True)

if st.session_state["step"] > unlocked_until():
    st.session_state["step"] = unlocked_until()
stepper()
step = st.session_state["step"]
patient = st.session_state["patient"]

# ===========================================================================
# STEP 1 — LOAD DATA
# ===========================================================================
if step == 0:
    header("1. Load MRI data", "Use the FLAIR TIFF slices of the LGG Segmentation Dataset. "
           "A reference mask is optional; without it the app reports measurements only.")

    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    local_patients = get_patient_list(local_dir)
    sources = ["Single slice", "ZIP of slices"] + (["Local patient folder"] if local_patients else [])
    source = st.segmented_control("Input source", sources, default=sources[0], key="source")

    if source == "Single slice":
        c1, c2 = st.columns(2)
        with c1:
            mri_file = st.file_uploader("FLAIR slice (TIFF, PNG or JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
        with c2:
            mask_file = st.file_uploader("Reference mask (optional)", type=["tif", "tiff", "png", "jpg", "jpeg"])
        if mri_file is not None:
            ckey = f"{mri_file.name}_{mri_file.size}_{mask_file.name if mask_file else ''}_{mask_file.size if mask_file else ''}"
            if st.session_state.get("_last_key") != ckey:
                try:
                    set_patient(load_mri_and_mask(mri_file, mask_file))
                    st.session_state["_last_key"] = ckey
                except Exception as e:
                    st.error(f"Could not read the file: {e}")
    elif source == "ZIP of slices":
        zip_file = st.file_uploader("ZIP archive with slices; masks end in _mask (e.g. case_12.tif and case_12_mask.tif)",
                                    type=["zip"])
        if zip_file is not None:
            zkey = f"zip_{zip_file.name}_{zip_file.size}"
            if st.session_state.get("_last_key") != zkey:
                try:
                    with st.spinner("Reading archive…"):
                        set_patient(handle_upload(zip_file))
                    st.session_state["_last_key"] = zkey
                except Exception as e:
                    st.error(f"Could not read the archive: {e}")
    elif source == "Local patient folder":
        names = [os.path.basename(p) for p in local_patients]
        sel = st.selectbox(f"{len(names)} patients in data/raw", names, index=None, placeholder="Choose a patient")
        if sel is not None and st.session_state.get("_last_local") != sel:
            with st.spinner(f"Loading {sel}…"):
                set_patient(load_patient_data(local_patients[names.index(sel)]))
            st.session_state["_last_local"] = sel

    patient = st.session_state["patient"]
    if patient is None:
        st.info("No data loaded yet. Choose an input source above.", icon=":material/info:")
    else:
        st.success(f"Loaded **{patient['patient_id']}**: {patient['slice_count']} slice(s), "
                   f"{patient['image_shape'][0]}×{patient['image_shape'][1]} px.", icon=":material/check_circle:")
        m1, m2, m3 = st.columns(3)
        m1.metric("Slices", patient["slice_count"])
        m2.metric("Reference masks", "Yes" if patient["has_masks"] else "No")
        m3.metric("Slices with annotated abnormality", len(patient.get("tumor_slices", [])) if patient["has_masks"] else "–")
        idx = slice_picker(patient, "pick_load")
        c1, c2 = st.columns(2)
        with c1:
            show(patient["images"][idx], "FLAIR slice")
        with c2:
            ref = ref_mask(patient, idx)
            if ref is not None:
                show((ref * 255).astype(np.uint8), "Reference mask" + ("" if ref.any() else " (empty)"))
            else:
                st.info("No reference mask for this slice.", icon=":material/info:")

    with st.expander("About the method and its limitations", icon=":material/help:"):
        st.markdown(f"""
- **Input:** channel 1 (FLAIR) of the three-channel LGG TIFF (channel order documented as pre-contrast, FLAIR, post-contrast).
- **Pipeline:** min–max normalization → Gaussian (σ = 1.0) → CLAHE (clip 2.0) → brain mask from the normalized image →
  threshold → opening then closing ({MORPHOLOGY['kernel_size']}×{MORPHOLOGY['kernel_size']}) → keep components of
  {MIN_COMPONENT_AREA} px to {int(MAX_COMPONENT_FRACTION * 100)}% of the image → largest component.
- **Held-out result (98 patients):** patient-level Dice 0.389 ± 0.249 for the default 75th-percentile threshold.
- **Known limitations:** the pipeline keeps a component on almost every slice, so it usually predicts a region
  even where no abnormality is annotated; pixel counts therefore overestimate abnormality extent.
- **Measurements are pixel counts.** No physical areas or volumes are reported.
- Code, data splits and results: [{GITHUB}]({GITHUB})
""")
    nav()

# ===========================================================================
# STEP 2 — SEGMENT
# ===========================================================================
elif step == 1:
    header("2. Segment", "Choose a threshold method and run the pipeline on every slice. "
           "Defaults are the configuration evaluated in the paper.")
    nav("top")
    method = st.radio("Threshold method", list(METHOD_LABELS), format_func=lambda m: METHOD_LABELS[m] +
                      (" (default)" if m == DEFAULT_METHOD else ""),
                      index=list(METHOD_LABELS).index(st.session_state["method_used"]), horizontal=True)
    st.caption(f"Parameters: {METHOD_PARAMS[method] or 'threshold chosen automatically'}; "
               f"opening then closing, {MORPHOLOGY['kernel_size']}×{MORPHOLOGY['kernel_size']} ellipse; "
               f"components {MIN_COMPONENT_AREA} px to {int(MAX_COMPONENT_FRACTION * 100)}% of the image; largest kept.")
    if st.session_state["processed"] and method != st.session_state["method_used"]:
        st.warning("The method changed. Run the pipeline again to update the results.", icon=":material/sync_problem:")

    if st.button(f"Run on all {patient['slice_count']} slices", type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0, text="Starting…")
        masks = []
        for i, img in enumerate(patient["images"]):
            masks.append(run_pipeline(img, method=method)["final"])
            bar.progress((i + 1) / patient["slice_count"], text=f"Slice {i + 1} of {patient['slice_count']}")
        refs = patient["masks"] if patient["has_masks"] else None
        st.session_state.update({
            "all_masks": masks, "method_used": method, "processed": True,
            "count_stats": summarize_foreground(masks, refs),
            "slice_df": slice_dataframe(masks, patient["image_files"], refs),
            "eval_patient": patient_metrics(masks, patient["masks"]) if patient["has_masks"] else None,
        })
        bar.empty()
        st.rerun()

    if st.session_state["processed"]:
        m = st.session_state["method_used"]
        st.success(f"Done: {METHOD_LABELS[m]} applied to {patient['slice_count']} slices. "
                   "Inspect any slice below, then continue to Measure.", icon=":material/check_circle:")
        idx = slice_picker(patient, "pick_seg")
        img = patient["images"][idx]
        res = run_pipeline(img, method=m)
        steps_ = res["pp_steps"]
        ref = ref_mask(patient, idx)

        st.markdown("##### Result")
        legend(ref is not None)
        c1, c2, c3 = st.columns(3)
        with c1:
            show(img, "FLAIR slice")
        with c2:
            show(res["final"], "Predicted mask")
        with c3:
            st.image(overlay(img, res["final"] > 0, ref), caption="Overlay", width="stretch")

        with st.expander("Step 1 · Preprocessing", icon=":material/tune:"):
            cols = st.columns(4)
            for col, (k, cap) in zip(cols, [("original", "Input"), ("normalized", "Normalized"),
                                            ("filtered", "Gaussian (σ = 1.0)"), ("enhanced", "CLAHE (clip 2.0)")]):
                with col:
                    show(steps_[k], cap)
            st.caption("Filter comparison on the normalized slice")
            fc = compare_filters(normalize_intensity(img))
            cols = st.columns(3)
            for col, (k, cap) in zip(cols, [("none", "No filter"), ("gaussian", "Gaussian"), ("median", "Median (3×3)")]):
                with col:
                    show(fc[k], cap)
        with st.expander("Step 2 · Brain mask", icon=":material/masks:"):
            cols = st.columns(3)
            with cols[0]:
                show(steps_["normalized"], "Normalized slice")
            with cols[1]:
                show(res["brain_mask"], "Brain mask")
            with cols[2]:
                show(cv2.bitwise_and(res["enhanced"], res["enhanced"], mask=res["brain_mask"]), "CLAHE image inside mask")
            st.caption("Computed before CLAHE, because CLAHE raises background noise above the tissue threshold.")
        with st.expander("Step 3 · Threshold methods (before morphology)", icon=":material/compare:"):
            cols = st.columns(3)
            for col, meth in zip(cols, METHOD_LABELS):
                cand = segment_tumor_candidate(res["enhanced"], meth, METHOD_PARAMS[meth], brain_mask=res["brain_mask"])
                with col:
                    show(cand["binary_mask"], METHOD_LABELS[meth] + (" · selected" if meth == m else ""))
                    if cand["threshold_value"] is not None:
                        st.caption(f"Threshold: {float(cand['threshold_value']):.0f}")
        with st.expander("Step 4 · Morphology", icon=":material/blur_on:"):
            mc = get_morphology_comparison(res["seg"]["binary_mask"])
            names = [("original", "Candidate"), ("erosion", "Erosion"), ("dilation", "Dilation"),
                     ("opening", "Opening"), ("closing", "Closing"), ("opening_then_closing", "Opening → closing (used)")]
            for row in (names[:3], names[3:]):
                cols = st.columns(3)
                for col, (k, cap) in zip(cols, row):
                    with col:
                        show(mc[k], cap)
        with st.expander("Step 5 · Component selection", icon=":material/filter_alt:"):
            cc = find_connected_components(res["morph"]["output_mask"])
            sizes = sorted(cc["component_sizes"], reverse=True)[:5]
            st.caption(f"{cc['num_components']} components after morphology; largest: "
                       f"{', '.join(str(s) for s in sizes) or 'none'} px")
            cols = st.columns(3)
            for col, (im, cap) in zip(cols, [(res["morph"]["output_mask"], "After morphology"),
                                             (res["filtered"], f"Components {MIN_COMPONENT_AREA} px to {int(MAX_COMPONENT_FRACTION*100)}%"),
                                             (res["final"], "Largest component (final)")]):
                with col:
                    show(im, cap)
    nav()

# ===========================================================================
# STEP 3 — MEASURE
# ===========================================================================
elif step == 2:
    header("3. Measure", "Foreground pixel counts of the predicted masks. Verified spatial calibration is not "
           "available for these TIFF images, so no physical areas or volumes are reported.")
    nav("top")
    cs, sdf = st.session_state["count_stats"], st.session_state["slice_df"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted foreground (all slices)", f"{cs['pred_total_px']:,} px")
    c2.metric("Slices with a prediction", f"{cs['pred_affected_slices']} / {cs['total_slices']}")
    if "ref_total_px" in cs:
        c3.metric("Reference foreground (all slices)", f"{cs['ref_total_px']:,} px")
        if cs["signed_count_error_pct"] is not None:
            st.markdown(f"Signed count error **{cs['signed_count_error_pct']:+.1f}%** "
                        f"(100 × (predicted − reference) / reference; reference abnormality on "
                        f"{cs['ref_affected_slices']} slices).")
            neg_pred = sum(int(r["pred_px"]) for r in st.session_state["eval_patient"]["per_slice"]
                           if not r["positive"]) if st.session_state["eval_patient"] else 0
            if neg_pred:
                st.caption(f"{neg_pred:,} predicted pixels lie on slices whose reference mask is empty.")

    if len(sdf) > 1:
        fig, ax = plt.subplots(figsize=(10, 3))
        xs = np.arange(len(sdf))
        ax.bar(xs, sdf["Predicted px"], color="#0E7490", label="Predicted")
        if "Reference px" in sdf:
            ax.plot(xs, sdf["Reference px"], color="#EA580C", marker="o", ms=3, lw=1.5, label="Reference")
        ax.set_xlabel("Slice (anatomical order)")
        ax.set_ylabel("Foreground pixels")
        ax.legend(frameon=False)
        styled_axes(ax, fig)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("##### Per-slice counts")
    st.dataframe(sdf, width="stretch", hide_index=True)

    st.markdown("##### Inspect a slice")
    idx = slice_picker(patient, "pick_meas")
    mask = st.session_state["all_masks"][idx]
    fp = foreground_pixels(mask)
    ref = ref_mask(patient, idx)
    legend(ref is not None)
    c1, c2, c3 = st.columns(3)
    with c1:
        show(patient["images"][idx], "FLAIR slice")
    with c2:
        show(mask, f"Predicted mask · {fp['foreground_pixels']:,} px ({fp['foreground_percentage']:.2f}% of image)")
    with c3:
        st.image(overlay(patient["images"][idx], mask > 0, ref), caption="Overlay", width="stretch")
    nav()

# ===========================================================================
# STEP 4 — EVALUATE
# ===========================================================================
elif step == 3:
    header("4. Evaluate", "Agreement with the reference masks. Overlap metrics are computed on annotated "
           "(positive) slices only; empty-reference slices are reported separately.")
    nav("top")
    ev_p = st.session_state["eval_patient"]
    if ev_p is None:
        st.info("No reference masks were provided, so accuracy cannot be assessed. The outputs of this run "
                "are unvalidated mask measurements.", icon=":material/info:")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positive slices", ev_p["n_positive"])
        if ev_p["mean_dice"] is not None:
            sd = f"SD {ev_p['sd_dice']:.3f} across positive slices" if ev_p["sd_dice"] is not None else None
            c2.metric("Mean Dice", f"{ev_p['mean_dice']:.3f}", help=sd)
            c3.metric("Precision / recall", f"{ev_p['mean_precision']:.2f} / {ev_p['mean_recall']:.2f}")
        c4.metric("Empty slices predicted", f"{ev_p['neg_nonempty']} / {ev_p['n_negative']}",
                  help="Slices whose reference mask is empty but which still received a predicted region")
        if ev_p["sd_dice"] is not None:
            st.caption(f"Mean Dice over {ev_p['n_positive']} positive slices: "
                       f"{ev_p['mean_dice']:.3f} ± {ev_p['sd_dice']:.3f} (SD).")

        pos = [(i, r["dice"]) for i, r in enumerate(ev_p["per_slice"]) if r["positive"]]
        if pos:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar([i for i, _ in pos], [d for _, d in pos], color="#0E7490")
            if ev_p["mean_dice"] is not None:
                ax.axhline(ev_p["mean_dice"], color="#EA580C", ls="--", lw=1.5, label=f"Mean {ev_p['mean_dice']:.3f}")
                ax.legend(frameon=False)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Slice (anatomical order); annotated slices only")
            ax.set_ylabel("Dice")
            styled_axes(ax, fig)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.markdown("##### Inspect a slice")
        idx = slice_picker(patient, "pick_eval")
        ref = ref_mask(patient, idx)
        mask = st.session_state["all_masks"][idx]
        if ref is None:
            st.info("This slice has no reference mask.", icon=":material/info:")
        else:
            ev = slice_metrics(mask > 0, ref)
            if ev["positive"]:
                cols = st.columns(5)
                for col, (name, key) in zip(cols, [("Dice", "dice"), ("IoU", "iou"), ("Precision", "precision"),
                                                   ("Recall", "recall"), ("Specificity", "specificity")]):
                    col.metric(name, f"{ev[key]:.3f}")
                if ev["pred_px"] == 0:
                    st.caption("Empty prediction on an annotated slice: precision is set to 0.")
            else:
                st.markdown("The reference mask of this slice is **empty**: Dice, IoU, precision and recall are "
                            f"undefined. Predicted pixels here are false positives: **{ev['pred_px']:,} px** "
                            f"(specificity {ev['specificity']:.4f}).")
            with st.expander("Pixel counts (TP / FP / FN / TN)"):
                st.write(f"TP {ev['tp']:,} · FP {ev['fp']:,} · FN {ev['fn']:,} · TN {ev['tn']:,}")
            legend(True)
            c1, c2, c3 = st.columns(3)
            with c1:
                show(mask, "Predicted mask")
            with c2:
                show((ref * 255).astype(np.uint8), "Reference mask")
            with c3:
                st.image(overlay(patient["images"][idx], mask > 0, ref), caption="Overlay", width="stretch")
    nav()

# ===========================================================================
# STEP 5 — REPORT
# ===========================================================================
elif step == 4:
    header("5. Report", "Summary of this run with downloadable results.")
    m = st.session_state["method_used"]
    cs, ev_p = st.session_state["count_stats"], st.session_state["eval_patient"]
    L = ["BRAINSCAN – ANALYSIS REPORT", "=" * 62,
         f"Patient ID          : {patient['patient_id']}",
         f"Generated           : {datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"Slices              : {patient['slice_count']} ({patient['image_shape'][0]} x {patient['image_shape'][1]} px)",
         f"Reference masks     : {'available' if patient['has_masks'] else 'not available'}", "",
         "PIPELINE", "-" * 62,
         f"Version             : {PIPELINE_VERSION}",
         "Input channel       : FLAIR (channel 1 of the LGG TIFF)",
         "Preprocessing       : min-max normalization, Gaussian (sigma 1.0), CLAHE (clip 2.0, 8x8)",
         "Brain mask          : threshold 15 on normalized image, largest component, hole fill, 9x9 erosion",
         f"Threshold           : {METHOD_LABELS[m]} {METHOD_PARAMS[m] or ''}",
         f"Morphology          : opening then closing ({MORPHOLOGY['kernel_size']}x{MORPHOLOGY['kernel_size']} ellipse)",
         f"Components          : {MIN_COMPONENT_AREA} px to {int(MAX_COMPONENT_FRACTION * 100)}% of image; largest kept", "",
         "MASK MEASUREMENT (pixels; no spatial calibration available)", "-" * 62,
         f"Predicted foreground: {cs['pred_total_px']:,} px on {cs['pred_affected_slices']}/{cs['total_slices']} slices"]
    if "ref_total_px" in cs:
        L.append(f"Reference foreground: {cs['ref_total_px']:,} px on {cs['ref_affected_slices']} slices")
        if cs["signed_count_error_pct"] is not None:
            L.append(f"Signed count error  : {cs['signed_count_error_pct']:+.1f}%  (100*(P-G)/G)")
    L += ["", "EVALUATION AGAINST REFERENCE MASKS", "-" * 62]
    if ev_p:
        L.append(f"Positive slices     : {ev_p['n_positive']}")
        if ev_p["mean_dice"] is not None:
            L.append(f"Mean Dice / IoU     : {ev_p['mean_dice']:.4f} / {ev_p['mean_iou']:.4f}")
            L.append(f"Mean precision/recall: {ev_p['mean_precision']:.4f} / {ev_p['mean_recall']:.4f}")
        L.append(f"Negative slices     : {ev_p['n_negative']} ({ev_p['neg_nonempty']} with a non-empty prediction)")
    else:
        L.append("No reference masks: outputs are unvalidated measurements.")
    L += ["", "Research prototype; not for clinical use.", f"Source: {GITHUB}"]
    report = "\n".join(L)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.code(report, language=None)
    with c2:
        st.markdown("##### Downloads")
        pid = patient["patient_id"]
        st.download_button("Report (TXT)", report, f"brainscan_report_{pid}.txt", "text/plain",
                           icon=":material/description:", width="stretch", type="primary")
        df = st.session_state["slice_df"].copy()
        if ev_p:
            per = iter(ev_p["per_slice"])
            rows = [next(per) if (patient["masks"][i] is not None) else None for i in range(len(df))]
            for k in ["dice", "iou", "precision", "recall", "specificity"]:
                df[k.capitalize()] = [None if r is None or r[k] is None else round(r[k], 4) for r in rows]
        st.download_button("Per-slice results (CSV)", df.to_csv(index=False), f"brainscan_slices_{pid}.csv",
                           "text/csv", icon=":material/table:", width="stretch")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, mk in zip(patient["image_files"], st.session_state["all_masks"]):
                ok, png = cv2.imencode(".png", (mk > 0).astype(np.uint8) * 255)
                zf.writestr(os.path.splitext(name)[0] + "_pred.png", png.tobytes())
        st.download_button("Predicted masks (ZIP of PNG)", buf.getvalue(), f"brainscan_masks_{pid}.zip",
                           "application/zip", icon=":material/folder_zip:", width="stretch")
        st.caption("Masks are binary PNGs (0/255) with the same file names as the input slices.")
    nav()

st.markdown(f'<div class="footer">BrainScan · pipeline v{PIPELINE_VERSION} · '
            f'<a href="{GITHUB}" target="_blank" rel="noopener">Source code and results</a> · '
            'Research prototype, not for clinical use</div>', unsafe_allow_html=True)
