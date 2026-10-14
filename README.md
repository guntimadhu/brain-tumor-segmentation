# Classical FLAIR Abnormality Segmentation and Mask Measurement

Classical image-processing pipeline (no machine learning) that segments FLAIR
abnormalities in brain MRI slices and reports foreground pixel counts, with a
Streamlit interface. Research prototype; not for clinical use.

Live app: https://minor2-brain-tumor-segmentation.streamlit.app

## Pipeline (version 2.0.0, `src/pipeline.py`)

FLAIR channel (index 1 of the LGG TIFF) → min-max normalisation → Gaussian
(σ = 1.0) → CLAHE (clip 2.0, 8×8) → brain mask from the normalised image
(threshold 15, largest component, hole fill, 9×9 erosion) → threshold
(brain-masked 75th percentile by default; Otsu and adaptive Gaussian also
available) → opening then closing (5×5 ellipse) → keep components of 100 px to
25 % of the image → largest component → foreground pixel count.

No physical areas or volumes are reported: the TIFF data carry no verified
spatial calibration.

## Setup and app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

LGG Segmentation Dataset (Buda et al.), Kaggle `mateuszbuda/lgg-mri-segmentation`
(110 patients, `kaggle_3m`). Files are paired by name (`<case>_<n>.tif` with
`<case>_<n>_mask.tif`) and ordered by slice number. For the app, copy patient
folders into `data/raw/`.

## Reproducing the paper results

```bash
python experiments/evaluate.py --data <path>/kaggle_3m --out results
python experiments/make_figures.py --data <path>/kaggle_3m --results results
```

* Development cohort: `experiments/dev_patients.txt` (12 patients) — used for
  all design choices and for selecting the percentile (sweep 70–95).
* Held-out test cohort: the remaining 98 patients, evaluated once with the
  frozen configuration.

Outputs in `results/`: `summary.json` (configuration, environment, definitions,
all aggregate results, Wilcoxon tests, timing), `per_slice_*.csv`,
`per_patient_main.csv`, `file_list.csv` (exact image/mask pairs and split),
`predictions/*.npz` (bit-packed predicted masks per patient), `figures/`.

`python test_pipeline.py` runs a smoke test, including an image/mask pairing check.
