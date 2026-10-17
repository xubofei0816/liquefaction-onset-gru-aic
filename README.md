# Liquefaction-triggering time from surface accelerations

This repository accompanies **“Estimating liquefaction triggering timing from
surface ground motion accelerations using a neural network”** by Bofei Xu and
Adda Athanasopoulos-Zekkos. It contains the trained five-fold Model 6.3
no-time ensemble, its fold-specific scalar scalers, an input template, and a
reproducible inference and Accumulative Inference Curve (AIC) workflow.

## What is included

- `models/model6_3_no_time/`: five trained Keras models and their matching
  two-feature `StandardScaler` objects.
- `notebooks/inference_template.ipynb`: the recommended user entry point.
- `src/liquefaction_timing/workflow.py`: preprocessing, ensemble inference,
  AIC post-processing, plotting, and output functions.
- `data/input_record_template.csv` and `data/input_manifest_template.csv`:
  blank header templates ready to copy.
- `data/example/`: a synthetic three-component record and manifest that show
  the required format. The synthetic values are not research data.
- `metadata/`: the record manifest, reference-label catalog, and event-grouped
  five-fold train/validation assignments used in the study.
- `paper_results/`: compact record-level predictions reported for the GM,
  centrifuge, and production sets. Raw acceleration histories and large
  window-level prediction files are not redistributed.
- `tests/`: checks for preprocessing, AIC logic, and release artifacts.

## Quick start

The saved models were produced with Python 3.10 and TensorFlow/Keras 2.10.
Create the pinned environment before opening the notebook. With Conda:

```powershell
conda env create -f environment.yml
conda activate liquefaction-timing
python -m ipykernel install --user --name liquefaction-timing --display-name "Python (liquefaction-timing)"
jupyter lab
```

Alternatively, use a standard Python 3.10 virtual environment. On Windows,
place the environment at a short path because TensorFlow 2.10 contains deeply
nested files that can exceed the legacy Windows path-length limit when the
environment is created inside a long repository path:

```powershell
py -3.10 -m venv "$env:USERPROFILE\.venvs\liquefaction-timing"
& "$env:USERPROFILE\.venvs\liquefaction-timing\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\.venvs\liquefaction-timing\Scripts\python.exe" -m pip install -r requirements.txt
& "$env:USERPROFILE\.venvs\liquefaction-timing\Scripts\python.exe" -m ipykernel install --user --name liquefaction-timing --display-name "Python (liquefaction-timing)"
& "$env:USERPROFILE\.venvs\liquefaction-timing\Scripts\jupyter-lab.exe"
```

Open `notebooks/inference_template.ipynb`, select the
`Python (liquefaction-timing)` kernel, and run the cells in order. The example
manifest runs a synthetic format-check record. Replace it with your own
manifest and motion CSV files for research use.

## Input contract

Each motion CSV must contain:

| Column | Meaning |
| --- | --- |
| `time_s` | Strictly increasing, uniformly sampled time in seconds |
| `h1_g` | First horizontal surface acceleration in units of `g` |
| `h2_g` | Orthogonal horizontal surface acceleration in units of `g` |
| `v_g` | Vertical surface acceleration in units of `g` |

The manifest must contain `recording_id`, `csv_path`, and `Mw`. An optional
`reference_tL_sec` is plotted but is never used to obtain the prediction.
See [docs/INPUT_FORMAT.md](docs/INPUT_FORMAT.md) for the complete contract.

The workflow does not infer units, perform baseline correction, or impute a
missing magnitude. The user is responsible for providing aligned components
and an appropriate event magnitude. Application to centrifuge or other
non-earthquake records requires an external-domain magnitude assumption and
is outside the primary natural-ground-motion input contract.

## Published inference rule

1. Linearly resample each full component to 0.005 s when needed.
2. Generate 18 horizontal rotations from 0 to 85 degrees in 5-degree steps.
3. Form six time-series channels: `H1`, `H2`, `V`, and their centered
   12-sample rolling standard deviations.
4. Form the scalar input `[rotation angle, Mw]` and transform it with the
   scaler saved for the corresponding fold.
5. Classify every one-sample-stepped 5 s window with each of the five fold
   models, then average the five softmax probability vectors.
6. For each rotation, construct the hard-count Class-1 AIC and the accumulated
   Class-1 softmax probability curve. A window contributes its Class-1
   probability only when Class 1 is its argmax class.
7. Take the pointwise median of each curve over the 18 rotations. Triggering is
   detected only if the peak median hard-count AIC is at least 100.
8. For a detected record, report `tL` as the median time of the global-maximum
   plateau of the median accumulated Class-1 softmax probability curve. If the
   hard-count threshold is not reached, report no detection and no `tL`.

The threshold of 100 was selected from out-of-fold validation records. Among
thresholds evaluated in increments of 50, it was the highest threshold that
retained triggering detections in the largest attainable number of validation
records. No external test set was used to tune the threshold.

## Outputs

For every input record, the notebook writes:

- `summary.json`: detection decision, predicted `tL`, and run metadata;
- `ensemble_predictions.npz`: five-fold-averaged softmax probabilities for all
  18 rotations;
- `aic_curves.csv`: the two median curves used for detection and timing;
- `inference_plot.png`: the AIC and three acceleration components in the style
  used for the paper; and
- a continuously updated `inference_summary.csv` at the output root.

## Scope and limitations

The training data contain liquefied natural ground-motion records and no
robustly annotated non-liquefied natural ground-motion records. Performance
was stronger on the GM test set than on the centrifuge test set. The model can
also be unreliable when triggering occurs within approximately one input
window of the beginning or end of a record. A detection is a model output, not
an independent engineering determination of liquefaction. Review the article
and [MODEL_CARD.md](MODEL_CARD.md) before downstream use.

## Data availability

The repository intentionally excludes raw acceleration histories obtained
through personal communications because redistribution permission was not
obtained. Public-source records should be downloaded from the repositories
cited in the article. K-NET and KiK-net records and station information are
available from NIED under DOI
[10.17598/NIED.0004](https://doi.org/10.17598/NIED.0004). LEAP-UCD-2017 source
records are available through the DesignSafe citations listed in the article.

## Citation and license

Use `CITATION.cff` for software citation metadata. The archived release is
available at [https://doi.org/10.5281/zenodo.23430058](https://doi.org/10.5281/zenodo.23430058).

Copyright (c) 2026 Bofei Xu. The code, notebooks, trained model files, and
repository-generated metadata and result tables are distributed under the
[BSD 3-Clause License](LICENSE). The license applies only to material included
in this repository. It does not grant rights to the original ground-motion or
centrifuge records, which are not redistributed and remain subject to their
respective providers' terms.

