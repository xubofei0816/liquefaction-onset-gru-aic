# Input format

## Motion CSV

Use one CSV per parent recording. The four required headers are:

```text
time_s,h1_g,h2_g,v_g
```

A blank copyable file is provided at `data/input_record_template.csv`.

Requirements:

- `time_s` is in seconds, strictly increasing, and uniformly sampled;
- `h1_g`, `h2_g`, and `v_g` are finite acceleration values in units of `g`;
- `h1_g` and `h2_g` are orthogonal horizontal components from the same
  recording location and time axis;
- `v_g` is the corresponding vertical component;
- all components have the same number of rows; and
- at least 5 s of data remain after conversion to 0.005 s.

The workflow rebases the first sample to `time_s=0`. If the supplied time step
is not 0.005 s, it linearly interpolates all components onto a 0.005 s grid.
It does not convert units, filter, detrend, baseline-correct, align mismatched
components, or fill missing acceleration values.

## Manifest CSV

The notebook reads a manifest with one row per parent recording:

```text
recording_id,csv_path,Mw,partition,reference_tL_sec
```

- `recording_id` is a unique, filesystem-safe identifier.
- `csv_path` is absolute or relative to the manifest location.
- `Mw` is required and numeric.
- `partition` is optional descriptive metadata, such as `new_data`.
- `reference_tL_sec` is optional and is used only as a plot overlay.

The example manifest is `data/example/input_manifest.csv`. Do not place 18
rotations in the manifest. The workflow creates those internally. A blank
copyable manifest is provided at `data/input_manifest_template.csv`.

## Magnitude for data outside the natural-GM domain

The final network expects `Mw` because magnitude was a training feature. For a
centrifuge or other synthetic motion, `Mw` is undefined. Any assigned value is
an external-domain imputation assumption, not a measured property. The paper's
centrifuge evaluation used `Mw=6.9`; that choice should not be generalized
without a separate justification and sensitivity analysis.

