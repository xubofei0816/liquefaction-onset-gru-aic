# Reproduction details

This file states the fixed operations implemented in
`src/liquefaction_timing/workflow.py`.

## Preprocessing

1. Components are represented in units of `g` and resampled to `dt=0.005 s`
   by linear interpolation when necessary.
2. Components are aligned and equal in length before inference.
3. The original horizontal pair is defined as 0 degrees. For angle `a`:

   ```text
   H1(a) = cos(a) H1 - sin(a) H2
   H2(a) = sin(a) H1 + cos(a) H2
   ```

4. Angles are `0, 5, ..., 85` degrees.
5. Each rotated component receives a centered, 12-sample pandas rolling
   standard deviation with `min_periods=1`, the pandas sample-standard-
   deviation default, and `NaN` values filled with zero.
6. The feature order is exactly:

   ```text
   H1, H2, V, rolling_std(H1), rolling_std(H2), rolling_std(V)
   ```

7. Every possible 1000-sample window is evaluated with a one-sample stride.
8. The raw scalar order is exactly `[rotation angle in degrees, Mw]`. Fold `n`
   uses `preprocessing/fold_n/scaler_single.joblib`.

## Fold ensemble

The five complete fold models are loaded sequentially. A model is loaded once
per parent record and used for all 18 rotations. For each rotation and window,
the five three-class softmax vectors are averaged arithmetically.

## AIC calculation

Let `c_j` be the argmax class of window `j`, and let `p_j1` be its Class-1
softmax probability. Every window spans 1000 samples.

- Hard-count contribution: 1 over the window support if `c_j=1`, otherwise 0.
- Accumulated-softmax contribution: `p_j1` over the support if `c_j=1`,
  otherwise 0.

The contributions are summed over all overlapping windows for each rotation.
The pointwise median over 18 rotations produces the two record-level curves.
The peak of the median hard-count curve must be at least 100 for a detection.
For a detected record, `tL` is the median time of all samples on the global-
maximum plateau of the median accumulated-softmax curve.

