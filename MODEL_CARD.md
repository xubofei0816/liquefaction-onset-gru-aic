# Model card: Model 6.3 no-time

## Intended use

The model estimates whether liquefaction triggering is detected in a
three-component surface ground-motion record and, when detected, estimates the
triggering time `tL`. It is intended as a research aid for geotechnical and
earthquake engineering. It is not a substitute for expert review, site
characterization, pore-pressure measurements, or an independent triggering
analysis.

## Inputs

- A uniformly sampled or linearly resampleable surface record with two
  orthogonal horizontal components and one vertical component, in `g`.
- Event moment magnitude `Mw`.
- Eighteen rotation angles are generated internally: 0, 5, ..., 85 degrees.

The final model has six time-series channels and does **not** use an explicit
time-coordinate input:

1. horizontal component 1;
2. horizontal component 2;
3. vertical component;
4. centered 12-sample rolling standard deviation of component 1;
5. centered 12-sample rolling standard deviation of component 2; and
6. centered 12-sample rolling standard deviation of the vertical component.

The two scalar features are `[rotation angle, Mw]`. Each fold uses the scaler
fit only to that fold's training partition.

## Architecture and ensemble

Each fold model has two inputs. The time-series branch contains GRU layers
with 128 and 64 units, followed by the attention mechanism serialized in the
checkpoint. The attended sequence representation is concatenated with the two
scaled scalar features and passed through a 128-unit dense layer and a
three-class softmax output. The public prediction is the arithmetic mean of
the five fold softmax vectors for each window and azimuthal variant.

Class definitions during training were:

- Class 0: the full 5 s window occurs before `tL`;
- Class 1: the 5 s window contains `tL`; and
- Class 2: the full 5 s window occurs after `tL`.

## Training summary

- Five event-grouped folds.
- Adam with Keras 2.10 default parameters and learning rate 0.001.
- Batch size 32; maximum 20 epochs; early-stopping patience 5.
- Oversampling with replacement raised each minority class to at least half
  of the majority-class population, followed by balanced class weights.
- Four folds stopped after six epochs; one fold completed 20 epochs.

The individual checkpoints are legacy Keras HDF5 models stored with the
historical `.keras` filenames. Load them with
`tf.keras.models.load_model(path, compile=False)` in the pinned environment.

## Record-level decision rule

The hard-count Class-1 AIC is used for detection, with a threshold of 100. The
accumulated Class-1 softmax probability curve is used for timing. Both curves
are formed separately for the 18 rotations and then reduced by their
pointwise median. No peak magnitude, symmetry, curve width, or triangle-fit
quantity should be interpreted as a confidence or uncertainty estimate.

## Evaluation context

In the article, the model detected triggering in 15 of 16 GM test records and
36 of 38 surely liquefied centrifuge tests. Timing MAE was 1.49 s for detected
GM records under the nearest-annotation comparison and 2.53 s for detected
centrifuge tests. See `paper_results/` for the compact record-level outputs
and the article for definitions and limitations.

## Important limitations

- The natural-GM training set contains liquefied records and lacks a robustly
  annotated non-liquefied natural-GM class.
- External-domain behavior was weaker on centrifuge records.
- The centrifuge evaluation used an externally assigned `Mw=6.9`; magnitude is
  not a measured property of centrifuge motions.
- Timing near the first or last 5 s of a record is less reliable because the
  full set of Class-1-containing windows cannot exist at the boundary.
- Raw records from personal communications are not included in this release.

