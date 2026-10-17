# Compact paper-result tables

These CSV files contain the record-level outputs associated with the final
Model 6.3 no-time ensemble and the paper's canonical AIC rule:

- `gm_test_predictions.csv`: 16 GM test records. Where both `tL1` and `tL2`
  were available, the primary timing comparison uses the annotation nearest
  the prediction and calls the signed difference the nearest-annotation
  residual. Both annotations are preserved.
- `centrifuge_test_predictions.csv`: 40 centrifuge records. One record,
  `KyU1_Motion2`, was reported but excluded from timing-error metrics because
  its liquefaction state/reference timing was ambiguous. The paper's principal
  detection statement is 36 detections among 38 surely liquefied tests; the
  surely non-liquefied `KyU1_Motion1` record produced no detection.
- `production_set_predictions.csv`: 25 unannotated or site-condition-screened
  production records. These are model applications, not an independent test
  set. The file was regenerated directly from the saved ensemble prediction
  JSONs using the canonical two-curve rule.

Only the hard-count AIC quantity required for the detection decision is
retained. Discarded peak-shape, symmetry, triangle-fit, calibration, and
confidence/uncertainty quantities are intentionally omitted.

