# Trained ensemble artifacts

This directory contains the five fold models used for the article and the
matching scalar scaler for each fold.

```text
checkpoints/fold_n/best_model.keras
preprocessing/fold_n/scaler_single.joblib
preprocessing/fold_n/preprocessing_metadata.json
```

Important details:

- Each model is a complete legacy Keras HDF5 model saved with a `.keras`
  filename under Keras 2.10.
- Load with `compile=False` in the pinned TensorFlow 2.10 environment.
- The first input shape is `(None, 1000, 6)`.
- The second input shape is `(None, 2)`.
- The scalar order is `[rotation angle in degrees, Mw]`.
- Never fit a replacement scaler during inference. Fold `n` must use
  `preprocessing/fold_n/scaler_single.joblib`.
- Public probabilities are the arithmetic mean of all five fold models.

