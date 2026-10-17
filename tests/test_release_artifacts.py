from __future__ import annotations

import json
from pathlib import Path

import h5py
import joblib
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPOSITORY_ROOT / "models" / "model6_3_no_time"


def test_all_five_model_scaler_pairs_are_present() -> None:
    for fold in range(1, 6):
        model_path = MODEL_ROOT / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        scaler_path = (
            MODEL_ROOT / "preprocessing" / f"fold_{fold}" / "scaler_single.joblib"
        )
        metadata_path = (
            MODEL_ROOT
            / "preprocessing"
            / f"fold_{fold}"
            / "preprocessing_metadata.json"
        )
        assert model_path.is_file()
        assert scaler_path.is_file()
        assert metadata_path.is_file()


def test_models_are_serialized_hdf5_keras_models() -> None:
    for fold in range(1, 6):
        model_path = MODEL_ROOT / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        assert h5py.is_hdf5(model_path)
        with h5py.File(model_path, "r") as stream:
            config = json.loads(stream.attrs["model_config"])
            assert config["class_name"] == "Functional"


def test_scalar_scalers_have_expected_feature_order_and_shape() -> None:
    for fold in range(1, 6):
        scaler_path = (
            MODEL_ROOT / "preprocessing" / f"fold_{fold}" / "scaler_single.joblib"
        )
        scaler = joblib.load(scaler_path)
        assert tuple(scaler.mean_.shape) == (2,)
        assert tuple(scaler.scale_.shape) == (2,)
        assert np.all(np.isfinite(scaler.mean_))
        assert np.all(scaler.scale_ > 0)

