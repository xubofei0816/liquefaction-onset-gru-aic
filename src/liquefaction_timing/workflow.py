"""Published Model 6.3 no-time preprocessing, inference, and AIC workflow."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DT_SECONDS = 0.005
SEGMENT_SECONDS = 5.0
SEGMENT_POINTS = int(SEGMENT_SECONDS / DT_SECONDS)
ROLLING_WINDOW_POINTS = 12
ROTATION_ANGLES_DEGREES = tuple(range(0, 90, 5))
HARD_COUNT_DETECTION_THRESHOLD = 100.0
TIME_SERIES_FEATURES = (
    "component_h1",
    "component_h2",
    "component_v",
    "component_h1_rolling_std_12",
    "component_h2_rolling_std_12",
    "component_v_rolling_std_12",
)
SCALAR_FEATURES = ("degrees_rotated_from_the_original", "Mw")
REQUIRED_MOTION_COLUMNS = ("time_s", "h1_g", "h2_g", "v_g")
REQUIRED_MANIFEST_COLUMNS = ("recording_id", "csv_path", "Mw")


@dataclass(frozen=True)
class MotionRecord:
    recording_id: str
    time_s: np.ndarray
    h1_g: np.ndarray
    h2_g: np.ndarray
    v_g: np.ndarray
    mw: float
    partition: str = "inference"
    reference_tl_sec: float | None = None


@dataclass(frozen=True)
class AICResult:
    count_curves: np.ndarray
    probability_curves: np.ndarray
    median_count: np.ndarray
    median_probability: np.ndarray
    time_s: np.ndarray
    hard_count_peak: float
    probability_peak: float
    detected: bool
    predicted_tl_sec: float | None


def _finite_float(value: Any, field_name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric; received {value!r}.") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{field_name} must be finite; received {value!r}.")
    return converted


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return _finite_float(value, field_name)


def load_motion_csv(
    path: str | Path,
    recording_id: str,
    mw: float,
    partition: str = "inference",
    reference_tl_sec: float | None = None,
) -> MotionRecord:
    """Load a three-component record and linearly resample it to 200 Hz.

    Input acceleration must already be expressed in units of g. The returned
    time axis starts at zero. For a uniformly sampled record whose interval is
    not 0.005 s, the resampling endpoint follows the historical preprocessing:
    ``np.arange(0, n_samples * original_dt, 0.005)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Motion CSV was not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_MOTION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {missing}")
    if len(frame) < 2:
        raise ValueError(f"{path}: at least two time samples are required.")

    arrays: dict[str, np.ndarray] = {}
    for column in REQUIRED_MOTION_COLUMNS:
        arrays[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.all(np.isfinite(arrays[column])):
            raise ValueError(f"{path}: non-finite value(s) in {column!r}.")

    source_time = arrays["time_s"] - arrays["time_s"][0]
    time_steps = np.diff(source_time)
    if np.any(time_steps <= 0):
        raise ValueError(f"{path}: time_s must be strictly increasing.")

    source_dt = float(np.median(time_steps))
    is_uniform = np.allclose(time_steps, source_dt, rtol=1e-5, atol=1e-9)
    if not is_uniform:
        raise ValueError(
            f"{path}: time_s is not uniformly sampled. Resample to a uniform "
            "time step before using this workflow."
        )

    if np.isclose(source_dt, DT_SECONDS, rtol=0.0, atol=1e-9):
        target_time = np.arange(len(frame), dtype=np.float64) * DT_SECONDS
        h1 = arrays["h1_g"]
        h2 = arrays["h2_g"]
        vertical = arrays["v_g"]
    else:
        target_time = np.arange(
            0.0,
            len(frame) * source_dt,
            DT_SECONDS,
            dtype=np.float64,
        )
        h1 = np.interp(target_time, source_time, arrays["h1_g"])
        h2 = np.interp(target_time, source_time, arrays["h2_g"])
        vertical = np.interp(target_time, source_time, arrays["v_g"])

    if len(target_time) < SEGMENT_POINTS:
        raise ValueError(
            f"{path}: {len(target_time)} samples remain after resampling; at least "
            f"{SEGMENT_POINTS} samples ({SEGMENT_SECONDS:g} s) are required."
        )

    return MotionRecord(
        recording_id=str(recording_id),
        time_s=target_time.astype(np.float32),
        h1_g=h1.astype(np.float32),
        h2_g=h2.astype(np.float32),
        v_g=vertical.astype(np.float32),
        mw=_finite_float(mw, "Mw"),
        partition=str(partition),
        reference_tl_sec=_optional_float(reference_tl_sec, "reference_tL_sec"),
    )


def load_manifest(manifest_path: str | Path) -> list[MotionRecord]:
    """Load all records listed in an inference manifest."""
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{manifest_path}: missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{manifest_path}: the manifest has no records.")
    if frame["recording_id"].astype(str).duplicated().any():
        duplicates = frame.loc[
            frame["recording_id"].astype(str).duplicated(keep=False), "recording_id"
        ].tolist()
        raise ValueError(f"{manifest_path}: duplicate recording_id values: {duplicates}")

    records: list[MotionRecord] = []
    for _, row in frame.iterrows():
        csv_path = Path(str(row["csv_path"]))
        if not csv_path.is_absolute():
            csv_path = manifest_path.parent / csv_path
        records.append(
            load_motion_csv(
                csv_path,
                recording_id=str(row["recording_id"]),
                mw=row["Mw"],
                partition=(
                    str(row["partition"])
                    if "partition" in frame and not pd.isna(row["partition"])
                    else "inference"
                ),
                reference_tl_sec=(
                    row["reference_tL_sec"]
                    if "reference_tL_sec" in frame
                    else None
                ),
            )
        )
    return records


def rotate_horizontal_components(
    h1: np.ndarray, h2: np.ndarray, angle_degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate two orthogonal horizontal components using the published rule."""
    angle_radians = np.deg2rad(angle_degrees)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    h1_rotated = cosine * h1 - sine * h2
    h2_rotated = sine * h1 + cosine * h2
    return h1_rotated.astype(np.float32), h2_rotated.astype(np.float32)


def centered_rolling_std(values: np.ndarray) -> np.ndarray:
    """Return the historical centered 12-sample pandas standard deviation."""
    return (
        pd.Series(np.asarray(values, dtype=np.float32))
        .rolling(ROLLING_WINDOW_POINTS, center=True, min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def build_feature_variations(record: MotionRecord) -> list[np.ndarray]:
    """Create the 18 [samples, 6] feature matrices used by the final model."""
    variations: list[np.ndarray] = []
    vertical_std = centered_rolling_std(record.v_g)
    for angle in ROTATION_ANGLES_DEGREES:
        h1_rotated, h2_rotated = rotate_horizontal_components(
            record.h1_g, record.h2_g, angle
        )
        matrix = np.column_stack(
            (
                h1_rotated,
                h2_rotated,
                record.v_g,
                centered_rolling_std(h1_rotated),
                centered_rolling_std(h2_rotated),
                vertical_std,
            )
        ).astype(np.float32, copy=False)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{record.recording_id}: non-finite feature(s) at {angle} degrees."
            )
        variations.append(matrix)
    return variations


def validate_artifacts(model_root: str | Path) -> None:
    """Confirm that all five model/scaler pairs are present."""
    model_root = Path(model_root)
    missing: list[Path] = []
    for fold in range(1, 6):
        model_path = model_root / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        scaler_path = (
            model_root / "preprocessing" / f"fold_{fold}" / "scaler_single.joblib"
        )
        for path in (model_path, scaler_path):
            if not path.is_file():
                missing.append(path)
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model artifact(s):\n{formatted}")


def _configure_tensorflow() -> tuple[Any, str]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model inference. Install the pinned "
            "environment in requirements.txt or environment.yml."
        ) from exc

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"TensorFlow GPU memory-growth warning: {exc}")
        device = "/GPU:0"
    else:
        device = "/CPU:0"
    return tf, device


def _validate_loaded_model(model: Any, fold: int) -> None:
    input_shapes = [tuple(shape) for shape in model.input_shape]
    expected = [(None, SEGMENT_POINTS, len(TIME_SERIES_FEATURES)), (None, 2)]
    if input_shapes != expected:
        raise ValueError(
            f"Fold {fold} model input shapes are {input_shapes}; expected {expected}."
        )
    output_shape = tuple(model.output_shape)
    if output_shape != (None, 3):
        raise ValueError(
            f"Fold {fold} model output shape is {output_shape}; expected (None, 3)."
        )


def _predict_variation_in_batches(
    model: Any,
    feature_matrix: np.ndarray,
    scaled_scalar_row: np.ndarray,
    batch_size: int,
    tf: Any,
    device: str,
    progress_prefix: str,
) -> np.ndarray:
    n_windows = feature_matrix.shape[0] - SEGMENT_POINTS + 1
    predictions = np.empty((n_windows, 3), dtype=np.float32)
    n_batches = int(np.ceil(n_windows / batch_size))

    for batch_number, batch_start in enumerate(
        range(0, n_windows, batch_size), start=1
    ):
        batch_end = min(batch_start + batch_size, n_windows)
        time_batch = np.stack(
            [
                feature_matrix[start : start + SEGMENT_POINTS]
                for start in range(batch_start, batch_end)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        scalar_batch = np.repeat(
            scaled_scalar_row, repeats=batch_end - batch_start, axis=0
        ).astype(np.float32, copy=False)
        with tf.device(device):
            batch_predictions = model(
                [time_batch, scalar_batch], training=False
            ).numpy()
        predictions[batch_start:batch_end] = batch_predictions.astype(
            np.float32, copy=False
        )
        if batch_number == 1 or batch_number % 25 == 0 or batch_number == n_batches:
            print(
                f"{progress_prefix} | batch {batch_number}/{n_batches} | "
                f"windows {batch_start:,}:{batch_end:,}/{n_windows:,}"
            )
        del time_batch, scalar_batch, batch_predictions
    return predictions


def predict_record_ensemble(
    record: MotionRecord,
    model_root: str | Path,
    batch_size: int = 256,
) -> np.ndarray:
    """Average softmax probabilities from the five published fold models."""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load the fold-specific scalar scalers."
        ) from exc
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    model_root = Path(model_root)
    validate_artifacts(model_root)
    tf, device = _configure_tensorflow()
    feature_variations = build_feature_variations(record)
    ensemble_sums = [
        np.zeros((len(record.time_s) - SEGMENT_POINTS + 1, 3), dtype=np.float32)
        for _ in ROTATION_ANGLES_DEGREES
    ]
    started = time.perf_counter()

    for fold in range(1, 6):
        model_path = model_root / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        scaler_path = (
            model_root / "preprocessing" / f"fold_{fold}" / "scaler_single.joblib"
        )
        print(
            f"[{record.partition}] {record.recording_id} | fold {fold}/5 | "
            f"loading model and scalar scaler"
        )
        tf.keras.backend.clear_session()
        gc.collect()
        scaler = joblib.load(scaler_path)
        if tuple(getattr(scaler, "mean_", np.empty(0)).shape) != (2,):
            raise ValueError(f"Fold {fold} scaler is not a two-feature scaler.")
        model = tf.keras.models.load_model(model_path, compile=False)
        _validate_loaded_model(model, fold)

        for variation_number, (angle, matrix) in enumerate(
            zip(ROTATION_ANGLES_DEGREES, feature_variations), start=1
        ):
            scalar_row = np.asarray([[angle, record.mw]], dtype=np.float32)
            scaled_scalar_row = scaler.transform(scalar_row).astype(
                np.float32, copy=False
            )
            prefix = (
                f"[{record.partition}] {record.recording_id} | fold {fold}/5 | "
                f"variation {variation_number}/18 ({angle} deg)"
            )
            ensemble_sums[variation_number - 1] += _predict_variation_in_batches(
                model,
                matrix,
                scaled_scalar_row,
                batch_size,
                tf,
                device,
                prefix,
            )

        del model, scaler
        tf.keras.backend.clear_session()
        gc.collect()
        print(
            f"[{record.partition}] {record.recording_id} | fold {fold}/5 complete | "
            f"elapsed {time.perf_counter() - started:.1f} s"
        )

    return np.stack([values / 5.0 for values in ensemble_sums]).astype(
        np.float32, copy=False
    )


def accumulated_aic_curves(
    all_variation_predictions: np.ndarray,
    segment_points: int = SEGMENT_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard-count and Class-1 accumulated-softmax curves.

    A window contributes its Class-1 probability only when Class 1 is the
    window's argmax class. This is the rule used for the paper results.
    """
    predictions = np.asarray(all_variation_predictions, dtype=np.float64)
    if predictions.ndim == 2:
        predictions = predictions[None, :, :]
    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError(
            "Predictions must have shape [variation, window, 3]; received "
            f"{predictions.shape}."
        )
    if predictions.shape[1] == 0 or not np.all(np.isfinite(predictions)):
        raise ValueError("Prediction array is empty or contains non-finite values.")
    if segment_points <= 0:
        raise ValueError("segment_points must be positive.")

    predicted_classes = np.argmax(predictions, axis=2)
    count_values = (predicted_classes == 1).astype(np.float64)
    probability_values = np.where(
        predicted_classes == 1, predictions[:, :, 1], 0.0
    )
    variation_count, window_count = count_values.shape
    full_length = window_count + segment_points - 1

    def box_accumulate(values: np.ndarray) -> np.ndarray:
        difference = np.zeros(
            (variation_count, full_length + 1), dtype=np.float64
        )
        difference[:, :window_count] += values
        difference[
            :, segment_points : segment_points + window_count
        ] -= values
        return np.cumsum(difference, axis=1)[:, :full_length]

    return box_accumulate(count_values), box_accumulate(probability_values)


def analyze_aic(
    all_variation_predictions: np.ndarray,
    dt_seconds: float = DT_SECONDS,
    threshold: float = HARD_COUNT_DETECTION_THRESHOLD,
) -> AICResult:
    """Apply the paper's detection and timing rules to ensemble predictions."""
    count_curves, probability_curves = accumulated_aic_curves(
        all_variation_predictions
    )
    median_count = np.median(count_curves, axis=0)
    median_probability = np.median(probability_curves, axis=0)
    hard_count_peak = float(np.max(median_count))
    probability_peak = float(np.max(median_probability))
    detected = bool(hard_count_peak >= threshold)
    predicted_tl_sec: float | None = None

    if detected:
        maximum_indices = np.flatnonzero(
            np.isclose(
                median_probability,
                probability_peak,
                rtol=0.0,
                atol=1e-10,
            )
        )
        if not maximum_indices.size:
            raise ValueError("Could not locate the accumulated-softmax maximum.")
        predicted_tl_sec = float(np.median(maximum_indices) * dt_seconds)

    time_s = np.arange(median_count.size, dtype=np.float64) * dt_seconds
    return AICResult(
        count_curves=count_curves,
        probability_curves=probability_curves,
        median_count=median_count,
        median_probability=median_probability,
        time_s=time_s,
        hard_count_peak=hard_count_peak,
        probability_peak=probability_peak,
        detected=detected,
        predicted_tl_sec=predicted_tl_sec,
    )


def plot_inference_result(
    record: MotionRecord,
    aic: AICResult,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the AIC result and three input components in the paper's style."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1.08, 1.0, 1.0, 1.0]},
    )
    figure.suptitle(f"Recording: {record.recording_id}", fontsize=17, y=0.995)

    aic_axis = axes[0]
    for curve in aic.count_curves:
        aic_axis.plot(aic.time_s, curve, color="tab:blue", alpha=0.15, lw=0.8)
    aic_axis.plot(
        aic.time_s,
        aic.median_count,
        color="tab:blue",
        lw=2.2,
        label="Median AIC (count)",
    )
    aic_axis.plot(
        aic.time_s,
        aic.median_probability,
        color="tab:orange",
        lw=2.2,
        label="Median accumulated softmax (prob)",
    )
    aic_axis.axhline(
        HARD_COUNT_DETECTION_THRESHOLD,
        color="tab:red",
        ls=":",
        lw=1.7,
        label="AIC detection threshold = 100",
    )

    if record.reference_tl_sec is not None:
        for axis in axes:
            axis.axvline(
                record.reference_tl_sec,
                color="firebrick",
                ls="-.",
                lw=1.6,
                label=("Reference $t_L$" if axis is aic_axis else None),
            )
    if aic.predicted_tl_sec is not None:
        for axis in axes:
            axis.axvline(
                aic.predicted_tl_sec,
                color="black",
                ls="--",
                lw=1.6,
                label=("Predicted $t_L$" if axis is aic_axis else None),
            )

    aic_axis.set_title("Accumulated Predictions", fontsize=16)
    aic_axis.set_ylabel("Accumulated value")
    aic_axis.legend(loc="upper right", fontsize=9)
    status_text = (
        f"Predicted $t_L$: {aic.predicted_tl_sec:.3f} s"
        if aic.predicted_tl_sec is not None
        else "No detection"
    )
    aic_axis.text(
        0.985,
        0.07,
        status_text,
        transform=aic_axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    components = (
        (record.h1_g, "Horizontal Component 1", "H1", "tab:blue"),
        (record.h2_g, "Horizontal Component 2", "H2", "tab:green"),
        (record.v_g, "Vertical Component", "Vertical", "tab:red"),
    )
    for axis, (values, title, label, color) in zip(axes[1:], components):
        axis.plot(record.time_s, values, color=color, lw=0.8, label=label)
        axis.set_title(title, fontsize=15)
        axis.set_ylabel("Acceleration (g)")
        axis.legend(loc="upper right", fontsize=9)

    for label, axis in zip(("a)", "b)", "c)", "d)"), axes):
        axis.text(
            0.012,
            0.88,
            label,
            transform=axis.transAxes,
            fontsize=20,
            va="top",
        )
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(record.time_s[0], record.time_s[-1])
    figure.tight_layout(rect=(0, 0, 1, 0.982))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        figure.savefig(temporary, dpi=300, bbox_inches="tight", format="png")
        os.replace(temporary, output_path)
    return figure


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_record_outputs(
    record: MotionRecord,
    predictions: np.ndarray,
    aic: AICResult,
    output_root: str | Path,
) -> dict[str, Any]:
    """Atomically save raw probabilities, AIC curves, plot, and summary."""
    import matplotlib.pyplot as plt

    output_directory = Path(output_root) / record.recording_id
    output_directory.mkdir(parents=True, exist_ok=True)

    prediction_path = output_directory / "ensemble_predictions.npz"
    temporary_prediction = prediction_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_prediction,
        all_variations=predictions.astype(np.float32, copy=False),
        angles_degrees=np.asarray(ROTATION_ANGLES_DEGREES, dtype=np.int16),
    )
    os.replace(temporary_prediction, prediction_path)

    aic_path = output_directory / "aic_curves.csv"
    _atomic_csv(
        aic_path,
        pd.DataFrame(
            {
                "time_s": aic.time_s,
                "median_hard_count_class1_aic": aic.median_count,
                "median_accumulated_class1_softmax": aic.median_probability,
            }
        ),
    )

    plot_path = output_directory / "inference_plot.png"
    figure = plot_inference_result(record, aic, plot_path)
    plt.close(figure)

    summary = {
        "recording_id": record.recording_id,
        "partition": record.partition,
        "Mw": record.mw,
        "reference_tL_sec": record.reference_tl_sec,
        "sample_interval_sec": DT_SECONDS,
        "sample_count": int(len(record.time_s)),
        "window_count": int(predictions.shape[1]),
        "variation_count": int(predictions.shape[0]),
        "fold_count": 5,
        "peak_median_hard_count_class1_aic": aic.hard_count_peak,
        "hard_count_detection_threshold": HARD_COUNT_DETECTION_THRESHOLD,
        "liquefaction_triggering_detected": aic.detected,
        "predicted_tL_sec": aic.predicted_tl_sec,
        "detection_curve": "median hard-count Class-1 AIC",
        "timing_curve": "median accumulated Class-1 softmax probability",
        "timing_rule": "median time of the global-maximum plateau",
        "files": {
            "predictions": prediction_path.name,
            "aic_curves": aic_path.name,
            "plot": plot_path.name,
        },
    }
    _atomic_json(output_directory / "summary.json", summary)
    return summary


def run_manifest(
    manifest_path: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    batch_size: int = 256,
    display_plots: bool = True,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Run every manifest record and save results after each record."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
            return iterable

    records = load_manifest(manifest_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    for record_number, record in enumerate(
        tqdm(records, desc="Records", unit="record"), start=1
    ):
        summary_path = output_root / record.recording_id / "summary.json"
        if skip_existing and summary_path.is_file():
            print(
                f"[{record.partition}] record {record_number}/{len(records)} | "
                f"{record.recording_id} | existing outputs retained"
            )
            with summary_path.open("r", encoding="utf-8") as stream:
                summaries.append(json.load(stream))
            continue

        record_start = time.perf_counter()
        print(
            f"[{record.partition}] record {record_number}/{len(records)} | "
            f"{record.recording_id} | starting"
        )
        predictions = predict_record_ensemble(
            record, model_root=model_root, batch_size=batch_size
        )
        aic = analyze_aic(predictions)
        summary = save_record_outputs(record, predictions, aic, output_root)
        summaries.append(summary)

        combined = pd.DataFrame(summaries)
        _atomic_csv(output_root / "inference_summary.csv", combined)
        print(
            f"[{record.partition}] record {record_number}/{len(records)} | "
            f"{record.recording_id} complete | detected={aic.detected} | "
            f"predicted_tL={aic.predicted_tl_sec} | "
            f"record elapsed {time.perf_counter() - record_start:.1f} s | "
            f"total elapsed {time.perf_counter() - overall_start:.1f} s"
        )
        if display_plots:
            try:
                from IPython.display import Image, display

                display(Image(filename=str(output_root / record.recording_id / "inference_plot.png")))
            except ImportError:
                print("IPython is unavailable; inspect the saved inference_plot.png.")

        del predictions, aic
        gc.collect()

    result = pd.DataFrame(summaries)
    _atomic_csv(output_root / "inference_summary.csv", result)
    return result


def public_configuration() -> dict[str, Any]:
    """Return the immutable preprocessing and decision settings."""
    return {
        "dt_seconds": DT_SECONDS,
        "segment_seconds": SEGMENT_SECONDS,
        "segment_points": SEGMENT_POINTS,
        "rolling_window_points": ROLLING_WINDOW_POINTS,
        "rotation_angles_degrees": list(ROTATION_ANGLES_DEGREES),
        "time_series_features": list(TIME_SERIES_FEATURES),
        "scalar_features": list(SCALAR_FEATURES),
        "hard_count_detection_threshold": HARD_COUNT_DETECTION_THRESHOLD,
        "fold_count": 5,
    }

