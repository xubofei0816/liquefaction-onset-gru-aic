from __future__ import annotations

from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_record_manifest_matches_paper_counts() -> None:
    manifest = pd.read_csv(REPOSITORY_ROOT / "metadata" / "record_manifest.csv")
    assert len(manifest) == 111
    assert manifest["recording_id"].is_unique
    counts = manifest["study_use"].value_counts().to_dict()
    assert counts == {
        "centrifuge_test": 40,
        "training_and_cross_validation": 30,
        "gm_production": 25,
        "gm_test": 16,
    }


def test_reference_catalog_matches_manifest() -> None:
    manifest = pd.read_csv(REPOSITORY_ROOT / "metadata" / "record_manifest.csv")
    labels = pd.read_csv(REPOSITORY_ROOT / "metadata" / "reference_labels.csv")
    assert labels["recording_id"].is_unique
    assert set(labels["recording_id"]) == set(manifest["recording_id"])


def test_paper_result_tables_use_canonical_detection_fields_only() -> None:
    manifest_ids = set(
        pd.read_csv(REPOSITORY_ROOT / "metadata" / "record_manifest.csv")[
            "recording_id"
        ]
    )
    expected_rows = {
        "gm_test_predictions.csv": 16,
        "centrifuge_test_predictions.csv": 40,
        "production_set_predictions.csv": 25,
    }
    forbidden_fragments = (
        "confidence",
        "symmetry",
        "triangle",
        "sharpness",
        "pwhm",
        "plateau",
    )

    for filename, row_count in expected_rows.items():
        frame = pd.read_csv(REPOSITORY_ROOT / "paper_results" / filename)
        assert len(frame) == row_count
        assert frame["recording_id"].is_unique
        assert set(frame["recording_id"]).issubset(manifest_ids)
        assert (frame["hard_count_detection_threshold"] == 100.0).all()
        missed = ~frame["triggering_detected"].astype(bool)
        assert frame.loc[missed, "predicted_tL_sec"].isna().all()
        lowered = [column.lower() for column in frame.columns]
        assert not any(
            fragment in column
            for fragment in forbidden_fragments
            for column in lowered
        )

