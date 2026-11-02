"""Characterisation tests for DICOM ingestion (pydicom absent -> simulated anonymisation)."""
from __future__ import annotations

import sys

import pytest

from dicom_integration import DICOMAnonymizer, DICOMDataIngestionPipeline, DICOMViewType


@pytest.fixture(autouse=True)
def _no_pydicom(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydicom", None)  # force the ImportError branch


@pytest.mark.parametrize("desc,view", [
    ("Apical 4ch", DICOMViewType.APICAL_4_CHAMBER),
    ("PLAX view", DICOMViewType.PARASTERNAL_LONG_AXIS),
    ("parasternal short", DICOMViewType.PARASTERNAL_LONG_AXIS),  # 'parasternal' matches PLAX first
    ("zzz", DICOMViewType.OTHER),
])
def test_infer_view_type(desc, view):
    ds = type("DS", (), {"SeriesDescription": desc})()
    assert DICOMAnonymizer()._infer_view_type(ds) is view


def test_simulated_anonymisation_report():
    report = DICOMAnonymizer().anonymize_dicom_file("in.dcm", "out.dcm")
    assert report["status"] == "simulated"
    assert report["phi_removed"]["PATIENT_NAME"] == "REDACTED"


def test_ingest_missing_directory(tmp_path):
    pipe = DICOMDataIngestionPipeline(tmp_path / "out")
    assert pipe.ingest_from_hospital_export(tmp_path / "nope", "H")["status"] == "error"


def test_ingest_counts_and_report(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    for name in ("a.dcm", "b.dcm"):
        (export / name).write_bytes(b"synthetic")
    pipe = DICOMDataIngestionPipeline(tmp_path / "out")
    result = pipe.ingest_from_hospital_export(export, "H")
    # the glob pattern visits top-level files twice (existing behaviour)
    assert result["status"] == "complete" and result["total_files"] == 4
    assert result["successful"] == 4 and result["failed"] == 0
    pipe.register_validation_measurements("SERIES_00000", {"mv_mm": 28.5})
    rep = pipe.generate_ingestion_report()
    assert rep["total_series_ingested"] == 4
    assert rep["validation_measurements_registered"] == 1
    assert rep["series_summary"]["by_view"] == {"a4c": 4}
    assert rep["pixel_spacing_variance"]["x_spacing_mm"]["mean"] == pytest.approx(0.30)
