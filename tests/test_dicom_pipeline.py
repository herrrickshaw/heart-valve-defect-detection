"""DICOM view inference, anonymiser edge paths and the ingestion pipeline (synthetic files only)."""
import json
import logging
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

import dicom_integration as di
from dicom_integration import (
    DICOMAnonymizer,
    DICOMDataIngestionPipeline,
    DICOMSeries,
    DICOMViewType,
)

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402


def write_dcm(path, series_desc="Apical 4 chamber", **extra):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Synthetica^Testa"
    ds.PatientID = "SYN-1"
    ds.Modality = "US"
    if series_desc is not None:
        ds.SeriesDescription = series_desc
    for k, v in extra.items():
        setattr(ds, k, v)
    ds.save_as(str(path), enforce_file_format=True)
    return path


class TestViewInference:
    @pytest.mark.parametrize("desc,view", [
        ("Parasternal SHORT axis", DICOMViewType.PARASTERNAL_SHORT_AXIS),
        ("psax mid", DICOMViewType.PARASTERNAL_SHORT_AXIS),
        ("SAX", DICOMViewType.PARASTERNAL_SHORT_AXIS),
        ("Parasternal long axis", DICOMViewType.PARASTERNAL_LONG_AXIS),
        ("PLAX", DICOMViewType.PARASTERNAL_LONG_AXIS),
        ("lax view", DICOMViewType.PARASTERNAL_LONG_AXIS),
        ("A4C", DICOMViewType.APICAL_4_CHAMBER),
        ("Apical 4 chamber", DICOMViewType.APICAL_4_CHAMBER),
        ("four chamber", DICOMViewType.APICAL_4_CHAMBER),
        ("4 chamber", DICOMViewType.APICAL_4_CHAMBER),
        ("4-Chamber", DICOMViewType.APICAL_4_CHAMBER),
        ("something else", DICOMViewType.OTHER),
        ("Parasternal", DICOMViewType.OTHER),       # shared word alone is not a view
        ("series 4", DICOMViewType.OTHER),          # a bare digit must not match
        ("", DICOMViewType.OTHER),
    ])
    def test_series_description_keywords(self, desc, view):
        ds = SimpleNamespace(SeriesDescription=desc)
        assert DICOMAnonymizer()._infer_view_type(ds) is view

    def test_missing_series_description_is_other(self):
        assert DICOMAnonymizer()._infer_view_type(SimpleNamespace()) is DICOMViewType.OTHER


class TestAnonymizerPaths:
    def test_simulated_mode_when_pydicom_missing(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "pydicom", None)
        with caplog.at_level(logging.WARNING, logger="dicom_integration"):
            rep = DICOMAnonymizer().anonymize_dicom_file("in.dcm", "out.dcm")
        assert rep["status"] == "simulated" and rep["dicom_file"] == "in.dcm" and rep["output"] == "out.dcm"
        assert set(rep["phi_removed"]) == {"PATIENT_NAME", "PATIENT_ID", "ACCESSION_NUMBER", "INSTITUTION_NAME"}
        assert all(v == "REDACTED" for v in rep["phi_removed"].values())
        assert "pydicom not installed" in caplog.text

    def test_unreadable_file_reports_error(self, tmp_path):
        rep = DICOMAnonymizer().anonymize_dicom_file(str(tmp_path / "missing.dcm"), str(tmp_path / "o.dcm"))
        assert rep["status"] == "error" and rep["message"]
        assert not (tmp_path / "o.dcm").exists()

    def test_success_report_and_log_entry(self, tmp_path):
        anon = DICOMAnonymizer()
        src = write_dcm(tmp_path / "a.dcm", StudyDate="20260101", SeriesDate="20260102", PatientSex="F")
        rep = anon.anonymize_dicom_file(str(src), str(tmp_path / "out.dcm"))
        assert rep["status"] == "success" and rep["private_tags_removed"] is True
        assert rep["dicom_file"] == str(src) and rep["output"] == str(tmp_path / "out.dcm")
        assert rep["phi_removed"]["PATIENT_SEX"] == "<removed>"
        assert rep["phi_removed"]["STUDY_DATE"] == "<replaced>" and rep["phi_removed"]["SERIES_DATE"] == "<replaced>"
        assert "ACCESSION_NUMBER" not in rep["phi_removed"]       # only fields actually present are reported
        datetime.fromisoformat(rep["timestamp"])
        assert anon.anonymization_log == [rep]
        ds = pydicom.dcmread(str(tmp_path / "out.dcm"))
        assert ds.StudyDate == "20250101" and ds.SeriesDate == "20250101"
        assert ds.StudyDescription == "Cardiac Ultrasound"
        assert "PatientSex" not in ds

    def test_dates_not_added_when_absent(self, tmp_path):
        rep = DICOMAnonymizer().anonymize_dicom_file(str(write_dcm(tmp_path / "a.dcm")), str(tmp_path / "o.dcm"))
        assert "STUDY_DATE" not in rep["phi_removed"] and "SERIES_DATE" not in rep["phi_removed"]
        ds = pydicom.dcmread(str(tmp_path / "o.dcm"))
        assert "StudyDate" not in ds and "SeriesDate" not in ds

    def test_keep_views_only_false_drops_descriptions(self, tmp_path):
        anon = DICOMAnonymizer(keep_views_only=False)
        anon.anonymize_dicom_file(str(write_dcm(tmp_path / "a.dcm")), str(tmp_path / "o.dcm"))
        ds = pydicom.dcmread(str(tmp_path / "o.dcm"))
        assert "SeriesDescription" not in ds and "StudyDescription" not in ds

    def test_private_tags_are_removed(self, tmp_path):
        src = tmp_path / "a.dcm"
        write_dcm(src)
        ds = pydicom.dcmread(str(src))
        block = ds.private_block(0x0009, "SYNTH CREATOR", create=True)
        block.add_new(0x01, "LO", "secret")
        ds.save_as(str(src))
        assert any(e.tag.is_private for e in pydicom.dcmread(str(src)))
        DICOMAnonymizer().anonymize_dicom_file(str(src), str(tmp_path / "o.dcm"))
        assert not any(e.tag.is_private for e in pydicom.dcmread(str(tmp_path / "o.dcm")))

    def test_export_log_round_trip(self, tmp_path):
        anon = DICOMAnonymizer()
        anon.anonymize_dicom_file(str(write_dcm(tmp_path / "a.dcm")), str(tmp_path / "o.dcm"))
        out = tmp_path / "log.json"
        anon.export_anonymization_log(str(out))
        data = json.loads(out.read_text())
        assert len(data) == 1 and data[0]["status"] == "success"

    def test_export_empty_log(self, tmp_path):
        out = tmp_path / "log.json"
        DICOMAnonymizer().export_anonymization_log(str(out))
        assert json.loads(out.read_text()) == []


class TestIngestionPipeline:
    def test_output_dir_created(self, tmp_path):
        target = tmp_path / "a" / "b"
        DICOMDataIngestionPipeline(target)
        assert target.is_dir()

    def test_missing_export_dir(self, tmp_path):
        p = DICOMDataIngestionPipeline(tmp_path / "out")
        assert p.ingest_from_hospital_export(tmp_path / "nope", "H") == {
            "status": "error", "message": "Export directory not found"}

    def test_ingest_nested_files_and_report(self, tmp_path):
        src = tmp_path / "src"
        (src / "sub").mkdir(parents=True)
        write_dcm(src / "sub" / "n1.dcm", "PLAX")
        p = DICOMDataIngestionPipeline(tmp_path / "out")
        rep = p.ingest_from_hospital_export(src, "HospA")
        assert rep["status"] == "complete" and rep["institution"] == "HospA"
        assert rep["total_files"] == 1 and rep["successful"] == 1 and rep["failed"] == 0
        assert rep["series_ingested"] == 1
        assert rep["details"] == [{"file": "n1.dcm", "status": "ingested", "view": "a4c"}]
        assert (tmp_path / "out" / "anon_n1.dcm").exists()
        s = p.ingested_series[0]
        assert s.series_id == "SERIES_00000" and s.study_id.startswith("STUDY_HospA_")

    def test_simulated_anonymisation_counts_as_success(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pydicom", None)          # demo mode: no pydicom available
        src = tmp_path / "src" / "d"
        src.mkdir(parents=True)
        (src / "x.dcm").write_bytes(b"placeholder")
        p = DICOMDataIngestionPipeline(tmp_path / "out")
        rep = p.ingest_from_hospital_export(tmp_path / "src", "H")
        assert rep["successful"] == 1 and rep["failed"] == 0 and rep["series_ingested"] == 1
        assert rep["details"] == [{"file": "x.dcm", "status": "ingested", "view": "a4c"}]

    def test_failed_anonymisation_is_counted_with_message(self, tmp_path):
        src = tmp_path / "src"
        (src / "d").mkdir(parents=True)
        (src / "d" / "bad.dcm").write_bytes(b"not a dicom file")
        p = DICOMDataIngestionPipeline(tmp_path / "out")
        rep = p.ingest_from_hospital_export(src, "H")
        assert rep["failed"] == 1 and rep["successful"] == 0 and rep["series_ingested"] == 0
        assert rep["details"][0]["status"] == "failed" and rep["details"][0]["file"] == "bad.dcm"
        assert rep["details"][0]["error"]

    @pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="dicom_integration.py: ingest_from_hospital_export lists `glob('*.dcm') + glob('**/*.dcm')`; "
               "'**' also matches the top directory, so every top-level file is processed twice",
    )
    def test_top_level_file_is_ingested_once(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        write_dcm(src / "top.dcm")
        rep = DICOMDataIngestionPipeline(tmp_path / "out").ingest_from_hospital_export(src, "H")
        assert rep["total_files"] == 1 and rep["series_ingested"] == 1

    def test_series_ids_increment(self, tmp_path):
        p = DICOMDataIngestionPipeline(tmp_path / "out")
        a = p._extract_series_metadata("x.dcm", "H")
        p.ingested_series.append(a)
        b = p._extract_series_metadata("y.dcm", "H")
        assert (a.series_id, b.series_id) == ("SERIES_00000", "SERIES_00001")
        assert (a.frame_rate, a.num_frames, a.image_quality_estimate) == (60, 30, "good")

    def test_register_validation_measurements(self, tmp_path):
        p = DICOMDataIngestionPipeline(tmp_path / "out")
        p.register_validation_measurements("S0", {"m": 1})
        e = p.validation_measurements[0]
        assert e["series_id"] == "S0" and e["measurements"] == {"m": 1}
        assert e["purpose"] == "validation_check_local_calibration"
        datetime.fromisoformat(e["registered_at"])


def _series(view, quality, sx, sy):
    return DICOMSeries("s", "st", view, 60, sx, sy, 10, "2025-01-01T00:00:00", quality)


class TestReports:
    def test_empty_report(self, tmp_path):
        rep = DICOMDataIngestionPipeline(tmp_path / "o").generate_ingestion_report()
        assert rep["total_series_ingested"] == 0 and rep["validation_measurements_registered"] == 0
        assert rep["series_summary"] == {"by_view": {}, "image_quality": {}}
        assert rep["pixel_spacing_variance"] == {}

    def test_counts_and_spacing_statistics(self, tmp_path):
        p = DICOMDataIngestionPipeline(tmp_path / "o")
        p.ingested_series = [
            _series(DICOMViewType.APICAL_4_CHAMBER, "good", 0.2, 0.1),
            _series(DICOMViewType.APICAL_4_CHAMBER, "poor", 0.4, 0.3),
            _series(DICOMViewType.PARASTERNAL_LONG_AXIS, "good", 0.3, 0.2),
        ]
        p.register_validation_measurements("s", {})
        rep = p.generate_ingestion_report()
        assert rep["total_series_ingested"] == 3 and rep["validation_measurements_registered"] == 1
        assert rep["series_summary"]["by_view"] == {"a4c": 2, "PLAX": 1}
        assert rep["series_summary"]["image_quality"] == {"good": 2, "poor": 1}
        sp = rep["pixel_spacing_variance"]
        assert sp["x_spacing_mm"] == {"min": 0.2, "max": 0.4, "mean": pytest.approx(0.3)}
        assert sp["y_spacing_mm"] == {"min": 0.1, "max": 0.3, "mean": pytest.approx(0.2)}
        assert "per-site" in sp["note"]


def test_example_ingestion_runs_offline(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(di, "Path", lambda p: tmp_path / "ingest")
    with caplog.at_level(logging.INFO, logger="dicom_integration"):
        di.example_dicom_ingestion()
    assert "Series ingested: 0" in caplog.text and "Validation measurements: 1" in caplog.text
