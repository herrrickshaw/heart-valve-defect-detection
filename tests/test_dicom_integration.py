"""DICOMAnonymizer on a small synthetic DICOM file (no real patient data)."""
import json

import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dicom_integration import DICOMAnonymizer, DICOMViewType

SYNTH_NAME = "Synthetica^Testa"
SYNTH_ID = "SYN-9999"


def _write(path, series_desc="Apical 4 chamber"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = SYNTH_NAME
    ds.PatientID = SYNTH_ID
    ds.PatientBirthDate = "19800101"
    ds.StudyDate = "20260301"
    ds.SeriesDescription = series_desc
    ds.InstitutionName = "Synthetic Hospital"
    ds.Modality = "US"
    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_phi_removed_from_output_file(tmp_path):
    out = tmp_path / "out.dcm"
    DICOMAnonymizer().anonymize_dicom_file(str(_write(tmp_path / "in.dcm")), str(out))
    ds = pydicom.dcmread(str(out))
    assert "PatientName" not in ds and "PatientID" not in ds and "InstitutionName" not in ds
    assert ds.StudyDate != "20260301"


def test_anonymisation_log_does_not_contain_the_phi_it_removed(tmp_path):
    """The report kept the original values under 'phi_removed' and export wrote them to disk."""
    anon = DICOMAnonymizer()
    report = anon.anonymize_dicom_file(str(_write(tmp_path / "in.dcm")), str(tmp_path / "out.dcm"))
    log = tmp_path / "log.json"
    anon.export_anonymization_log(str(log))
    text = log.read_text() + json.dumps(report, default=str)
    for phi in ("Synthetica", SYNTH_ID, "19800101", "Synthetic Hospital", "20260301"):
        assert phi not in text
    assert "PATIENT_NAME" in report["phi_removed"]          # still records WHICH fields were removed


@pytest.mark.parametrize("desc,view", [("Apical 4 chamber", DICOMViewType.APICAL_4_CHAMBER),
                                       ("PLAX", DICOMViewType.PARASTERNAL_LONG_AXIS),
                                       ("Parasternal short axis", DICOMViewType.PARASTERNAL_SHORT_AXIS)])
def test_view_label_inferred_from_original_series_description(tmp_path, desc, view):
    """SeriesDescription was deleted before the view was inferred, so every series became Echo_other."""
    out = tmp_path / "out.dcm"
    DICOMAnonymizer().anonymize_dicom_file(str(_write(tmp_path / "in.dcm", desc)), str(out))
    assert pydicom.dcmread(str(out)).SeriesDescription == f"Echo_{view.value}"
