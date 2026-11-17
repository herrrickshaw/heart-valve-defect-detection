"""Clinician sign-off workflow (synthetic identifiers only)."""
import json

import pytest

from clinical_review_system import ClinicalReviewWorkflow, RegionalFeatureGate, ReviewStatus


def _wf():
    wf = ClinicalReviewWorkflow(patient_id="SYN-PATIENT", study_id="SYN-STUDY")
    wf.add_ai_finding("F1", {"finding_type": "severe_aortic_stenosis", "severity": "severe"})
    wf.add_ai_finding("F2", {"finding_type": "low_confidence_segmentation", "severity": "moderate"})
    return wf


def test_finalize_refuses_while_any_finding_is_pending():
    wf = _wf()
    wf.clinician_review("F1", "C1", "Dr Synth", ReviewStatus.CLINICIAN_APPROVED)
    with pytest.raises(ValueError, match="1 unreviewed"):
        wf.finalize_study()
    assert [p["finding_id"] for p in wf.get_pending_reviews()] == ["F2"]


def test_only_signed_off_findings_enter_the_record():
    wf = _wf()
    wf.clinician_review("F1", "C1", "Dr Synth", ReviewStatus.CLINICIAN_APPROVED)
    wf.clinician_review("F2", "C1", "Dr Synth", ReviewStatus.CLINICIAN_REJECTED)
    rec = wf.finalize_study()
    assert list(rec["approved_findings"]) == ["F1"]
    assert rec["audit_trail"][-1]["details"]["rejected_findings"] == 1


def test_clinician_modified_finding_enters_record_with_modified_severity():
    """A finding the clinician signed off with a changed severity was silently dropped from the record."""
    wf = _wf()
    wf.clinician_review("F1", "C1", "Dr Synth", ReviewStatus.CLINICIAN_MODIFIED, modified_severity="moderate")
    wf.clinician_review("F2", "C1", "Dr Synth", ReviewStatus.CLINICIAN_REJECTED)
    rec = wf.finalize_study()
    assert "F1" in rec["approved_findings"]
    assert rec["approved_findings"]["F1"]["clinician_modified_severity"] == "moderate"


def test_review_of_unknown_finding_rejected():
    with pytest.raises(ValueError):
        _wf().clinician_review("NOPE", "C1", "Dr Synth", ReviewStatus.CLINICIAN_APPROVED)


def test_audit_trail_export_is_valid_json(tmp_path):
    wf = _wf()
    wf.clinician_review("F1", "C1", "Dr Synth", ReviewStatus.CLINICIAN_APPROVED, confidence_in_ai=0.7)
    out = tmp_path / "audit.json"
    wf.export_audit_trail(str(out))
    data = json.loads(out.read_text())
    assert [e["event_type"] for e in data["audit_trail"]] == ["finding_generated", "finding_generated",
                                                              "review_completed"]
    assert data["reviews"]["F1"]["status"] == "clinician_approved"


@pytest.mark.parametrize("region,enabled", [("US", True), ("EU", False), ("ASIA", False), ("MARS", False)])
def test_regional_gate(region, enabled):
    assert RegionalFeatureGate.is_feature_enabled("aortic_stenosis_classifier", region)[0] is enabled
    assert RegionalFeatureGate.is_feature_enabled("unknown_feature", "US")[0] is False
