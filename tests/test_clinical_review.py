"""Characterisation tests for the clinician review workflow (synthetic data only)."""
from __future__ import annotations

import json

import pytest

from clinical_review_system import (
    ClinicalDecisionSupportReport,
    ClinicalReviewWorkflow,
    RegionalFeatureGate,
    ReviewStatus,
)

FINDING = {"finding_type": "severe_aortic_stenosis", "severity": "severe",
           "measurement_value": 0.85, "reference_threshold": 1.0}


def _workflow():
    wf = ClinicalReviewWorkflow("P", "S")
    wf.add_ai_finding("F0", dict(FINDING))
    wf.add_ai_finding("F1", {"finding_type": "low_confidence_segmentation", "severity": "moderate"})
    return wf


def test_findings_start_pending_and_block_finalisation():
    wf = _workflow()
    assert [f["finding_id"] for f in wf.get_pending_reviews()] == ["F0", "F1"]
    with pytest.raises(ValueError, match="2 unreviewed findings"):
        wf.finalize_study()


def test_only_approved_findings_are_finalised():
    wf = _workflow()
    wf.clinician_review("F0", "D1", "Dr A", ReviewStatus.CLINICIAN_APPROVED, modified_severity="mild")
    wf.clinician_review("F1", "D1", "Dr A", ReviewStatus.CLINICIAN_REJECTED)
    final = wf.finalize_study()
    assert list(final["approved_findings"]) == ["F0"]
    assert final["approved_findings"]["F0"]["clinician_modified_severity"] == "mild"
    last = final["audit_trail"][-1]
    assert last["event_type"] == "study_finalized"
    assert last["details"] == {"total_findings": 2, "approved_findings": 1, "rejected_findings": 1}


def test_review_of_unknown_finding_raises():
    with pytest.raises(ValueError, match="not found"):
        _workflow().clinician_review("nope", "D", "Dr", ReviewStatus.CLINICIAN_APPROVED)


def test_audit_trail_export_roundtrip(tmp_path):
    wf = _workflow()
    wf.clinician_review("F0", "D1", "Dr A", ReviewStatus.CLINICIAN_APPROVED)
    out = tmp_path / "audit.json"
    wf.export_audit_trail(str(out))
    data = json.loads(out.read_text())
    assert data["study_id"] == "S"
    assert data["reviews"]["F0"]["status"] == "clinician_approved"
    assert [e["event_type"] for e in data["audit_trail"]] == [
        "finding_generated", "finding_generated", "review_completed"]


def test_report_mentions_precedence_rule_and_findings():
    text = ClinicalDecisionSupportReport.generate_review_report(
        "P", "S", [FINDING], {"view_type": "a4c", "frame_rate": 60})
    assert "CLINICAL PRECEDENCE RULE" in text
    assert "1. SEVERE_AORTIC_STENOSIS" in text
    assert "Frame Rate: 60 fps" in text
    assert "No findings flagged." not in text
    empty = ClinicalDecisionSupportReport.generate_review_report("P", "S", [], {})
    assert "No findings flagged." in empty


@pytest.mark.parametrize("region,enabled", [("US", True), ("EU", False), ("ASIA", False),
                                            ("GLOBAL", False), ("MARS", False)])
def test_regional_gate(region, enabled):
    mode = RegionalFeatureGate.get_deployment_mode(region)
    assert mode["as_classifier_enabled"] is enabled
    expected = ("segmentation_measurement_and_decision_support" if enabled
                else "segmentation_measurement_only")
    assert mode["deployment_mode"] == expected


def test_unknown_feature_is_disabled():
    assert RegionalFeatureGate.is_feature_enabled("x", "US") == (False, "Unknown feature: x")
