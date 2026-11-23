"""Boundary tests for the review workflow, report text and regional gate (synthetic identifiers only)."""
import json
from datetime import datetime

import pytest

import clinical_review_system as crs
from clinical_review_system import (
    AuditLog,
    ClinicalDecisionSupportReport,
    ClinicalReviewWorkflow,
    ClinicianReview,
    RegionalFeatureGate,
    ReviewStatus,
)


def _wf():
    wf = ClinicalReviewWorkflow(patient_id="SYN-P", study_id="SYN-S")
    wf.add_ai_finding("F1", {"finding_type": "type_a", "severity": "severe", "measurement_value": 0.5})
    wf.add_ai_finding("F2", {"finding_type": "type_b", "severity": "moderate"})
    wf.add_ai_finding("F3", {"finding_type": "type_c", "severity": "mild"})
    return wf


class TestDataclasses:
    def test_review_to_dict_serialises_timestamp_and_status(self):
        ts = datetime(2026, 3, 1, 12, 30, 5)
        r = ClinicianReview("F1", "C1", "Dr Synth", ts, ReviewStatus.CLINICIAN_MODIFIED,
                            comments="c", modified_severity="mild", confidence_in_ai=0.25)
        d = r.to_dict()
        assert d["review_timestamp"] == "2026-03-01T12:30:05" and d["status"] == "clinician_modified"
        assert d["comments"] == "c" and d["modified_severity"] == "mild" and d["confidence_in_ai"] == 0.25
        assert json.dumps(d)  # fully JSON-serialisable

    def test_review_defaults(self):
        r = ClinicianReview("F1", "C1", "n", datetime(2026, 1, 1), ReviewStatus.PENDING_REVIEW)
        assert r.comments == "" and r.modified_severity is None and r.confidence_in_ai == 0.0

    def test_audit_log_to_dict(self):
        a = AuditLog("evt", datetime(2026, 1, 2, 3, 4, 5), "system", {"k": 1})
        assert a.to_dict() == {"event_type": "evt", "timestamp": "2026-01-02T03:04:05",
                               "actor": "system", "details": {"k": 1}}

    def test_status_values(self):
        assert [s.value for s in ReviewStatus] == [
            "pending_review", "clinician_approved", "clinician_rejected", "clinician_modified", "archived"]


class TestWorkflow:
    def test_new_finding_is_pending_and_audited(self):
        wf = _wf()
        f = wf.findings["F1"]
        assert f["status"] == "pending_review" and f["measurement_value"] == 0.5
        assert "ai_generated_timestamp" in f
        first = wf.audit_trail[0]
        assert (first.event_type, first.actor) == ("finding_generated", "AI")
        assert first.details == {"finding_id": "F1", "finding_type": "type_a", "severity": "severe"}

    def test_audit_details_tolerate_missing_keys(self):
        wf = ClinicalReviewWorkflow("p", "s")
        wf.add_ai_finding("X", {})
        assert wf.audit_trail[0].details == {"finding_id": "X", "finding_type": None, "severity": None}

    def test_review_default_confidence_and_status_update(self):
        wf = _wf()
        r = wf.clinician_review("F1", "C1", "Dr Synth", ReviewStatus.CLINICIAN_APPROVED)
        assert r.confidence_in_ai == 0.5 and wf.reviews["F1"] is r
        assert wf.findings["F1"]["status"] == "clinician_approved"
        assert "clinician_modified_severity" not in wf.findings["F1"]
        ev = wf.audit_trail[-1]
        assert ev.event_type == "review_completed" and ev.actor == "Dr Synth"
        assert ev.details == {"finding_id": "F1", "review_status": "clinician_approved",
                              "clinician_confidence": 0.5, "modified": False}

    def test_empty_modified_severity_is_not_recorded(self):
        wf = _wf()
        wf.clinician_review("F1", "C1", "n", ReviewStatus.CLINICIAN_MODIFIED, modified_severity="")
        assert "clinician_modified_severity" not in wf.findings["F1"]
        assert wf.audit_trail[-1].details["modified"] is False

    def test_modified_severity_recorded(self):
        wf = _wf()
        wf.clinician_review("F1", "C1", "n", ReviewStatus.CLINICIAN_MODIFIED, modified_severity="mild")
        assert wf.findings["F1"]["clinician_modified_severity"] == "mild"
        assert wf.audit_trail[-1].details["modified"] is True

    def test_unknown_finding_message(self):
        with pytest.raises(ValueError, match="Finding NOPE not found"):
            _wf().clinician_review("NOPE", "C", "n", ReviewStatus.CLINICIAN_APPROVED)

    def test_re_review_overwrites_previous_decision(self):
        wf = _wf()
        wf.clinician_review("F1", "C1", "n", ReviewStatus.CLINICIAN_REJECTED)
        wf.clinician_review("F1", "C1", "n", ReviewStatus.CLINICIAN_APPROVED)
        assert wf.findings["F1"]["status"] == "clinician_approved"
        assert wf.reviews["F1"].status == ReviewStatus.CLINICIAN_APPROVED
        assert sum(e.event_type == "review_completed" for e in wf.audit_trail) == 2

    def test_pending_list_shrinks_as_reviews_arrive(self):
        wf = _wf()
        assert [p["finding_id"] for p in wf.get_pending_reviews()] == ["F1", "F2", "F3"]
        assert wf.get_pending_reviews()[0]["finding_type"] == "type_a"
        wf.clinician_review("F2", "C", "n", ReviewStatus.CLINICIAN_REJECTED)
        assert [p["finding_id"] for p in wf.get_pending_reviews()] == ["F1", "F3"]

    def test_finalize_error_names_pending_findings(self):
        wf = _wf()
        wf.clinician_review("F1", "C", "n", ReviewStatus.CLINICIAN_APPROVED)
        with pytest.raises(ValueError) as e:
            wf.finalize_study()
        msg = str(e.value)
        assert "SYN-S" in msg and "2 unreviewed" in msg and "['F2', 'F3']" in msg

    def test_finalize_counts_and_exclusions(self):
        wf = _wf()
        wf.clinician_review("F1", "C", "n", ReviewStatus.CLINICIAN_APPROVED)
        wf.clinician_review("F2", "C", "n", ReviewStatus.CLINICIAN_REJECTED)
        wf.clinician_review("F3", "C", "n", ReviewStatus.ARCHIVED)
        rec = wf.finalize_study()
        assert rec["patient_id"] == "SYN-P" and rec["study_id"] == "SYN-S"
        assert list(rec["approved_findings"]) == ["F1"]       # rejected and archived stay out
        fin = rec["audit_trail"][-1]
        assert fin["event_type"] == "study_finalized" and fin["actor"] == "system"
        assert fin["details"] == {"total_findings": 3, "approved_findings": 1, "rejected_findings": 1}
        datetime.fromisoformat(rec["finalized_timestamp"])
        assert wf.audit_trail[-1].event_type == "study_finalized"

    def test_finalize_empty_study(self):
        rec = ClinicalReviewWorkflow("p", "s").finalize_study()
        assert rec["approved_findings"] == {} and rec["audit_trail"][-1]["details"]["total_findings"] == 0

    def test_export_audit_trail_contents(self, tmp_path):
        wf = _wf()
        wf.clinician_review("F1", "C1", "Dr Synth", ReviewStatus.CLINICIAN_MODIFIED, modified_severity="mild")
        out = tmp_path / "a.json"
        wf.export_audit_trail(str(out))
        data = json.loads(out.read_text())
        assert data["patient_id"] == "SYN-P" and data["study_id"] == "SYN-S"
        datetime.fromisoformat(data["exported_at"])
        assert list(data["reviews"]) == ["F1"] and data["reviews"]["F1"]["modified_severity"] == "mild"
        assert len(data["audit_trail"]) == 4


class TestReport:
    def _lines(self, **kw):
        base = dict(patient_id="SYN-P", study_id="SYN-S", findings=[], metadata={})
        base.update(kw)
        return ClinicalDecisionSupportReport.generate_review_report(**base).splitlines()

    def test_header_and_precedence_rule(self):
        lines = self._lines()
        assert lines[0] == "=" * 70 and lines[-1] == "=" * 70
        assert lines[1] == "CARDIAC ULTRASOUND ANALYSIS REPORT"
        assert "Patient ID: SYN-P" in lines and "Study ID: SYN-S" in lines
        assert any(ln.startswith("Report Generated: ") for ln in lines)
        assert "IMPORTANT: CLINICAL PRECEDENCE RULE" in lines
        assert "No findings enter the patient record without clinician approval." in lines

    def test_no_findings_and_no_metadata(self):
        lines = self._lines()
        assert "No findings flagged." in lines
        assert "ACQUISITION METADATA:" not in lines and "FLAGGED FINDINGS REQUIRING REVIEW:" not in lines

    def test_metadata_defaults_to_na(self):
        lines = self._lines(metadata={"view_type": "a4c"})
        assert "View Type: a4c" in lines and "Cardiac Phase: N/A" in lines
        assert "Image Quality: N/A" in lines and "Frame Rate: N/A fps" in lines

    def test_findings_are_numbered_and_uppercased(self):
        lines = self._lines(findings=[
            {"finding_type": "alpha", "severity": "severe", "measurement_value": 0.8, "reference_threshold": 1.0},
            {},
        ])
        assert "1. ALPHA" in lines and "   Severity: severe" in lines
        assert "   Measurement: 0.8" in lines and "   Threshold: 1.0" in lines
        assert "2. UNKNOWN" in lines and "   Severity: N/A" in lines
        assert lines.count("   Status: PENDING CLINICIAN REVIEW") == 2
        assert "No findings flagged." not in lines

    def test_measurements_section_toggle(self):
        assert "QUANTITATIVE MEASUREMENTS:" in self._lines()
        assert "QUANTITATIVE MEASUREMENTS:" not in self._lines(include_measurements=False)

    def test_next_steps_always_last(self):
        lines = self._lines(include_measurements=False)
        assert lines[-7] == "NEXT STEPS:" and lines[-6] == "-" * 70
        assert lines[-5].startswith("1. ") and lines[-2].startswith("4. ")


class TestRegionalGate:
    @pytest.mark.parametrize("region,status", [("US", "FDA_CLEARED"), ("EU", "PENDING_CE_MARKING"),
                                               ("ASIA", "REGIONAL_APPROVAL_REQUIRED"),
                                               ("GLOBAL", "MEASUREMENT_ONLY"), ("MARS", "MEASUREMENT_ONLY")])
    def test_status_and_reason(self, region, status):
        enabled, reason = RegionalFeatureGate.is_feature_enabled("aortic_stenosis_classifier", region)
        assert enabled is (region == "US")
        assert reason == f"aortic_stenosis_classifier in {region}: {status}"

    def test_unknown_feature(self):
        assert RegionalFeatureGate.is_feature_enabled("nope", "US") == (False, "Unknown feature: nope")

    def test_deployment_mode(self):
        us = RegionalFeatureGate.get_deployment_mode("US")
        assert us["as_classifier_enabled"] is True and us["notes"] == ""
        assert us["deployment_mode"] == "segmentation_measurement_and_decision_support"
        eu = RegionalFeatureGate.get_deployment_mode("EU")
        assert eu["as_classifier_enabled"] is False and eu["region"] == "EU"
        assert eu["deployment_mode"] == "segmentation_measurement_only"
        assert eu["notes"].startswith("Decision-support features disabled")
        assert eu["regulatory_status"].endswith("PENDING_CE_MARKING")


@pytest.mark.xfail(
    strict=True, raises=AttributeError,
    reason="clinical_review_system.py example_clinical_review_workflow() calls `logger.basicConfig(...)`, but "
           "`logger` is a logging.Logger (no basicConfig; should be logging.basicConfig)",
)
def test_example_workflow_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ClinicalReviewWorkflow, "export_audit_trail", lambda self, p: None)
    crs.example_clinical_review_workflow()
    assert "CARDIAC ULTRASOUND ANALYSIS REPORT" in capsys.readouterr().out
