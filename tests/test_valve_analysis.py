"""Characterisation tests for hinge-point extraction, AS classification and the pipeline."""
from __future__ import annotations

import numpy as np
import pytest
import stubs

import heart_valve_analysis as hva

META = hva.FrameMetadata("ED", "a4c", 0, 0.3, 0.15, "good")


def _mask():
    m = np.zeros((40, 40, 3), np.float32)
    m[5:30, 5:20, 0] = 1
    m[5:30, 20:35, 1] = 1
    return m


def test_mv_diameter_is_euclidean():
    est = hva.HingePointEstimate((0.0, 0.0), (3.0, 4.0))
    assert est.mv_diameter_mm == pytest.approx(5.0)


def test_hinge_extraction_applies_bias_correction():
    raw = hva.HingePointExtractor.extract_hinge_points(_mask(), META, apply_bias_correction=False)
    biased = hva.HingePointExtractor.extract_hinge_points(_mask(), META)
    dx = biased.anterior_hinge[0] - raw.anterior_hinge[0]
    dy = biased.anterior_hinge[1] - raw.anterior_hinge[1]
    assert dx == pytest.approx(hva.CAMUS_Benchmark.BIAS_HORIZONTAL_MM)
    assert dy == pytest.approx(hva.CAMUS_Benchmark.BIAS_VERTICAL_MM)


def test_hinge_fallback_when_no_contact_line():
    m = np.zeros((20, 20, 3), np.float32)
    m[2:10, 2:10, 0] = 1
    est = hva.HingePointExtractor.extract_hinge_points(m, META, apply_bias_correction=False)
    assert est.anterior_hinge[0] == pytest.approx(2 * 0.3)
    assert est.posterior_hinge[0] == pytest.approx(9 * 0.3)


@pytest.mark.parametrize("area,flagged", [(0.5, True), (0.99, True), (1.0, False), (2.0, False)])
def test_aortic_stenosis_flags_only_severe(area, flagged):
    finding = hva.AorticStenosisClassifier.classify_aortic_stenosis(area, 4.5)
    assert (finding is not None) is flagged
    if finding:
        assert finding.finding_type == "severe_aortic_stenosis"
        assert finding.reference_threshold == 1.0


class _FakeModel:
    def to(self, _device):
        return self

    def __call__(self, _tensor):
        return stubs.FakeTensor(_mask()[np.newaxis])


def _run(region):
    pipe = hva.HeartVallueAnalysisPipeline(_FakeModel(), region=region, device="cpu")
    return pipe.analyze_study(np.zeros((3, 40, 40), np.uint8), META)


def test_pipeline_us_enables_as_placeholder_finding():
    res = _run("US")
    assert res["frames_analyzed"] == 3 and len(res["hinge_point_estimates"]) == 3
    assert res["requires_clinician_review"] is True
    assert [f.finding_type for f in res["flagged_findings"]] == ["severe_aortic_stenosis"]


def test_pipeline_non_us_has_no_as_finding():
    res = _run("EU")
    assert res["flagged_findings"] == [] and res["requires_clinician_review"] is False


def test_validator_perfect_prediction():
    m = np.stack([_mask(), _mask()])
    metrics = hva.SegmentationValidator.compute_metrics(m, m, 0.3, 0.15)
    assert metrics["dice_lv_median"] == pytest.approx(1.0)
    assert metrics["x_error_median_mm"] == 0.0
