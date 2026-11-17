"""Non-network logic of heart_valve_analysis on synthetic masks.

Architecture/workflow scaffolding only: no trained model or real echo data is
involved, and nothing here is a clinical claim.
"""
import importlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def hva():
    return importlib.import_module("heart_valve_analysis")


def _meta(hva, spacing=0.5):
    return hva.FrameMetadata(cardiac_phase="ED", view_type="a4c", frame_idx=0,
                             pixel_spacing_x=spacing, pixel_spacing_y=spacing, image_quality="good")


def _lv_la_mask(h=64, w=64, lv_rows=(10, 32), la_rows=(32, 50), cols=(20, 44), lv_p=1.0, la_p=1.0):
    """(H, W, 3) mask: LV block directly above an LA block, sharing the row-32 boundary."""
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[lv_rows[0]:lv_rows[1], cols[0]:cols[1], 0] = lv_p
    m[la_rows[0]:la_rows[1], cols[0]:cols[1], 1] = la_p
    m[..., 2] = 1.0 - np.clip(m[..., 0] + m[..., 1], 0, 1)
    return m


def test_module_imports_without_missing_symbols():
    """`from sklearn.metrics import dice` does not exist; the module could not be imported at all."""
    importlib.import_module("heart_valve_analysis")


def test_mv_diameter_is_euclidean_distance(hva):
    est = hva.HingePointEstimate(anterior_hinge=(1.0, 2.0), posterior_hinge=(4.0, 6.0))
    assert est.mv_diameter_mm == pytest.approx(5.0)


def test_hinge_points_span_the_lv_la_contact_line(hva):
    est = hva.HingePointExtractor.extract_hinge_points(_lv_la_mask(), _meta(hva), apply_bias_correction=False)
    # contact width is 24 px (cols 20..43) -> ~23 px between endpoints at 0.5 mm/px
    assert 10.0 <= est.mv_diameter_mm <= 12.5


def test_bias_correction_shifts_both_points_equally(hva):
    raw = hva.HingePointExtractor.extract_hinge_points(_lv_la_mask(), _meta(hva), apply_bias_correction=False)
    cor = hva.HingePointExtractor.extract_hinge_points(_lv_la_mask(), _meta(hva), apply_bias_correction=True)
    b = hva.CAMUS_Benchmark
    assert cor.anterior_hinge[0] - raw.anterior_hinge[0] == pytest.approx(b.BIAS_HORIZONTAL_MM)
    assert cor.anterior_hinge[1] - raw.anterior_hinge[1] == pytest.approx(b.BIAS_VERTICAL_MM)
    assert cor.mv_diameter_mm == pytest.approx(raw.mv_diameter_mm)


def test_crisp_mask_has_full_confidence(hva):
    est = hva.HingePointExtractor.extract_hinge_points(_lv_la_mask(), _meta(hva))
    assert est.confidence_score == pytest.approx(1.0)


def test_uncertain_mask_has_low_confidence(hva):
    """Confidence compared each mask with itself (always 1.0), so the < 0.85 review flag could never fire."""
    est = hva.HingePointExtractor.extract_hinge_points(_lv_la_mask(lv_p=0.55, la_p=0.55), _meta(hva))
    assert est.confidence_score < 0.85


def test_as_rule_by_valve_area(hva):
    clf = hva.AorticStenosisClassifier.classify_aortic_stenosis
    assert clf(0.8, 3.0).severity == "severe"
    assert clf(1.0, 3.0) is None          # 1.0 cm2 is not < 1.0
    assert clf(1.6, 2.0) is None


def test_as_rule_uses_its_own_peak_velocity_threshold(hva):
    """PEAK_VELOCITY_THRESHOLDS declares >= 4.0 m/s as 'severe' but the rule never read velocity."""
    f = hva.AorticStenosisClassifier.classify_aortic_stenosis(1.2, 4.5)
    assert f is not None and f.severity == "severe"
    assert hva.AorticStenosisClassifier.classify_aortic_stenosis(1.2, 3.9) is None


class _FakeSegmenter:
    """Returns a fixed (1, 3, H, W) probability map like the real U-Net's softmax output."""
    def __init__(self, mask_hwc):
        self.out = np.transpose(mask_hwc, (2, 0, 1))[np.newaxis]

    def to(self, device):
        return self

    def __call__(self, x):
        import torch
        return torch.from_numpy(self.out)


def _pipeline(hva, region="US"):
    return hva.HeartVallueAnalysisPipeline(_FakeSegmenter(_lv_la_mask()), region=region, device="cpu")


def test_pipeline_feeds_channel_last_mask_to_hinge_extractor(hva):
    """The network returns (C, H, W); the extractor indexes (H, W, C)."""
    frames = np.zeros((1, 64, 64), dtype=np.uint8)
    res = _pipeline(hva).analyze_study(frames, _meta(hva))
    direct = hva.HingePointExtractor.extract_hinge_points(_lv_la_mask(), _meta(hva))
    assert res["hinge_point_estimates"][0].mv_diameter_mm == pytest.approx(direct.mv_diameter_mm)


def test_pipeline_does_not_invent_severe_stenosis_without_a_measurement(hva):
    """analyze_study hard-coded valve area 0.8 cm2, so every study was flagged severe AS."""
    frames = np.zeros((2, 64, 64), dtype=np.uint8)
    res = _pipeline(hva).analyze_study(frames, _meta(hva))
    assert not any(f.finding_type == "severe_aortic_stenosis" for f in res["flagged_findings"])


def test_pipeline_flags_supplied_severe_measurement_for_review(hva):
    frames = np.zeros((1, 64, 64), dtype=np.uint8)
    res = _pipeline(hva).analyze_study(frames, _meta(hva), aortic_valve_area_cm2=0.7, peak_velocity_ms=4.2)
    assert [f.finding_type for f in res["flagged_findings"]] == ["severe_aortic_stenosis"]
    assert res["requires_clinician_review"] is True


def test_as_rule_disabled_outside_us(hva):
    frames = np.zeros((1, 64, 64), dtype=np.uint8)
    res = _pipeline(hva, region="EU").analyze_study(frames, _meta(hva), aortic_valve_area_cm2=0.7,
                                                    peak_velocity_ms=4.2)
    assert res["flagged_findings"] == []


def test_validator_dice_on_identical_and_disjoint_masks(hva):
    a = _lv_la_mask()[np.newaxis]
    same = hva.SegmentationValidator.compute_metrics(a, a, 0.5, 0.5)
    assert same["dice_lv_median"] == pytest.approx(1.0)
    assert same["x_error_median_mm"] == pytest.approx(0.0)
    shifted = _lv_la_mask(lv_rows=(40, 60), la_rows=(2, 20))[np.newaxis]
    assert hva.SegmentationValidator.compute_metrics(a, shifted, 0.5, 0.5)["dice_lv_median"] == pytest.approx(0.0)
