"""Logic-only tests for heart_valve_analysis (synthetic masks, fake torch objects, no model weights).

Nothing here makes a clinical claim: it checks the code's own arithmetic, thresholds and bookkeeping.
"""
import contextlib
import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def hva():
    return importlib.import_module("heart_valve_analysis")


def meta(hva, sx=0.5, sy=0.5, quality="good", phase="ED"):
    return hva.FrameMetadata(cardiac_phase=phase, view_type="a4c", frame_idx=0,
                             pixel_spacing_x=sx, pixel_spacing_y=sy, image_quality=quality)


def block_mask(lv=None, la=None, h=64, w=64, p=1.0):
    """(H, W, 3) mask with optional rectangular LV / LA blocks ((r0, r1, c0, c1) each)."""
    m = np.zeros((h, w, 3), dtype=np.float32)
    if lv:
        m[lv[0]:lv[1], lv[2]:lv[3], 0] = p
    if la:
        m[la[0]:la[1], la[2]:la[3], 1] = p
    m[..., 2] = 1.0 - np.clip(m[..., 0] + m[..., 1], 0, 1)
    return m


class TestDice:
    def test_values(self, hva):
        a = np.array([[1, 1, 0, 0]], dtype=bool)
        b = np.array([[1, 0, 1, 0]], dtype=bool)
        assert hva.dice(a, a) == 1.0
        assert hva.dice(a, b) == pytest.approx(0.5)
        assert hva.dice(a, ~a) == 0.0

    def test_both_empty_is_one_and_one_empty_is_zero(self, hva):
        z = np.zeros((3, 3))
        assert hva.dice(z, z) == 1.0
        assert hva.dice(z, np.ones((3, 3))) == 0.0

    def test_non_boolean_input_is_binarised(self, hva):
        assert hva.dice(np.array([0.2, 0.0]), np.array([5, 0])) == 1.0


class TestDataclasses:
    def test_estimate_defaults(self, hva):
        e = hva.HingePointEstimate((0.0, 0.0), (3.0, 4.0))
        assert (e.x_uncertainty_mm, e.y_uncertainty_mm, e.confidence_score) == (1.35, 0.75, 0.0)
        assert e.mv_diameter_mm == 5.0

    def test_finding_repr_and_defaults(self, hva):
        f = hva.ClinicalFinding("severe_aortic_stenosis", "severe", 0.8, 1.0)
        assert f.requires_review is True and f.supporting_measurements is None
        assert repr(f) == "SEVERE_AORTIC_STENOSIS: severe (0.80 vs. threshold 1.00)"


class TestClassifierBoundaries:
    @pytest.mark.parametrize("area,vel,expected", [
        (0.99, 0.0, "severe"), (1.0, 0.0, None), (1.49, 0.0, None), (1.5, 0.0, None), (3.0, 0.0, None),
        (3.0, 4.0, "severe"), (3.0, 3.99, None), (1.2, 4.0, "severe"), (0.5, 5.0, "severe"),
    ])
    def test_flags_only_severe(self, hva, area, vel, expected):
        f = hva.AorticStenosisClassifier.classify_aortic_stenosis(area, vel)
        assert (f.severity if f else None) == expected

    def test_finding_payload(self, hva):
        f = hva.AorticStenosisClassifier.classify_aortic_stenosis(0.7, 4.2)
        assert (f.finding_type, f.measurement_value, f.reference_threshold) == ("severe_aortic_stenosis", 0.7, 1.0)
        assert f.requires_review is True
        assert f.supporting_measurements == {"peak_velocity_ms": 4.2, "guideline": "ACC/AHA 2014"}

    def test_threshold_tables_are_contiguous(self, hva):
        t = hva.AorticStenosisClassifier.AORTIC_VALVE_AREA_THRESHOLDS
        assert t["severe"][1] == t["moderate"][0] and t["moderate"][1] == t["mild"][0]
        v = hva.AorticStenosisClassifier.PEAK_VELOCITY_THRESHOLDS
        assert v["mild"][1] == v["moderate"][0] and v["moderate"][1] == v["severe"][0]


class TestHingeExtraction:
    def test_fallback_uses_lv_extremes_when_no_contact(self, hva):
        # LV block rows 10..31, cols 20..43; LA far away -> no contact line
        m = block_mask(lv=(10, 32, 20, 44), la=(50, 60, 20, 44))
        est = hva.HingePointExtractor.extract_hinge_points(m, meta(hva, 0.3, 0.15), apply_bias_correction=False)
        assert est.anterior_hinge == pytest.approx((20 * 0.3, 20.5 * 0.15))
        assert est.posterior_hinge == pytest.approx((43 * 0.3, 20.5 * 0.15))
        assert est.mv_diameter_mm == pytest.approx(23 * 0.3)

    def test_pixel_spacing_scales_each_axis(self, hva):
        m = block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44))
        a = hva.HingePointExtractor.extract_hinge_points(m, meta(hva, 1.0, 1.0), apply_bias_correction=False)
        b = hva.HingePointExtractor.extract_hinge_points(m, meta(hva, 2.0, 3.0), apply_bias_correction=False)
        assert b.anterior_hinge[0] == pytest.approx(2.0 * a.anterior_hinge[0])
        assert b.anterior_hinge[1] == pytest.approx(3.0 * a.anterior_hinge[1])
        assert b.posterior_hinge[0] == pytest.approx(2.0 * a.posterior_hinge[0])
        assert b.posterior_hinge[1] == pytest.approx(3.0 * a.posterior_hinge[1])

    def test_bias_constants_applied_to_both_points(self, hva):
        m = block_mask(lv=(10, 32, 20, 44), la=(50, 60, 20, 44))
        raw = hva.HingePointExtractor.extract_hinge_points(m, meta(hva), apply_bias_correction=False)
        cor = hva.HingePointExtractor.extract_hinge_points(m, meta(hva))   # default: corrected
        b = hva.CAMUS_Benchmark
        for r, c in ((raw.anterior_hinge, cor.anterior_hinge), (raw.posterior_hinge, cor.posterior_hinge)):
            assert c[0] - r[0] == pytest.approx(b.BIAS_HORIZONTAL_MM)
            assert c[1] - r[1] == pytest.approx(b.BIAS_VERTICAL_MM)

    def test_confidence_is_mean_of_both_chambers(self, hva):
        # LV crisp (1.0), LA uncertain (0.6): soft Dice of a flat map p vs hard mask is 2p/(p+1)
        m = block_mask(lv=(10, 32, 20, 44))
        m[32:50, 20:44, 1] = 0.6
        est = hva.HingePointExtractor.extract_hinge_points(m, meta(hva))
        # LA probs are 0.6 > 0.5 so the hard mask is the same block
        assert est.confidence_score == pytest.approx((1.0 + 2 * 0.6 / 1.6) / 2)

    def test_confidence_one_when_both_masks_empty_requires_nonempty_lv(self, hva):
        m = block_mask(lv=(10, 32, 20, 44))           # LA empty -> its agreement is defined as 1.0
        est = hva.HingePointExtractor.extract_hinge_points(m, meta(hva))
        assert est.confidence_score == pytest.approx(1.0)

    def test_threshold_is_strictly_above_half(self, hva):
        raw = dict(apply_bias_correction=False)
        m = block_mask(lv=(10, 32, 20, 44))
        m[28:50, 20:44, 1] = 0.5                       # LA overlaps the LV edge but exactly 0.5 is not LA
        est = hva.HingePointExtractor.extract_hinge_points(m, meta(hva), **raw)
        assert est.anterior_hinge == pytest.approx((10.0, 10.25))     # fallback path
        m[28:50, 20:44, 1] = 0.51                      # just above 0.5 -> contact line found
        est = hva.HingePointExtractor.extract_hinge_points(m, meta(hva), **raw)
        assert 13.9 <= est.anterior_hinge[1] <= 15.6 and 13.9 <= est.posterior_hinge[1] <= 15.6

    @pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="heart_valve_analysis.py line 253: the 'two endpoints' of the contact line are chosen as the two "
               "points farthest from the centroid, which are neighbours at the same end rather than opposite ends, "
               "so the hinge-to-hinge distance collapses (0.5 mm here for a 24 px wide contact)",
    )
    def test_contact_line_endpoints_are_opposite_ends(self, hva):
        m = block_mask(lv=(10, 32, 20, 44), la=(28, 50, 20, 44))
        est = hva.HingePointExtractor.extract_hinge_points(m, meta(hva), apply_bias_correction=False)
        assert 10.0 <= est.mv_diameter_mm <= 13.0

    @pytest.mark.xfail(
        strict=True, raises=ValueError,
        reason="heart_valve_analysis.py line 247: the no-contact fallback calls np.min/np.max on the LV "
               "coordinates, so an all-background segmentation raises ValueError instead of returning "
               "a low-confidence estimate",
    )
    def test_empty_segmentation_does_not_crash(self, hva):
        est = hva.HingePointExtractor.extract_hinge_points(block_mask(), meta(hva))
        assert isinstance(est, hva.HingePointEstimate)


class _Tensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def to(self, *_a, **_k):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def __getitem__(self, i):
        return _Tensor(self.arr[i])

    def __len__(self):
        return len(self.arr)

    def item(self):
        return float(self.arr)


class _Loss(_Tensor):
    def __init__(self, v, log):
        super().__init__(v)
        self.log = log

    def backward(self):
        self.log.append("backward")


class _Optim:
    log = None
    lr = None

    def __init__(self, params, lr):
        type(self).lr = lr

    def zero_grad(self):
        _Optim.log.append("zero_grad")

    def step(self):
        _Optim.log.append("step")


class _Model:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.modes = []
        self.device = None

    def to(self, d):
        self.device = d
        return self

    def parameters(self):
        return []

    def train(self):
        self.modes.append("train")

    def eval(self):
        self.modes.append("eval")

    def __call__(self, x):
        return self.outputs.pop(0)


@pytest.fixture
def fake_torch(hva, monkeypatch):
    ft = SimpleNamespace(no_grad=contextlib.nullcontext, softmax=lambda x, dim: x, cuda=SimpleNamespace(
        is_available=lambda: False))
    monkeypatch.setattr(hva, "torch", ft)
    return ft


class TestTrainer:
    def _trainer(self, hva, monkeypatch, losses, model=None):
        log = []
        _Optim.log = log
        monkeypatch.setattr(hva, "Adam", _Optim)
        crit_vals = iter(losses)
        monkeypatch.setattr(hva, "nn", SimpleNamespace(CrossEntropyLoss=lambda: (lambda o, m: _Loss(next(crit_vals), log))))
        return hva.SegmentationTrainer(model or _Model(), device="cpu", learning_rate=0.01), log

    def test_constructor_wiring(self, hva, monkeypatch):
        model = _Model()
        tr, _ = self._trainer(hva, monkeypatch, [], model)
        assert model.device == "cpu" and tr.device == "cpu" and _Optim.lr == 0.01
        assert tr.history == {"train_loss": [], "val_dice": []}

    def test_train_epoch_averages_and_steps(self, hva, monkeypatch, fake_torch):
        model = _Model([_Tensor([0]), _Tensor([0]), _Tensor([0])])
        tr, log = self._trainer(hva, monkeypatch, [1.0, 2.0, 6.0], model)
        loader = [(_Tensor([1]), _Tensor([1]))] * 3
        loss = tr.train_epoch(loader)
        assert loss == pytest.approx(3.0) and tr.history["train_loss"] == [pytest.approx(3.0)]
        assert model.modes == ["train"]
        assert log == ["zero_grad", "backward", "step"] * 3

    def test_validate_median_dice_and_history(self, hva, monkeypatch, fake_torch):
        # batch of 2, (B, C, H, W) with 2x2 maps: LV identical, LA overlap 1 of (2 + 1) cells in sample 0
        lv = np.ones((2, 2, 2))
        pred_la = np.zeros((2, 2, 2))
        pred_la[0, 0, :] = 1
        true_la = np.zeros((2, 2, 2))
        true_la[0, 0, 0] = 1
        true_la[1] = pred_la[1] = np.eye(2)
        preds = np.stack([lv, pred_la], axis=1)
        truth = np.stack([lv, true_la], axis=1)
        model = _Model([_Tensor(preds)])
        tr, _ = self._trainer(hva, monkeypatch, [], model)
        out = tr.validate([(_Tensor(preds), _Tensor(truth), None)])
        assert model.modes == ["eval"]
        assert out["dice_lv"] == 1.0
        assert out["dice_la"] == pytest.approx((2 / 3 + 1.0) / 2)
        assert out["benchmark_lv_ed"] == hva.CAMUS_Benchmark.DICE_LV_ED
        assert out["benchmark_lv_es"] == hva.CAMUS_Benchmark.DICE_LV_ES
        assert tr.history["val_dice"] == [1.0]


class TestDatasets:
    def test_echo_dataset_defaults(self, hva, tmp_path):
        ds = hva.ECHODataset(tmp_path, transform="t")
        assert len(ds) == 0 and ds.transform == "t" and ds.target_shape == (256, 256)
        assert ds.dicom_dir == tmp_path and ds[0] is None

    def test_camus_dataset(self, hva, caplog):
        ds = hva.CAMUSDataset(split="val", target_shape=(64, 64))
        assert (ds.split, ds.target_shape, len(ds), ds[0]) == ("val", (64, 64), 0, None)
        with caplog.at_level(logging.INFO):
            ds.download()
        assert "creatis.insa-lyon.fr" in caplog.text
        assert hva.CAMUSDataset().split == "train"


class _Seg:
    """Stand-in segmenter returning a fixed (1, 3, H, W) map."""

    def __init__(self, hva, mask_hwc):
        self.hva = hva
        self.out = np.transpose(mask_hwc, (2, 0, 1))[np.newaxis]
        self.device = None
        self.calls = 0

    def to(self, d):
        self.device = d
        return self

    def __call__(self, x):
        self.calls += 1
        return self.hva.torch.from_numpy(self.out)


class TestPipeline:
    def _run(self, hva, region="US", frames=2, mask=None, **kw):
        mask = block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44)) if mask is None else mask
        seg = _Seg(hva, mask)
        pipe = hva.HeartVallueAnalysisPipeline(seg, region=region, device="cpu")
        res = pipe.analyze_study(np.zeros((frames, 64, 64), dtype=np.uint8), meta(hva, quality="poor"), **kw)
        return pipe, seg, res

    def test_constructor_state(self, hva):
        pipe, seg, _ = self._run(hva)
        assert seg.device == "cpu" and pipe.region == "US" and pipe.as_classifier_enabled is True
        assert hva.HeartVallueAnalysisPipeline(seg, region="EU", device="cpu").as_classifier_enabled is False

    def test_region_warning_logged(self, hva, caplog):
        with caplog.at_level(logging.WARNING):
            hva.HeartVallueAnalysisPipeline(_Seg(hva, block_mask()), region="ASIA", device="cpu")
        assert "AS decision-support disabled for region=ASIA" in caplog.text

    def test_each_frame_segmented_once(self, hva):
        _, seg, res = self._run(hva, frames=3)
        assert seg.calls == 3 and res["frames_analyzed"] == 3 and len(res["hinge_point_estimates"]) == 3
        assert res["requires_clinician_review"] is False and res["flagged_findings"] == []
        assert res["metadata"].image_quality == "poor"

    def test_low_confidence_frames_are_flagged_with_indices(self, hva):
        mask = block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44), p=0.55)
        _, _, res = self._run(hva, frames=2, mask=mask)
        flags = res["flagged_findings"]
        assert [f.finding_type for f in flags] == ["low_confidence_segmentation"] * 2
        assert [f.supporting_measurements["frame_idx"] for f in flags] == [0, 1]
        assert flags[0].supporting_measurements["image_quality"] == "poor"
        assert flags[0].reference_threshold == 0.85 and flags[0].severity == "moderate"
        assert flags[0].measurement_value < 0.85
        assert res["requires_clinician_review"] is True

    def test_confidence_exactly_at_threshold_is_not_flagged(self, hva):
        _, _, res = self._run(hva)           # crisp mask -> confidence 1.0
        assert res["flagged_findings"] == []

    def test_as_rule_needs_both_measurements(self, hva):
        _, _, res = self._run(hva, aortic_valve_area_cm2=0.5)
        assert res["flagged_findings"] == []
        _, _, res = self._run(hva, peak_velocity_ms=5.0)
        assert res["flagged_findings"] == []

    def test_as_finding_appended_after_low_confidence_flags(self, hva):
        mask = block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44), p=0.55)
        _, _, res = self._run(hva, frames=1, mask=mask, aortic_valve_area_cm2=0.5, peak_velocity_ms=1.0)
        assert [f.finding_type for f in res["flagged_findings"]] == [
            "low_confidence_segmentation", "severe_aortic_stenosis"]

    def test_non_severe_measurements_add_nothing(self, hva):
        _, _, res = self._run(hva, aortic_valve_area_cm2=2.0, peak_velocity_ms=2.0)
        assert res["flagged_findings"] == [] and res["requires_clinician_review"] is False

    def test_frames_are_normalised_to_unit_range(self, hva, monkeypatch):
        seen = []
        real = hva.torch.from_numpy
        monkeypatch.setattr(hva.torch, "from_numpy", lambda a: (seen.append(a), real(a))[1])
        pipe = hva.HeartVallueAnalysisPipeline(_Seg(hva, block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44))),
                                               device="cpu")
        frames = np.full((1, 64, 64), 255, dtype=np.uint8)
        pipe.analyze_study(frames, meta(hva))
        assert seen[0].shape == (1, 1, 64, 64) and seen[0].dtype == np.float32
        assert seen[0].max() == pytest.approx(1.0)


class TestValidator:
    def _pair(self, shift=0):
        a = block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44))
        b = block_mask(lv=(10 + shift, 32 + shift, 20, 44), la=(32 + shift, 50 + shift, 20, 44))
        return a, b

    def test_metrics_keys_and_values(self, hva):
        a, b = self._pair(shift=0)
        out = hva.SegmentationValidator.compute_metrics(np.stack([a, a]), np.stack([a, b]), 0.5, 0.25)
        assert out["dice_lv_median"] == 1.0 and out["dice_la_median"] == 1.0
        assert out["dice_lv_p15_p85"] == (1.0, 1.0)
        assert out["x_error_median_mm"] == 0.0 and out["y_error_p85_mm"] == 0.0
        assert out["benchmark_lv_ed"] == hva.CAMUS_Benchmark.DICE_LV_ED

    def test_vertical_shift_gives_y_error_scaled_by_spacing(self, hva):
        a, b = self._pair(shift=4)
        out = hva.SegmentationValidator.compute_metrics(a[None], b[None], 0.5, 0.25)
        assert out["dice_lv_median"] < 1.0
        assert out["x_error_median_mm"] == pytest.approx(0.0, abs=1e-9)
        assert out["y_error_median_mm"] > 0.0
        out2 = hva.SegmentationValidator.compute_metrics(a[None], b[None], 0.5, 0.5)
        assert out2["y_error_median_mm"] == pytest.approx(2.0 * out["y_error_median_mm"])

    def test_percentile_vs_median_over_batch(self, hva):
        a, b = self._pair(shift=0)
        _, far = self._pair(shift=8)
        preds = np.stack([a, a, a, far])
        truth = np.stack([a, a, a, a])
        out = hva.SegmentationValidator.compute_metrics(preds, truth, 0.5, 0.5)
        assert out["dice_lv_median"] == 1.0
        assert out["dice_lv_p15_p85"][0] <= out["dice_lv_median"] <= out["dice_lv_p15_p85"][1]
        assert out["y_error_p85_mm"] > out["y_error_median_mm"] == 0.0


class TestMain:
    def test_main_reports_results(self, hva, monkeypatch, caplog):
        mask = block_mask(lv=(10, 32, 20, 44), la=(32, 50, 20, 44), h=256, w=256)
        monkeypatch.setattr(hva, "UNetSegmenter", lambda in_channels, num_classes: _Seg(hva, mask))
        with caplog.at_level(logging.INFO):
            hva.main()
        text = caplog.text
        assert "Frames analyzed: 30" in text
        assert "Requires clinician review: False" in text
        assert "MITRAL VALVE MEASUREMENTS (First frame):" in text and "MV Diameter:" in text
