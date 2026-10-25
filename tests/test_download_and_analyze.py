"""quick_analyze_pretrained regressions (static sweep 2026-10-25: ruff F821 `np`, bandit B614)."""
import sys
import types

import numpy as np
import pytest


class _T:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


@pytest.fixture
def fake_runtime(monkeypatch):
    """A torch/DataLoader/heart_valve_analysis surface just big enough for the function."""
    import torch  # real or the conftest stub

    calls = {}

    class _Model:
        def __init__(self, *a, **k):
            pass

        def to(self, *a, **k):
            return self

        def load_state_dict(self, sd):
            pass

        def eval(self):
            pass

        def __call__(self, images):
            return images

    monkeypatch.setattr(torch, "load", lambda path, **kw: calls.setdefault("load_kwargs", kw) and {} or {}, raising=False)
    monkeypatch.setattr(torch, "softmax", lambda logits, dim: _T(np.zeros((2, 3, 4, 4))), raising=False)
    import torch.utils.data as tud
    monkeypatch.setattr(tud, "DataLoader", lambda ds, batch_size: [(_T(np.zeros((2, 1, 4, 4))), _T(np.zeros((2, 4, 4))))] * 2, raising=False)

    hva = types.ModuleType("heart_valve_analysis")
    hva.CAMUSDataset = lambda **k: object()
    hva.UNetSegmenter = _Model
    hva.SegmentationValidator = types.SimpleNamespace(
        compute_metrics=lambda p, g, **k: {"n": int(p.shape[0]) + int(g.shape[0]), "dice_lv_median": 0.0,
                                           "dice_la_median": 0.0, "x_error_median_mm": 0.0, "y_error_median_mm": 0.0})
    monkeypatch.setitem(sys.modules, "heart_valve_analysis", hva)
    return calls


def test_quick_analyze_concatenates_batches_with_numpy(fake_runtime, tmp_path):
    """The module used `np.concatenate` after the validation loop without importing numpy,
    so every pretrained analysis crashed with NameError after the full pass over the data."""
    import DOWNLOAD_AND_ANALYZE as mod
    model, metrics = mod.quick_analyze_pretrained(str(tmp_path / "model.pth"))
    assert metrics["n"] == 8  # 2 batches x 2 images, predictions + ground truths concatenated


def test_pretrained_weights_are_loaded_with_weights_only(fake_runtime, tmp_path):
    import DOWNLOAD_AND_ANALYZE as mod
    mod.quick_analyze_pretrained(str(tmp_path / "model.pth"))
    assert fake_runtime["load_kwargs"].get("weights_only") is True
