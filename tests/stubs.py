"""Lightweight stand-ins for torch / cv2 / sklearn so tests run without heavy ML dependencies.

Only the surface used by ``heart_valve_analysis`` at import time and in the inference path is
implemented. Nothing here performs real deep learning.
"""
from __future__ import annotations

import sys
import types

import numpy as np


class FakeTensor:
    """Minimal tensor wrapper around a numpy array."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __getitem__(self, item):
        return FakeTensor(self.array[item])


class _Module:
    """Stand-in for ``torch.nn.Module`` (records nothing, accepts anything)."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, _device):
        return self

    def parameters(self):
        return []

    def eval(self):
        return self

    def train(self):
        return self


def _layer_factory(name):
    return type(name, (_Module,), {})


def _canny(img, _low, _high):
    """Binary-edge approximation: foreground pixels with a background 4-neighbour."""
    fg = img > 0
    padded = np.pad(fg, 1)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return ((fg & ~inner) * 255).astype(np.uint8)


def dice(a, b):
    """Dice coefficient of two masks (2|A∩B| / (|A|+|B|))."""
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2.0 * (a & b).sum() / total)


def install() -> None:
    """Register fake modules in ``sys.modules`` (no-op for modules that are really installed)."""
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
            return
        except ImportError:
            pass
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = _Module
    for layer in ("Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "ConvTranspose2d",
                  "Softmax", "Sequential", "CrossEntropyLoss"):
        setattr(nn, layer, _layer_factory(layer))
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset = type("Dataset", (), {})
    data.DataLoader = type("DataLoader", (), {})
    optim = types.ModuleType("torch.optim")
    optim.Adam = lambda *a, **k: types.SimpleNamespace(zero_grad=lambda: None, step=lambda: None)
    torch.nn, torch.utils, torch.optim = nn, utils, optim
    utils.data = data
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.from_numpy = FakeTensor
    torch.cat = lambda tensors, dim=1: tensors[0]
    torch.softmax = lambda t, dim=1: t

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.no_grad = _NoGrad
    cv2 = types.ModuleType("cv2")
    cv2.Canny = _canny
    sklearn = types.ModuleType("sklearn")
    metrics = types.ModuleType("sklearn.metrics")
    metrics.dice = dice  # not a real sklearn symbol; see README "Limitations"
    metrics.jaccard_score = lambda *a, **k: 0.0
    sklearn.metrics = metrics
    sys.modules.update({
        "torch": torch, "torch.nn": nn, "torch.utils": utils, "torch.utils.data": data,
        "torch.optim": optim, "cv2": cv2, "sklearn": sklearn, "sklearn.metrics": metrics,
    })
