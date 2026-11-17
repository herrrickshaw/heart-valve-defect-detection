"""Test setup: repo root on sys.path and, when PyTorch is absent, a minimal stub.

torch (~700 MB) is only needed for the untrained U-Net. The stub lets the
numpy/OpenCV logic (hinge extraction, confidence, AS rule, pipeline wiring with
a fake segmenter) be imported and tested without it. Nothing here trains or
runs the network.
"""
import contextlib
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:  # pragma: no cover - real torch wins when installed
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
    import numpy as np

    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *a, **k):
            pass

        def to(self, *a, **k):
            return self

    class _Layer(_Module):
        pass

    nn.Module = _Module
    for name in ("Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "ConvTranspose2d", "Softmax",
                 "Sequential", "CrossEntropyLoss", "Dropout2d"):
        setattr(nn, name, _Layer)

    class _Tensor:
        """Wraps a numpy array with the few tensor methods the pipeline calls."""
        def __init__(self, arr):
            self.arr = np.asarray(arr)

        def to(self, *a, **k):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

        def __getitem__(self, i):
            return _Tensor(self.arr[i])

    torch.nn = nn
    torch.Tensor = _Tensor
    torch.from_numpy = lambda a: _Tensor(a)
    torch.no_grad = contextlib.nullcontext
    torch.cat = lambda xs, dim=0: _Tensor(np.concatenate([x.arr for x in xs], axis=dim))
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset = object
    data.DataLoader = object
    utils.data = data
    torch.utils = utils
    optim = types.ModuleType("torch.optim")
    optim.Adam = object
    torch.optim = optim
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.utils": utils,
                        "torch.utils.data": data, "torch.optim": optim})
