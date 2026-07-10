import shutil

import pytest
from toy_backend import ToyGradientDescentBackend

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend

BACKENDS = [
    ToyGradientDescentBackend(),
    SciPyBackend(),
    pytest.param(
        APLBackend(),
        marks=[
            pytest.mark.slow,
            pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
        ],
        id="trust-apl",
    ),
]
