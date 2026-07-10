import shutil

import pytest
from toy_backend import ToyGradientDescentBackend

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend

_APL_MARKS = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

_ENTRIES = [
    (ToyGradientDescentBackend(), []),
    (SciPyBackend(), []),
    (APLBackend(), _APL_MARKS),
]

BACKENDS = [
    pytest.param(backend, marks=marks, id=backend.name) if marks else backend for backend, marks in _ENTRIES
]

# One entry per (backend, method) pair, for contract tests that must
# exercise every method a backend declares, not just the first.
BACKEND_METHODS = [
    pytest.param(backend, method, marks=marks, id=f"{backend.name}-{method}")
    for backend, marks in _ENTRIES
    for method in backend.capabilities().methods
]
