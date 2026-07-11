import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.result import RunStatus
from trust_bench.studies.dimensionality import DENSE_METHODS, sweep

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


def test_trust_apls_dense_methods_converge_instead_of_reading_error():
    results = sweep(n_values=[10], methods=DENSE_METHODS, backends=[SciPyBackend(), APLBackend()])

    for method in DENSE_METHODS:
        result = results[(10, method, "trust-apl")]
        assert result.status is RunStatus.CONVERGED, method
        assert result.dist_to_opt < 1e-3, method
