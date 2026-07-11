import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.reporting.capability_matrix import derive_matrix

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


@pytest.mark.parametrize("method", ["BFGS", "trust-exact"])
def test_bounds_measurement_agrees_with_declaration_for_trust_apls_bfgs_and_trust_exact(method):
    # _measure_bounds probes with scipy's minimize-family per-dimension
    # tuple list unless it recognises the method as one of scipy's own
    # least-squares names; trust-apl's BFGS/trust-exact share those
    # names with scipy's minimize family but still expect trust-apl's
    # own least-squares-style (lower, upper) array pair.
    df = derive_matrix(backends=[APLBackend()], fields=["bounds"])

    row = df[df["method"] == method].iloc[0]
    assert bool(row["declared"]) is True, method
    assert bool(row["measured"]) is True, method
    assert bool(row["agrees"]) is True, method
