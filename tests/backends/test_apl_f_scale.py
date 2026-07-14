import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.problems.families import outliers

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()
START = "standard"


@pytest.mark.parametrize("method", ["lm", "BFGS", "trust-exact"])
def test_every_method_rejects_f_scale(method):
    # trust's Loss namespace bakes in a fixed per-loss tuning constant
    # (Loss.apln: huber=1.345, cauchy=2.385) plus its own MAD-based
    # auto-scaling recomputed every call - there is no per-request knob
    # to override either, so an explicit f_scale must be rejected rather
    # than silently ignored.
    problem = outliers.make(0.2)

    with pytest.raises(ValueError, match="f_scale"):
        BACKEND.solve(problem, method, START, RunConfig(max_iter=200, loss="linear", f_scale=1.345))
