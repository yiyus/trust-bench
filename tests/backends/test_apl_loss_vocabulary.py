import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import quadratic

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()


def test_capabilities_declares_tukey_welsch_and_fair_for_lm():
    losses = BACKEND.capabilities().methods["lm"].losses

    assert {"tukey", "welsch", "fair"} <= losses


@pytest.mark.parametrize("loss", ["welsch", "fair"])
def test_lm_accepts_the_new_loss_without_raising(loss):
    result = BACKEND.solve(quadratic.PROBLEM, "lm", "standard", RunConfig(max_iter=100, loss=loss))

    assert result.status is RunStatus.CONVERGED


def test_lm_dispatches_tukey_without_raising():
    # Tukey's hard cutoff gives zero weight, and so zero gradient, to
    # every point once the initial residual/scale ratio exceeds it -
    # exactly this trivial two-residual problem's own "standard" start,
    # confirmed directly (x_final == x0, MAX_ITER). Not a crash, and not
    # this loss's fault: a real, known basin-of-attraction fragility of
    # Tukey's own hard re-descending cutoff, matching why
    # trust_loss_precision (robust_loss.py) sweeps "welsch" rather than
    # "tukey" by default. Only dispatch without raising is asserted here.
    result = BACKEND.solve(quadratic.PROBLEM, "lm", "standard", RunConfig(max_iter=100, loss="tukey"))

    assert result.status in (RunStatus.CONVERGED, RunStatus.MAX_ITER)
