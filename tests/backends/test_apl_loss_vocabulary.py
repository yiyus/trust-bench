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


@pytest.mark.parametrize("loss", ["tukey", "welsch", "fair"])
def test_lm_accepts_the_new_loss_without_raising(loss):
    result = BACKEND.solve(quadratic.PROBLEM, "lm", "standard", RunConfig(max_iter=100, loss=loss))

    assert result.status is RunStatus.CONVERGED
