import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.studies.bounded import SCENARIOS, sweep

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


def test_trust_apls_bounded_lm_is_exercised_by_every_scenario_instead_of_reading_error():
    outcomes = sweep(backends=[SciPyBackend(), APLBackend()])

    for name in SCENARIOS:
        outcome = outcomes[(name, "lm", "trust-apl")]
        assert isinstance(outcome, RunResult), name
        assert outcome.status is RunStatus.CONVERGED, name
