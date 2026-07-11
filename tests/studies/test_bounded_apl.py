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


def test_trust_apls_bfgs_and_trust_exact_converge_to_the_same_constrained_optimum_as_lm():
    # BFGS/trust-exact are named to mirror scipy's own vocabulary but
    # still take trust-apl's least-squares-style (lower, upper) array
    # pair internally, not scipy's minimize-family per-dimension tuple
    # list; routing them by name alone sends the wrong format and
    # collapses or inverts the box.
    outcomes = sweep(backends=[SciPyBackend(), APLBackend()])

    for name in SCENARIOS:
        lm_outcome = outcomes[(name, "lm", "trust-apl")]
        assert isinstance(lm_outcome, RunResult), name
        for method in ["BFGS", "trust-exact"]:
            outcome = outcomes[(name, method, "trust-apl")]
            assert isinstance(outcome, RunResult), f"{name}/{method}"
            assert outcome.status is RunStatus.CONVERGED, f"{name}/{method}"
            assert outcome.x_final == pytest.approx(lm_outcome.x_final, abs=1e-3), f"{name}/{method}"
