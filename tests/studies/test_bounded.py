import pytest

from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.studies.bounded import METHODS, SCENARIOS, sweep

_TOL = 1e-6

# The APL backend does not exist yet, so this study covers only SciPy's
# trf/dogbox for now; a Coleman-Li-scaling comparison point is added
# once that backend exists.


def test_sweep_covers_every_scenario_and_method():
    outcomes = sweep()

    assert set(outcomes) == {
        (name, method, backend.name)
        for name in SCENARIOS
        for method in METHODS
        for backend in BACKENDS
    }


def test_inactive_bounds_converge_to_the_unconstrained_optimum():
    outcomes = sweep(scenarios={"inactive": SCENARIOS["inactive"]})

    for (name, method, backend_name), outcome in outcomes.items():
        assert isinstance(outcome, RunResult), f"{name}/{method}/{backend_name}"
        assert outcome.status is RunStatus.CONVERGED, f"{name}/{method}/{backend_name}"
        assert outcome.x_final[0] == pytest.approx(0.0, abs=_TOL), f"{name}/{method}/{backend_name}"
        assert outcome.x_final[1] == pytest.approx(0.0, abs=_TOL), f"{name}/{method}/{backend_name}"


def test_active_at_boundary_bounds_converge_exactly_to_the_boundary():
    outcomes = sweep(scenarios={"active_at_boundary": SCENARIOS["active_at_boundary"]})

    for (name, method, backend_name), outcome in outcomes.items():
        assert isinstance(outcome, RunResult), f"{name}/{method}/{backend_name}"
        assert outcome.status is RunStatus.CONVERGED, f"{name}/{method}/{backend_name}"
        assert outcome.x_final[0] == pytest.approx(0.5, abs=_TOL), f"{name}/{method}/{backend_name}"
        assert outcome.x_final[1] == pytest.approx(0.0, abs=_TOL), f"{name}/{method}/{backend_name}"


def test_infeasible_start_raises_a_clear_error_for_both_methods():
    # SciPy's least_squares validates x0 against bounds itself: an
    # infeasible start is not silently projected into the box, it is
    # rejected outright.
    outcomes = sweep(scenarios={"infeasible_start": SCENARIOS["infeasible_start"]})

    for (name, method, backend_name), outcome in outcomes.items():
        assert isinstance(outcome, ValueError), f"{name}/{method}/{backend_name}"
