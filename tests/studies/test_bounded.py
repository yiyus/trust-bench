import pytest

from trust_bench.backends import BACKENDS
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.studies.bounded import SCENARIOS, sweep

_TOL = 1e-3

# scipy's least_squares validates x0 against bounds itself and rejects an
# infeasible start outright; every other bounds-capable method here
# (scipy's minimize-family methods, trust-apl's Coleman-Li-scaled "lm")
# projects an infeasible start into the box and converges regardless.
_RAISES_ON_INFEASIBLE_START = frozenset({"trf", "dogbox"})


def _bounds_capable_methods(backends):
    return {
        (method, backend.name)
        for backend in backends
        for method, caps in backend.capabilities().methods.items()
        if caps.bounds
    }


def test_sweep_covers_every_scenario_and_every_bounds_capable_method():
    outcomes = sweep()

    assert set(outcomes) == {
        (name, method, backend_name)
        for name in SCENARIOS
        for method, backend_name in _bounds_capable_methods(BACKENDS)
    }


def test_sweep_exercises_more_than_the_least_squares_family():
    # scipy declares bounds=True for L-BFGS-B and trust-constr too, not
    # only trf/dogbox; a fixed method list would silently drop these.
    outcomes = sweep()

    methods = {method for _, method, _ in outcomes}
    assert {"L-BFGS-B", "trust-constr"} <= methods


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


def test_infeasible_start_is_rejected_by_least_squares_methods_and_projected_by_others():
    outcomes = sweep(scenarios={"infeasible_start": SCENARIOS["infeasible_start"]})

    for (name, method, backend_name), outcome in outcomes.items():
        if method in _RAISES_ON_INFEASIBLE_START:
            assert isinstance(outcome, ValueError), f"{name}/{method}/{backend_name}"
        else:
            assert isinstance(outcome, RunResult), f"{name}/{method}/{backend_name}"
            assert outcome.status is RunStatus.CONVERGED, f"{name}/{method}/{backend_name}"
            assert outcome.x_final[0] == pytest.approx(0.5, abs=_TOL), f"{name}/{method}/{backend_name}"


def test_sweep_skips_a_method_a_backend_does_not_declare_bounds_for():
    outcomes = sweep(scenarios={"inactive": SCENARIOS["inactive"]}, backends=[SciPyBackend()])

    assert "BFGS" not in {method for _, method, _ in outcomes}
