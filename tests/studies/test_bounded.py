from trust_bench.core.result import RunResult, RunStatus
from trust_bench.studies.bounded import METHODS, SCENARIOS, sweep

# The APL backend does not exist yet, so this study covers only SciPy's
# trf/dogbox for now; a Coleman-Li-scaling comparison point is added
# once that backend exists.


def test_sweep_covers_every_scenario_and_method():
    outcomes = sweep()

    assert set(outcomes) == {(name, method) for name in SCENARIOS for method in METHODS}


def test_inactive_bounds_converge_to_the_unconstrained_optimum():
    outcomes = sweep(scenarios={"inactive": SCENARIOS["inactive"]})

    for (name, method), outcome in outcomes.items():
        assert isinstance(outcome, RunResult), f"{name}/{method}"
        assert outcome.status is RunStatus.CONVERGED, f"{name}/{method}"
        assert outcome.x_final[0] == 0.0
        assert outcome.x_final[1] == 0.0


def test_active_at_boundary_bounds_converge_exactly_to_the_boundary():
    outcomes = sweep(scenarios={"active_at_boundary": SCENARIOS["active_at_boundary"]})

    for (name, method), outcome in outcomes.items():
        assert isinstance(outcome, RunResult), f"{name}/{method}"
        assert outcome.status is RunStatus.CONVERGED, f"{name}/{method}"
        assert outcome.x_final[0] == 0.5
        assert outcome.x_final[1] == 0.0


def test_infeasible_start_raises_a_clear_error_for_both_methods():
    # SciPy's least_squares validates x0 against bounds itself: an
    # infeasible start is not silently projected into the box, it is
    # rejected outright.
    outcomes = sweep(scenarios={"infeasible_start": SCENARIOS["infeasible_start"]})

    for (name, method), outcome in outcomes.items():
        assert isinstance(outcome, ValueError), f"{name}/{method}"
