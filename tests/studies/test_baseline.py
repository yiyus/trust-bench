import trust_bench.studies.baseline as baseline_module
from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunStatus
from trust_bench.problems import rosenbrock
from trust_bench.studies.baseline import CANONICAL_PROBLEMS, basin_rates, standard_start_results


def test_standard_start_converges_for_every_canonical_problem_and_backend():
    results = standard_start_results()

    assert len(results) == len(CANONICAL_PROBLEMS) * len(BACKENDS)
    for (problem_id, backend_name), result in results.items():
        assert result.status is RunStatus.CONVERGED, f"{problem_id} on {backend_name} did not converge"


def test_basin_rates_are_reported_per_problem_and_backend_within_zero_and_one():
    rates = basin_rates()

    assert len(rates) == len(CANONICAL_PROBLEMS) * len(BACKENDS)
    for (problem_id, backend_name), rate in rates.items():
        assert 0.0 <= rate <= 1.0, f"{problem_id} on {backend_name} rate out of range: {rate}"


def test_basin_rate_evaluates_every_registered_start_not_only_standard(monkeypatch):
    # Rosenbrock is the one canonical problem with a deliberately hard
    # second start ("far"); this proves basin_rates() aggregates across
    # every registered start rather than only "standard", which the
    # convergence-only assertions above cannot distinguish (every
    # canonical problem converges from every registered start at these
    # settings, so a rate of 1.0 alone doesn't prove "far" was evaluated).
    calls = []
    real_run = baseline_module.run

    def spy(problem, backend, method, start, config):
        calls.append((problem.id, start))
        return real_run(problem, backend, method, start, config)

    monkeypatch.setattr(baseline_module, "run", spy)

    basin_rates()

    rosenbrock_starts = {start for problem_id, start in calls if problem_id == "rosenbrock"}
    assert rosenbrock_starts == set(rosenbrock.PROBLEM.starts)
    assert "far" in rosenbrock_starts
