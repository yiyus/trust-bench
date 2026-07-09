import pytest

from trust_bench.backends import BACKENDS
from trust_bench.core.metrics import rate
from trust_bench.problems.families.large_residual import make
from trust_bench.studies.large_residual import (
    RHOS,
    backend_results,
    gn_spectral_radius,
    undamped_gauss_newton_errors,
)

_NEAR_PERTURBATION = 1.02


@pytest.mark.parametrize("rho", [10.0, 30.0])
def test_measured_gauss_newton_rate_matches_the_predicted_spectral_radius(rho):
    problem = make(rho)
    x_near = problem.optima[0].x_star * _NEAR_PERTURBATION

    predicted = gn_spectral_radius(problem)
    measured = rate(undamped_gauss_newton_errors(problem, x_near))

    assert predicted < 1.0
    assert measured == pytest.approx(predicted, rel=0.05)


def test_gauss_newton_diverges_once_the_predicted_spectral_radius_exceeds_one():
    problem = make(100.0)
    x_near = problem.optima[0].x_star * _NEAR_PERTURBATION

    predicted = gn_spectral_radius(problem)
    errors = undamped_gauss_newton_errors(problem, x_near)

    assert predicted > 1.0
    assert errors[-1] > errors[0]


def test_predicted_spectral_radius_increases_monotonically_with_rho():
    radii = [gn_spectral_radius(make(rho)) for rho in RHOS]

    assert all(a <= b for a, b in zip(radii, radii[1:]))


def test_backend_results_reports_grad_norm_final_and_basin_rate_for_every_rho_and_backend():
    # Section 9 item 2: unlike undamped Gauss-Newton above, a real
    # (globalized) backend is expected to converge across the whole rho
    # sweep, including past the divergence boundary; grad_norm_final and
    # the basin-of-attraction rate need only the final iterate, so they
    # are available regardless of whether the backend exposes a trace.
    results, rates = backend_results()

    assert len(results) == len(RHOS) * len(BACKENDS)
    assert len(rates) == len(RHOS) * len(BACKENDS)
    for (rho, backend_name), result in results.items():
        assert result.grad_norm_final is not None, f"rho={rho} on {backend_name}"
    for (rho, backend_name), r in rates.items():
        assert 0.0 <= r <= 1.0, f"rho={rho} on {backend_name} rate out of range: {r}"
