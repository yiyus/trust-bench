import pytest
from invariants import assert_known_optimum, assert_parity

from trust_bench.problems import beale, expdec, helical, linear, powell, quadratic, rosenbrock

PROBLEMS = [
    rosenbrock.PROBLEM,
    beale.PROBLEM,
    powell.PROBLEM,
    helical.PROBLEM,
    expdec.PROBLEM,
    quadratic.PROBLEM,
    linear.PROBLEM,
]


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_analytic_jacobian_and_hessian_match_finite_differences_at_every_probe_point(problem):
    assert_parity(problem, jacobian_tol=dict(atol=1e-6), hessian_tol=dict(atol=1e-2))


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_known_optimum_satisfies_gradient_and_cost_invariants(problem):
    assert_known_optimum(problem, atol=1e-8)
