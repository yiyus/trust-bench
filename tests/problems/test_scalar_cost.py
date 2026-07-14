import pytest
from invariants import assert_known_optimum

from trust_bench.problems import SCALAR_PROBLEMS as PROBLEMS


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_scalar_problems_declare_no_jacobian_or_hessian(problem):
    assert problem.kind == "scalar"
    assert problem.jacobian is None
    assert problem.hessian is None


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_known_optimum_satisfies_gradient_and_cost_invariants(problem):
    assert_known_optimum(problem, atol=1e-6)
