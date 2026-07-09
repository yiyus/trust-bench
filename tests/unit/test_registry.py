import pytest

from trust_bench.core.problem import Problem
from trust_bench.core.registry import Registry


def _make_problem(id="dummy", residual=lambda x: x, tags=frozenset()):
    return Problem(
        id=id,
        residual=residual,
        jacobian=None,
        hessian=None,
        starts={},
        optima=[],
        kind="residuals",
        tags=tags,
        probe_points=[],
        source="test fixture",
    )


def test_get_returns_the_registered_problem_by_id():
    registry = Registry()
    problem = _make_problem(id="rosenbrock")
    registry.register(problem)

    assert registry.get("rosenbrock") is problem


def test_by_tag_returns_the_registered_problems_with_that_tag():
    registry = Registry()
    tagged = _make_problem(id="a", tags=frozenset({"ill-conditioned"}))
    untagged = _make_problem(id="b", tags=frozenset())
    registry.register(tagged)
    registry.register(untagged)

    assert registry.by_tag("ill-conditioned") == [tagged]


@pytest.mark.parametrize("residual", [None, "not callable", 42])
def test_register_rejects_a_problem_without_a_callable_residual(residual):
    problem = _make_problem(residual=residual)

    with pytest.raises(TypeError):
        Registry().register(problem)
