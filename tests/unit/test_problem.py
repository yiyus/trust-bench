import dataclasses

import pytest

from trust_bench.core.problem import Problem


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


def test_problem_is_immutable():
    problem = _make_problem()

    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.id = "changed"
