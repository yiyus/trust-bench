import shutil

import pytest

from trust_bench.backends import apl_backend
from trust_bench.backends.apl_backend import APLBackend, evaluate_problem
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import rosenbrock

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


@pytest.fixture(autouse=True)
def _fresh_session():
    # Isolated from whatever session another test file may have already
    # warmed up: these tests assert on the session's own identity
    # (same/different pid), which only makes sense starting from a known
    # empty state, and must not leave a killed session behind for the
    # next file to silently pay for.
    apl_backend._kill_session()
    yield
    apl_backend._kill_session()


def test_two_independently_constructed_backends_share_one_session():
    APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))
    first_session = apl_backend._session

    APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert apl_backend._session is first_session


def test_evaluate_problem_and_solve_share_the_same_session():
    APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))
    first_session = apl_backend._session

    evaluate_problem("rosenbrock", [0.0, 0.0])

    assert apl_backend._session is first_session


def test_a_killed_session_is_transparently_replaced_by_the_next_call():
    APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))
    apl_backend._session.kill()
    apl_backend._session.wait()

    result = APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED
    assert apl_backend._session is not None
