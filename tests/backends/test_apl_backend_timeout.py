import subprocess

import pytest

from trust_bench.backends import apl_backend
from trust_bench.backends.apl_backend import APLBackend, evaluate_problem
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import rosenbrock


_real_run = subprocess.run


def _raise_timeout_for_the_harness_only(cmd, *args, **kwargs):
    # subprocess.run is a shared, module-level function: patching it
    # unconditionally would also break unrelated calls this test doesn't
    # own (e.g. harness_git_sha's own "git rev-parse HEAD").
    if "run_harness.sh" in cmd[1]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)
    return _real_run(cmd, *args, **kwargs)


def test_solve_reports_a_clean_error_status_when_the_harness_times_out(monkeypatch):
    monkeypatch.setattr(apl_backend.subprocess, "run", _raise_timeout_for_the_harness_only)

    result = APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert result.status is RunStatus.ERROR
    # The status alone is indistinguishable from any other harness-side
    # ERROR (e.g. an unknown problem_id); the message is what actually
    # tells a reader this was a timeout, not a crash.
    assert "did not complete" in result.message


def test_evaluate_problem_raises_a_clear_error_when_the_harness_times_out(monkeypatch):
    monkeypatch.setattr(apl_backend.subprocess, "run", _raise_timeout_for_the_harness_only)

    with pytest.raises(RuntimeError, match="did not complete"):
        evaluate_problem("rosenbrock", [0.0, 0.0])
