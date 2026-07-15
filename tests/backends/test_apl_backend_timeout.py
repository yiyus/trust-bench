import subprocess

import pytest

from trust_bench.backends import apl_backend
from trust_bench.backends.apl_backend import APLBackend, evaluate_problem
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import rosenbrock


def _never_ready(*args, **kwargs):
    return [], [], []


def _start_harmless_stub_session():
    # A real, always-available subprocess (`cat`, no dyalogscript
    # dependency) standing in for the session: these tests exercise
    # _send_request's own timeout/recovery bookkeeping, not the real APL
    # protocol, so they shouldn't require Dyalog to be installed to run.
    return subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


@pytest.fixture(autouse=True)
def _stub_session(monkeypatch):
    # _session is a module-level singleton shared with every other test
    # file; substituting a fresh stub (rather than reusing/killing
    # whatever real session another file may already have warmed up)
    # keeps this file's own timeout simulation isolated, and monkeypatch
    # restores the original _session/_start_session automatically at
    # teardown - killing the stub here must happen first, before that
    # restore, or _kill_session would tear down someone else's session.
    monkeypatch.setattr(apl_backend, "_session", None)
    monkeypatch.setattr(apl_backend, "_start_session", _start_harmless_stub_session)
    yield
    apl_backend._kill_session()


def test_solve_reports_a_clean_error_status_when_the_session_does_not_respond(monkeypatch):
    monkeypatch.setattr(apl_backend, "_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(apl_backend.select, "select", _never_ready)

    result = APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert result.status is RunStatus.ERROR
    # The status alone is indistinguishable from any other harness-side
    # ERROR (e.g. an unknown problem_id); the message is what actually
    # tells a reader this was a timeout, not a crash.
    assert "did not complete" in result.message


def test_evaluate_problem_raises_a_clear_error_when_the_session_does_not_respond(monkeypatch):
    monkeypatch.setattr(apl_backend, "_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(apl_backend.select, "select", _never_ready)

    with pytest.raises(RuntimeError, match="did not complete"):
        evaluate_problem("rosenbrock", [0.0, 0.0])


def test_a_timed_out_session_is_killed_not_reused(monkeypatch):
    monkeypatch.setattr(apl_backend, "_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(apl_backend.select, "select", _never_ready)

    APLBackend().solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert apl_backend._session is None
