import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.core.runner import run
from trust_bench.core.storage import append, load
from trust_bench.problems import rosenbrock

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

APL = APLBackend()


def test_a_real_apl_backend_run_round_trips_through_storage(tmp_path):
    path = tmp_path / "results.jsonl"
    method = next(iter(APL.capabilities().methods))

    result = run(rosenbrock.PROBLEM, APL, method, "standard", RunConfig(max_iter=200))
    append(result, path)
    df = load(path)

    assert result.status is RunStatus.CONVERGED
    assert len(df) == 1
    assert df.loc[0, "problem_id"] == "rosenbrock"
    assert df.loc[0, "backend"] == APL.name
    assert df.loc[0, "status"] == "CONVERGED"
    assert df.loc[0, "x_final"] == pytest.approx([1.0, 1.0], abs=1e-6)
