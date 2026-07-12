import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.result import RunStatus
from trust_bench.problems import TYPICAL_PROBLEMS
from trust_bench.studies.typical import sweep

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


def test_trust_apls_lm_and_bfgs_converge_on_every_typical_problem():
    results = sweep(backends=[SciPyBackend(), APLBackend()])

    for problem in TYPICAL_PROBLEMS:
        for method in ["lm", "BFGS"]:
            result = results[(problem.id, method, "trust-apl")]
            assert result.status is RunStatus.CONVERGED, f"{problem.id}/{method}"
            assert result.dist_to_opt < 1e-3, f"{problem.id}/{method}"


def test_trust_apls_trust_exact_diverges_where_the_true_hessian_is_indefinite_away_from_the_optimum():
    # A real, measured capability difference (not a bug): scipy's own
    # trust-exact converges cleanly on the exact same problems and start
    # (its trust-region safeguards handle an indefinite Hessian at the
    # starting point); trust's own engine does not, at least not here.
    # This pins the current, real behaviour so a future change to
    # trust's engine that fixes or worsens this is a deliberate test
    # change, not silent drift.
    results = sweep(backends=[SciPyBackend(), APLBackend()])

    for problem_id in ["noisy_expdec", "gaussian_peak"]:
        result = results[(problem_id, "trust-exact", "trust-apl")]
        assert result.status is RunStatus.MAX_ITER, problem_id
        assert result.dist_to_opt > 1.0, problem_id

    for problem_id in ["logistic", "michaelis_menten"]:
        result = results[(problem_id, "trust-exact", "trust-apl")]
        assert result.status is RunStatus.CONVERGED, problem_id
        assert result.dist_to_opt < 1e-3, problem_id
