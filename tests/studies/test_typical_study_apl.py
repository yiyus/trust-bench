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


def test_trust_apls_trust_exact_converges_despite_an_indefinite_hessian_away_from_the_optimum():
    # noisy_expdec/gaussian_peak's true hessian is indefinite at the
    # standard start (nonzero residual even at x_star; see
    # docs/methodology.md). trust's Newton.aplo engine used to diverge
    # there (MAX_ITER) until it started rejecting any step whose
    # quadratic model predicts a negative error decrement (vendored
    # trust commit 05f9010) - this pins the fixed behaviour so a
    # regression back to the old failure mode is caught, not silently
    # reintroduced.
    results = sweep(backends=[SciPyBackend(), APLBackend()])

    for problem in TYPICAL_PROBLEMS:
        result = results[(problem.id, "trust-exact", "trust-apl")]
        assert result.status is RunStatus.CONVERGED, problem.id
        assert result.dist_to_opt < 1e-3, problem.id
