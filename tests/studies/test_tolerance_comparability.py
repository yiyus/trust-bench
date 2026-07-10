import shutil

import pytest

from trust_bench.backends import BACKENDS
from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.problems import CANONICAL_PROBLEMS
from trust_bench.studies.tolerance_comparability import TOLERANCE, sweep


def test_sweep_covers_every_canonical_problem_and_backend():
    results = sweep()

    assert len(results) == len(CANONICAL_PROBLEMS) * len(BACKENDS)


def test_sweep_uses_the_same_explicit_tolerance_for_every_backend():
    results = sweep()

    for (problem_id, backend_name), result in results.items():
        assert result.config.tolerance == TOLERANCE, f"{problem_id} on {backend_name}"


def test_sweep_reports_dist_to_opt_and_grad_norm_final_for_every_result():
    results = sweep()

    for (problem_id, backend_name), result in results.items():
        assert result.dist_to_opt is not None, f"{problem_id} on {backend_name}"
        assert result.grad_norm_final is not None, f"{problem_id} on {backend_name}"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_sweep_surfaces_the_precision_gap_an_equal_tolerance_does_not_remove():
    # The confound #86 exists to make visible, not to eliminate: scipy's
    # "lm" reaches machine-zero on rosenbrock regardless of the intent-level
    # tolerance passed, while trust-apl's stall-detection convergence test
    # stops well short of it, because the two libraries' tolerance
    # parameters are not semantically equivalent stopping criteria (see
    # docs/methodology.md). Both are expected to converge; their precision
    # is not expected to match at the same nominal tolerance value.
    results = sweep(backends=[SciPyBackend(), APLBackend()])

    scipy_result = results[("rosenbrock", "scipy")]
    apl_result = results[("rosenbrock", "trust-apl")]

    assert scipy_result.dist_to_opt < 1e-9
    assert apl_result.dist_to_opt > 1e-8
