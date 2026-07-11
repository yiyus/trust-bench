import dataclasses
import shutil
import time

import numpy as np
import pytest

from trust_bench.backends.apl_backend import APLBackend, evaluate_problem
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems.families import dimensionality, ill_conditioned, large_residual, outliers, scaling

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()
START = "standard"

# One representative parameter value per family. scaling, large_residual
# and outliers match tests/problems/test_families.py's own parity-check
# values; ill_conditioned's kappa=100.0 matches ill_conditioning.py's own
# study KAPPAS list instead (that file's parity checks use 1e3, not 1e2).
DIFFICULTY_FAMILY_PROBLEMS = [
    scaling.make(s=10.0),
    ill_conditioned.make(kappa=100.0),
    large_residual.make(rho=10.0),
    outliers.make(fraction=0.3),
]


@pytest.mark.parametrize("problem", DIFFICULTY_FAMILY_PROBLEMS, ids=lambda p: p.id)
def test_solve_converges_to_the_known_optimum_instead_of_returning_error(problem):
    result = BACKEND.solve(problem, "lm", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED, result.status
    assert np.allclose(result.x_final, problem.optima[0].x_star, atol=1e-3)


def test_solve_reports_unknown_problem_id_for_an_unrecognised_parametrised_family():
    problem = dataclasses.replace(scaling.make(s=10.0), id="not_a_family(x=1.0)")

    result = BACKEND.solve(problem, "lm", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.ERROR


# dimensionality.py's own study only ever solves this family with its
# dense-Hessian methods (trust-exact, BFGS), never "lm" - matching that
# rather than reusing the "lm" convergence test above.
@pytest.mark.parametrize("method", ["trust-exact", "BFGS"])
def test_solve_dimensionality_converges_for_trust_exact_and_bfgs(method):
    problem = dimensionality.make(n=10)

    result = BACKEND.solve(problem, method, START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED, result.status
    assert np.allclose(result.x_final, problem.optima[0].x_star, atol=1e-3)


def test_solve_trust_exact_converges_at_n_1000():
    # trust-exact's per-iteration cost stays practical even at the
    # largest n the study sweeps (measured directly: ~13s total, well
    # under the harness's own subprocess timeout).
    problem = dimensionality.make(n=1000)

    result = BACKEND.solve(problem, "trust-exact", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED, result.status
    assert np.allclose(result.x_final, problem.optima[0].x_star, atol=1e-3)


def test_solve_bfgs_completes_without_a_timeout_at_a_practical_dimension():
    # BFGS's per-iteration cost, unlike trust-exact's, grows too steep
    # with n to finish a full 200-iteration budget at n=1000 within the
    # harness's subprocess timeout (measured directly: ~1.4s/iteration
    # there, versus ~0.04s/iteration at n=100) - a real limitation of
    # trust's own BFGS engine at that scale, not of this port. n=100
    # (also swept by the study) is the largest size this asserts a full
    # solve completes at; MAX_ITER, not convergence, is the scientifically
    # expected outcome here (a dense quasi-Newton method genuinely fails
    # to converge at this dimension - the study's own point).
    problem = dimensionality.make(n=100)

    result = BACKEND.solve(problem, "BFGS", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.MAX_ITER, result.status


def test_evaluate_at_n_1000_completes_quickly():
    # The port must not become the dominant cost of running the study at
    # the largest n it sweeps; a generous bound that would still fail
    # outright on an accidentally quadratic-in-n-squared construction.
    problem = dimensionality.make(n=1000)
    start = time.perf_counter()
    evaluate_problem(problem.id, problem.probe_points[0])
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0, elapsed
