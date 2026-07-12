import numpy as np
import pytest
from invariants import assert_known_optimum, assert_parity

from trust_bench.problems import TYPICAL_PROBLEMS as PROBLEMS


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_analytic_jacobian_and_hessian_match_finite_differences_at_every_probe_point(problem):
    assert_parity(problem, jacobian_tol=dict(atol=1e-6), hessian_tol=dict(atol=1e-2))


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_known_optimum_satisfies_gradient_and_cost_invariants(problem):
    # A looser bound than test_canonical.py's own 1e-8: these optima come
    # from a numerical solve against noisy data, not a closed form, and
    # logistic's parameter scale (D ~ 100) leaves a somewhat larger
    # absolute gradient floor than the trivial MGH problems have.
    assert_known_optimum(problem, atol=1e-4)


_INDEFINITE_AT_START = {"noisy_expdec", "gaussian_peak"}
_PSD_AT_START = {"logistic", "michaelis_menten"}


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_hessian_indefiniteness_at_the_standard_start_matches_trust_exacts_known_fragility(problem):
    # noisy_expdec/gaussian_peak's nonzero residual at x_star gives their
    # true hessian a residual-curvature correction term large enough to
    # flip an eigenvalue's sign at the standard start - the mechanism
    # behind trust-exact's MAX_ITER failure there (see
    # docs/methodology.md). logistic/michaelis_menten's canonical-link
    # likelihoods keep an always-PSD hessian, and trust-exact converges
    # cleanly on both.
    mineig = np.linalg.eigvalsh(problem.hessian(problem.starts["standard"])).min()
    if problem.id in _INDEFINITE_AT_START:
        assert mineig < 0, problem.id
    elif problem.id in _PSD_AT_START:
        assert mineig > 0, problem.id
