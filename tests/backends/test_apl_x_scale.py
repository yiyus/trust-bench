import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems.families import scaling

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()
START = "standard"


@pytest.mark.parametrize("method", ["lm", "BFGS", "trust-exact"])
def test_every_method_rejects_adaptive_x_scale(method):
    # No equivalent of scipy's x_scale="jac" (recomputed every
    # iteration from the current Jacobian): Newton.aplo's own damping
    # by diag(H) already gives an adaptive-scaling effect internally.
    # A blanket rejection, not lm-specific - trust-exact's own fixed
    # rejection below (a different reason) would mask this one, so
    # this is the only test exercising it there.
    problem = scaling.make(1.0)

    with pytest.raises(ValueError, match="x_scale"):
        BACKEND.solve(problem, method, START, RunConfig(max_iter=200, x_scale="jac"))


def test_trust_exact_rejects_a_fixed_x_scale():
    # trust's pscale wrapper only rescales a 2-item (value, derivative)
    # return (lm's (residual, jacobian), BFGS's (cost, gradient)); a
    # Hessian needs outer-product scaling on both axes, not a column
    # scale, so silently applying pscale there would be a silently
    # wrong answer rather than an unsupported one.
    problem = scaling.make(1.0)

    with pytest.raises(ValueError, match="x_scale"):
        BACKEND.solve(problem, "trust-exact", START, RunConfig(max_iter=200, x_scale=(1.0, 1.0)))


@pytest.mark.parametrize("method", ["lm", "BFGS"])
def test_a_fixed_x_scale_recovers_where_unscaled_fails_at_extreme_disparity(method):
    # A real, previously-unfixable capability gap: at scale=1e8,
    # trust-apl's lm reports FAILED with cost_final~2e16 unscaled.
    # Confirmed directly (matching the vendored trust bump's own
    # motivation): a fixed reparameterisation matching this problem's
    # own known anisotropy fixes it exactly, cost_final dropping to
    # ~1.9e-12.
    scale = 1e8
    problem = scaling.make(scale)
    config = RunConfig(max_iter=200, x_scale=(1.0, 1.0 / scale))

    result = BACKEND.solve(problem, method, START, config)

    assert result.status is RunStatus.CONVERGED, result.status
    assert result.dist_to_opt < 1e-6
