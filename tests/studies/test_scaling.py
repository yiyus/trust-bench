from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunStatus
from trust_bench.studies.scaling import METHODS, SCALES, X_SCALES, sweep

_LARGE_DISPARITY = 1e6
_EXTREME_DISPARITY = 1e8


def test_sweep_covers_every_scale_method_x_scale_and_backend():
    results = sweep()

    assert len(results) == len(SCALES) * len(METHODS) * len(X_SCALES) * len(BACKENDS)


def test_unscaled_default_converges_exactly_across_the_full_scale_sweep():
    results = sweep(x_scales=[None])

    for (scale, method, x_scale, backend_name), result in results.items():
        assert result.status is RunStatus.CONVERGED, f"{scale}/{method}/{backend_name}"
        assert result.dist_to_opt < 1e-6, f"{scale}/{method}/{backend_name}"


def test_jac_scaling_needs_more_function_evaluations_at_large_scale_disparity():
    # Section 9 item 6 asks for "with and without" x_scale, not that
    # scaling helps: this problem family's Hessian is diagonal, so an
    # unscaled Gauss-Newton step already accounts for the scale
    # disparity exactly, and adding SciPy's adaptive x_scale='jac' on
    # top costs extra function evaluations rather than saving any.
    results = sweep(scales=[_LARGE_DISPARITY])

    for method in METHODS:
        for backend in BACKENDS:
            unscaled = results[(_LARGE_DISPARITY, method, None, backend.name)]
            jac_scaled = results[(_LARGE_DISPARITY, method, "jac", backend.name)]
            assert jac_scaled.n_feval > unscaled.n_feval, f"{method}/{backend.name}"


def test_jac_scaling_can_converge_to_the_wrong_answer_at_extreme_scale_disparity():
    # A real, disclosed finding, not a hypothetical: at scale=1e8,
    # trf/dogbox with x_scale='jac' report RunStatus.CONVERGED (scipy's
    # own success flag is True) while sitting far from the optimum.
    # Status alone does not tell this story; dist_to_opt does.
    results = sweep(scales=[_EXTREME_DISPARITY], methods=["trf", "dogbox"])

    for method in ["trf", "dogbox"]:
        for backend in BACKENDS:
            jac_scaled = results[(_EXTREME_DISPARITY, method, "jac", backend.name)]
            unscaled = results[(_EXTREME_DISPARITY, method, None, backend.name)]
            assert jac_scaled.status is RunStatus.CONVERGED, f"{method}/{backend.name}"
            assert jac_scaled.dist_to_opt > 1.0, f"{method}/{backend.name}"
            assert unscaled.dist_to_opt < 1e-6, f"{method}/{backend.name}"
