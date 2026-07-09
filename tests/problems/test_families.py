import numpy as np
import pytest
from invariants import assert_known_optimum, assert_parity

from trust_bench.problems.families import dimensionality, ill_conditioned, large_residual, outliers, scaling


def _increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


# Families sweep parameters across many orders of magnitude (ill_conditioned,
# scaling), unlike the canonical problems' O(1) scale, so parity needs a
# relative tolerance alongside the absolute one to stay meaningful at both
# ends of the sweep.
_JACOBIAN_TOL = dict(rtol=1e-4, atol=1e-6)
_HESSIAN_TOL = dict(rtol=1e-3, atol=1e-2)


# ---------------------------------------------------------------------------
# large_residual
# ---------------------------------------------------------------------------
LARGE_RESIDUAL_RHOS = [0.0, 1.0, 5.0, 10.0, 30.0]


@pytest.mark.parametrize("rho", LARGE_RESIDUAL_RHOS)
def test_large_residual_make_produces_a_parity_passing_problem(rho):
    problem = large_residual.make(rho=rho)
    assert_parity(problem, jacobian_tol=_JACOBIAN_TOL, hessian_tol=_HESSIAN_TOL)
    assert_known_optimum(problem, atol=1e-4)


def test_large_residual_irreducible_residual_norm_grows_with_rho():
    norms = []
    for rho in LARGE_RESIDUAL_RHOS:
        problem = large_residual.make(rho=rho)
        x_star = problem.optima[0].x_star
        norms.append(np.linalg.norm(problem.residual(x_star)))
    assert _increasing(norms)


# ---------------------------------------------------------------------------
# ill_conditioned
# ---------------------------------------------------------------------------
ILL_CONDITIONED_KAPPAS = [1.0, 10.0, 1e3, 1e6]


@pytest.mark.parametrize("kappa", ILL_CONDITIONED_KAPPAS)
def test_ill_conditioned_make_produces_a_parity_passing_problem(kappa):
    problem = ill_conditioned.make(kappa=kappa)
    assert_parity(problem, jacobian_tol=_JACOBIAN_TOL, hessian_tol=_HESSIAN_TOL)
    assert_known_optimum(problem)


def test_ill_conditioned_condition_number_grows_with_kappa_and_matches_it():
    conds = []
    for kappa in ILL_CONDITIONED_KAPPAS:
        problem = ill_conditioned.make(kappa=kappa)
        conds.append(np.linalg.cond(problem.jacobian(problem.optima[0].x_star)))
    assert _increasing(conds)
    for kappa, cond in zip(ILL_CONDITIONED_KAPPAS, conds):
        assert np.isclose(cond, kappa, rtol=1e-6)


def test_ill_conditioned_rejects_a_condition_number_below_one():
    with pytest.raises(ValueError):
        ill_conditioned.make(kappa=0.5)


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------
OUTLIER_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize("fraction", OUTLIER_FRACTIONS)
def test_outliers_make_produces_a_parity_passing_problem(fraction):
    problem = outliers.make(fraction=fraction)
    assert_parity(problem, jacobian_tol=_JACOBIAN_TOL, hessian_tol=_HESSIAN_TOL)
    assert_known_optimum(problem)


def test_outliers_fit_distance_from_truth_grows_with_fraction():
    distances = []
    for fraction in OUTLIER_FRACTIONS:
        problem = outliers.make(fraction=fraction)
        x_star = problem.optima[0].x_star
        distances.append(np.linalg.norm(x_star - outliers.TRUE_PARAMETERS))
    assert _increasing(distances)


def test_outliers_rejects_a_majority_contamination_fraction():
    with pytest.raises(ValueError):
        outliers.make(fraction=0.5)


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------
SCALING_S_PARITY = [1.0, 10.0, 100.0, 1000.0]
SCALING_S_WIDE = [1.0, 10.0, 100.0, 1000.0, 1e6]


@pytest.mark.parametrize("s", SCALING_S_PARITY)
def test_scaling_make_produces_a_parity_passing_problem(s):
    problem = scaling.make(s=s)
    assert_parity(problem, jacobian_tol=_JACOBIAN_TOL, hessian_tol=_HESSIAN_TOL)
    assert_known_optimum(problem)


def test_scaling_hessian_diagonal_ratio_grows_with_s_and_matches_s_squared():
    ratios = []
    for s in SCALING_S_WIDE:
        problem = scaling.make(s=s)
        h = problem.hessian(problem.optima[0].x_star)
        ratios.append(h[1, 1] / h[0, 0])
    assert _increasing(ratios)
    for s, ratio in zip(SCALING_S_WIDE, ratios):
        assert np.isclose(ratio, s**2, rtol=1e-6)


# ---------------------------------------------------------------------------
# dimensionality
# ---------------------------------------------------------------------------
DIMENSIONALITY_NS_PARITY = [2, 10]
DIMENSIONALITY_NS_WIDE = [2, 10, 100, 1000]


@pytest.mark.parametrize("n", DIMENSIONALITY_NS_PARITY)
def test_dimensionality_make_produces_a_parity_passing_problem(n):
    problem = dimensionality.make(n=n)
    assert_parity(problem, jacobian_tol=_JACOBIAN_TOL, hessian_tol=_HESSIAN_TOL)
    assert_known_optimum(problem)


@pytest.mark.parametrize("n", DIMENSIONALITY_NS_WIDE)
def test_dimensionality_known_optimum_holds_at_every_swept_size(n):
    problem = dimensionality.make(n=n)
    assert_known_optimum(problem)


def test_dimensionality_grows_with_n():
    sizes = [len(dimensionality.make(n=n).optima[0].x_star) for n in DIMENSIONALITY_NS_WIDE]
    assert sizes == DIMENSIONALITY_NS_WIDE
    assert _increasing(sizes)


def test_dimensionality_rejects_an_odd_n():
    with pytest.raises(ValueError):
        dimensionality.make(n=3)
