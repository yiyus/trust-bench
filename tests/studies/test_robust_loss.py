from trust_bench.backends import BACKENDS
from trust_bench.problems.families import outliers
from trust_bench.studies.robust_loss import (
    FRACTIONS,
    SCIPY_LOSSES,
    irls_precision,
    scipy_loss_precision,
)

# The APL backend does not exist yet, so the hand-rolled IRLS reference
# below stands in for its redescending losses: it is the second
# comparison point against SciPy's built-in losses, not yet the third
# alongside a redescending APL implementation.

_HIGH_CONTAMINATION = 0.4


def test_scipy_loss_precision_is_recorded_across_the_fraction_and_loss_sweep():
    precision = scipy_loss_precision()

    assert len(precision) == len(FRACTIONS) * len(SCIPY_LOSSES) * len(BACKENDS)
    for (fraction, loss, backend_name), distance in precision.items():
        assert distance >= 0.0, f"fraction={fraction} loss={loss} on {backend_name}"


def test_all_scipy_losses_recover_the_true_parameters_with_no_contamination():
    precision = scipy_loss_precision(fractions=[0.0])

    for (fraction, loss, backend_name), distance in precision.items():
        assert distance < 1e-6, f"fraction={fraction} loss={loss} on {backend_name}"


def test_irls_precision_is_recorded_across_the_fraction_sweep():
    precision = irls_precision()

    assert len(precision) == len(FRACTIONS)
    for fraction, distance in precision.items():
        assert distance >= 0.0, f"fraction={fraction}"


def test_irls_recovers_true_parameters_past_the_contamination_level_where_every_scipy_loss_fails():
    scipy_distances = scipy_loss_precision(fractions=[_HIGH_CONTAMINATION])
    irls_distance = irls_precision(fractions=[_HIGH_CONTAMINATION])[_HIGH_CONTAMINATION]

    assert irls_distance < 1e-6
    for (fraction, loss, backend_name), distance in scipy_distances.items():
        assert distance > 0.1, f"loss={loss} on {backend_name} unexpectedly survived fraction={fraction}"


def test_outliers_true_parameters_are_the_fixed_underlying_trend_not_the_biased_l2_optimum():
    # Sanity check on the family itself (trust_bench.problems.families.
    # outliers): Problem.optima is the L2-biased fit to the corrupted
    # data, which moves away from TRUE_PARAMETERS as fraction grows; a
    # robust estimator's job is to recover TRUE_PARAMETERS instead,
    # which is why every test above compares against it.
    problem = outliers.make(_HIGH_CONTAMINATION)

    assert problem.optima[0].x_star.tolist() != outliers.TRUE_PARAMETERS.tolist()
