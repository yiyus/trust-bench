from trust_bench.backends import BACKENDS
from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture
from trust_bench.problems.families import outliers
from trust_bench.studies.robust_loss import (
    FRACTIONS,
    SCIPY_LOSSES,
    irls_precision,
    scipy_loss_precision,
)

_HIGH_CONTAMINATION = 0.4


class _UnmappedBackend(Backend):
    """A backend name scipy_loss_precision has no method mapping for.
    Proves it is skipped rather than raising, mirroring bounded.py's
    own skip-guard pattern for a backend/method combination the study
    doesn't know how to drive.
    """

    name = "unmapped-backend"

    def capabilities(self):
        return Capabilities(
            methods={
                "trf": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(SCIPY_LOSSES),
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                )
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        raise AssertionError(f"{self.name} should have been skipped, not solved")


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


def test_scipy_loss_precision_skips_a_backend_it_has_no_method_mapping_for():
    precision = scipy_loss_precision(fractions=[0.0], backends=[_UnmappedBackend()])

    assert precision == {}


def test_outliers_true_parameters_are_the_fixed_underlying_trend_not_the_biased_l2_optimum():
    # Sanity check on the family itself (trust_bench.problems.families.
    # outliers): Problem.optima is the L2-biased fit to the corrupted
    # data, which moves away from TRUE_PARAMETERS as fraction grows; a
    # robust estimator's job is to recover TRUE_PARAMETERS instead,
    # which is why every test above compares against it.
    problem = outliers.make(_HIGH_CONTAMINATION)

    assert problem.optima[0].x_star.tolist() != outliers.TRUE_PARAMETERS.tolist()
