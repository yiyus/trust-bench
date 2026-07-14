import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from trust_bench.backends import BACKENDS
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture
from trust_bench.problems.families import outliers
from trust_bench.studies.robust_loss import (
    _F_SCALE_FOR_LOSS,
    FRACTIONS,
    SCIPY_LOSSES,
    TRUST_LOSSES,
    irls_precision,
    scipy_loss_precision,
    trust_loss_precision,
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


def test_scipy_declares_none_of_trusts_redescending_losses():
    # least_squares has no equivalent beyond cauchy/arctan; this is
    # already true by omission from _LEAST_SQUARES_LOSSES, checked
    # directly here so a future change can't silently reintroduce one
    # of these names into scipy's own declared vocabulary by mistake.
    losses = SciPyBackend().capabilities().methods["trf"].losses

    assert not (set(TRUST_LOSSES) & losses)


def test_trust_loss_precision_is_empty_when_no_backend_supports_it():
    # The default BACKENDS is scipy-only, which has no equivalent for
    # any of trust's own redescending losses; skipped, not raised,
    # matching scipy_loss_precision's own skip-guard for an unmapped
    # backend.
    precision = trust_loss_precision(fractions=[0.0])

    assert precision == {}


def test_trust_loss_precision_skips_a_backend_it_has_no_method_mapping_for():
    precision = trust_loss_precision(fractions=[0.0], backends=[_UnmappedBackend()])

    assert precision == {}


def test_f_scale_for_loss_matches_trusts_own_textbook_tuning_constants():
    # trust's Loss.apln (backends_ext/apl/trust/APLSource/Loss.apln)
    # bakes these in directly: huber=1.345, cauchy=2.385, softl1=1,
    # arctan=1 - the ~95%-asymptotic-efficiency constants, not scipy's
    # own default of 1.0 for every loss.
    assert _F_SCALE_FOR_LOSS == {"huber": 1.345, "cauchy": 2.385, "soft_l1": 1.0, "arctan": 1.0}


def test_scipy_loss_precision_passes_trusts_tuning_constant_as_f_scale():
    # Pinned against a direct scipy call using the same f_scale, for a
    # loss where trust's constant (1.345) actually differs from scipy's
    # own default (1.0) - soft_l1/arctan wouldn't distinguish "wired
    # correctly" from "coincidentally already matching".
    fraction = 0.2
    problem = outliers.make(fraction)
    x0 = np.asarray(problem.starts["standard"], dtype=float)

    precision = scipy_loss_precision(fractions=[fraction], losses=["huber"], backends=[SciPyBackend()])

    expected = scipy_least_squares(
        problem.residual, x0, jac=problem.jacobian, method="trf", loss="huber", f_scale=1.345, max_nfev=200
    )
    expected_distance = float(np.linalg.norm(expected.x - outliers.TRUE_PARAMETERS))

    assert precision[(fraction, "huber", "scipy")] == expected_distance


def test_outliers_true_parameters_are_the_fixed_underlying_trend_not_the_biased_l2_optimum():
    # Sanity check on the family itself (trust_bench.problems.families.
    # outliers): Problem.optima is the L2-biased fit to the corrupted
    # data, which moves away from TRUE_PARAMETERS as fraction grows; a
    # robust estimator's job is to recover TRUE_PARAMETERS instead,
    # which is why every test above compares against it.
    problem = outliers.make(_HIGH_CONTAMINATION)

    assert problem.optima[0].x_star.tolist() != outliers.TRUE_PARAMETERS.tolist()
