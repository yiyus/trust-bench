import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.studies.robust_loss import SCIPY_LOSSES, scipy_loss_precision, trust_loss_precision

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


def test_trust_apls_lm_is_exercised_for_every_loss_instead_of_raising():
    precision = scipy_loss_precision(fractions=[0.0], backends=[SciPyBackend(), APLBackend()])

    for loss in SCIPY_LOSSES:
        distance = precision[(0.0, loss, "trust-apl")]
        assert distance < 1e-6, loss


def test_trust_apls_welsch_recovers_true_parameters_up_to_a_high_contamination_level():
    # Warm-started from an arctan fit (trust_loss_precision's own
    # methodology, matching irls_tukey's documented reasoning): trust's
    # native Welsch loss then closely tracks the hand-rolled Tukey IRLS
    # reference's own behaviour - exact recovery well past the
    # contamination level where every scipy built-in loss already
    # fails (see test_irls_recovers_true_parameters_past_the_
    # contamination_level_where_every_scipy_loss_fails), only breaking
    # down near the 50% theoretical limit for a redescending
    # M-estimator.
    precision = trust_loss_precision(fractions=[0.0, 0.3, 0.45], backends=[APLBackend()])

    for fraction in [0.0, 0.3, 0.45]:
        distance = precision[(fraction, "welsch", "trust-apl")]
        assert distance < 1e-6, fraction


def test_trust_apls_welsch_breaks_down_near_the_theoretical_limit():
    precision = trust_loss_precision(fractions=[0.49], backends=[APLBackend()])

    assert precision[(0.49, "welsch", "trust-apl")] > 0.1
