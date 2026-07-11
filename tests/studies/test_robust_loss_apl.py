import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.studies.robust_loss import SCIPY_LOSSES, scipy_loss_precision

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


def test_trust_apls_lm_is_exercised_for_every_loss_instead_of_raising():
    precision = scipy_loss_precision(fractions=[0.0], backends=[SciPyBackend(), APLBackend()])

    for loss in SCIPY_LOSSES:
        distance = precision[(0.0, loss, "trust-apl")]
        assert distance < 1e-6, loss
