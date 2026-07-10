import dataclasses

from trust_bench.backends import BACKENDS
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.backend import Backend, Capabilities
from trust_bench.core.provenance import capture
from trust_bench.reporting.capability_matrix import FIELDS, derive_matrix

_METHOD_COUNT = len(SciPyBackend().capabilities().methods)


def test_derive_matrix_covers_every_backend_method_and_field():
    df = derive_matrix()

    assert len(df) == len(BACKENDS) * _METHOD_COUNT * len(FIELDS)
    for column in ["backend", "method", "field", "declared", "measured", "agrees"]:
        assert column in df.columns


def test_declared_and_measured_bounds_agree_for_every_method():
    df = derive_matrix()

    bounds_rows = df[df["field"] == "bounds"]
    assert len(bounds_rows) > 0
    assert bounds_rows["agrees"].all()


def test_declared_and_measured_analytic_hessian_agree_for_every_method():
    df = derive_matrix()

    hessian_rows = df[df["field"] == "analytic_hessian"]
    assert len(hessian_rows) > 0
    assert hessian_rows["agrees"].all()


class _LyingAboutBoundsBackend(Backend):
    """Wraps SciPyBackend's lm but declares bounds=True: a lie, since lm
    genuinely rejects bounds. Proves derive_matrix's agreement check
    catches a declared/measured mismatch, not just a match.
    """

    name = "lying-backend"

    def __init__(self):
        self._scipy = SciPyBackend()

    def capabilities(self):
        lm = self._scipy.capabilities().methods["lm"]
        return Capabilities(methods={"lm": dataclasses.replace(lm, bounds=True)})

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        return self._scipy.solve(problem, method, start, config)


def test_derive_matrix_flags_a_declared_measured_mismatch():
    df = derive_matrix(backends=[_LyingAboutBoundsBackend()], fields=["bounds"])

    row = df.iloc[0]
    assert row["declared"] is True
    assert row["measured"] is False
    assert row["agrees"] is False
