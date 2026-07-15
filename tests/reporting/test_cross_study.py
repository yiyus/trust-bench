import shutil

import pandas as pd
import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.reporting import cross_study
from trust_bench.reporting.cross_study import _pivot_by_backend, frontier_panels, parity_frame

SCIPY = SciPyBackend()


@pytest.fixture
def small_dimensionality_panel(monkeypatch):
    # frontier_panels' own dimensionality panel solves BFGS (a dense
    # method) at n=1000 by default - the single largest cost in the
    # test suite (~110s per call). A test proving structural/display
    # behaviour doesn't need that scale: n=10 already gives a genuine
    # BFGS solve to build a real panel from, just fast.
    original_sweep = cross_study.dimensionality.sweep
    monkeypatch.setattr(
        cross_study.dimensionality,
        "sweep",
        lambda methods, backends: original_sweep(n_values=[10], methods=methods, backends=backends),
    )


@pytest.fixture
def long_df():
    return pd.DataFrame(
        [
            dict(problem_id="p1", backend="scipy", status="CONVERGED", dist_to_opt=1e-9),
            dict(problem_id="p1", backend="trust-apl", status="CONVERGED", dist_to_opt=1e-7),
            dict(problem_id="p2", backend="scipy", status="CONVERGED", dist_to_opt=1e-8),
            # trust-apl has no row at all for p2: dropped, not a KeyError.
        ]
    )


def test_pivot_by_backend_produces_one_row_per_shared_instance(long_df):
    wide = _pivot_by_backend(long_df, ["problem_id"], [SCIPY, APLBackend()])

    assert list(wide["problem_id"]) == ["p1"]
    assert wide["dist_to_opt_scipy"].iloc[0] == 1e-9
    assert wide["dist_to_opt_trust-apl"].iloc[0] == 1e-7


def test_pivot_by_backend_rejects_anything_but_exactly_two_backends(long_df):
    with pytest.raises(ValueError, match="two backends"):
        _pivot_by_backend(long_df, ["problem_id"], [SCIPY])


def test_pivot_by_backend_treats_an_unsupported_row_the_same_as_an_error_row():
    # A backend whose only rows are declared-unsupported rejections
    # (results_to_dataframe's "UNSUPPORTED" label) has no genuine
    # comparison point, exactly like one whose only rows are "ERROR" -
    # the missing-backend guard must catch both, not just the latter.
    df = pd.DataFrame(
        [
            dict(problem_id="p1", backend="scipy", status="CONVERGED", dist_to_opt=1e-9),
            dict(problem_id="p1", backend="trust-apl", status="UNSUPPORTED", dist_to_opt=None),
        ]
    )

    with pytest.raises(ValueError, match="no results for backend"):
        _pivot_by_backend(df, ["problem_id"], [SCIPY, APLBackend()])


def test_parity_frame_rejects_anything_but_exactly_two_backends():
    with pytest.raises(ValueError, match="two backends"):
        parity_frame(backends=[SCIPY])


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_parity_frame_pools_baseline_typical_and_bounded():
    df = parity_frame(backends=[SCIPY, APLBackend()])

    assert set(df["study"]) == {"baseline", "typical", "bounded"}
    assert {"dist_to_opt_scipy", "dist_to_opt_trust-apl", "converged"} <= set(df.columns)
    assert len(df) > 0


def test_frontier_panels_covers_every_difficulty_sweep(small_dimensionality_panel):
    panels = frontier_panels(backends=[SCIPY])

    assert set(panels) == {"ill_conditioning", "scaling", "dimensionality", "large_residual", "robust_loss"}
    for df, x, y in panels.values():
        assert len(df) > 0
        assert {x, y, "backend"} <= set(df.columns)


def test_frontier_panels_floors_an_exact_zero_metric_for_log_display(small_dimensionality_panel):
    panels = frontier_panels(backends=[SCIPY])

    scaling_df, _, y = panels["scaling"]
    assert (scaling_df[y] > 0).all()
