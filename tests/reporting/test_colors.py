from trust_bench.reporting.colors import BACKEND_COLORS, STATUS_COLORS, backend_color, status_color


def test_every_registered_backend_has_a_fixed_colour():
    assert set(BACKEND_COLORS) == {"scipy", "trust-apl"}


def test_every_run_status_name_has_a_registered_colour_entry():
    from trust_bench.core.result import RunStatus

    assert {status.value for status in RunStatus} <= set(STATUS_COLORS)


def test_backend_color_is_consistent_and_distinct():
    assert backend_color("scipy") == BACKEND_COLORS["scipy"]
    assert backend_color("trust-apl") == BACKEND_COLORS["trust-apl"]
    assert backend_color("scipy") != backend_color("trust-apl")


def test_backend_color_falls_back_for_an_unregistered_name():
    # A future backend shouldn't crash report generation just because
    # colors.py doesn't know it yet - a neutral fallback, not a KeyError.
    assert backend_color("some-future-backend") is not None


def test_status_color_distinguishes_converged_from_every_failure_mode():
    assert status_color("CONVERGED") != status_color("FAILED")
    assert status_color("CONVERGED") != status_color("ERROR")
    assert status_color("CONVERGED") != status_color("STALLED")
    assert status_color("CONVERGED") != status_color("MAX_ITER")


def test_status_color_treats_unsupported_as_neutral_not_a_failure():
    # UNSUPPORTED is a declared-unsupported rejection, an expected
    # outcome of a study's own probe - it should not read as a failure
    # (red) alongside a genuine FAILED/ERROR.
    assert status_color("UNSUPPORTED") != status_color("FAILED")
    assert status_color("UNSUPPORTED") != status_color("ERROR")
    assert status_color("UNSUPPORTED") != status_color("CONVERGED")


def test_every_known_run_status_has_a_registered_colour():
    from trust_bench.core.result import RunStatus

    for status in RunStatus:
        assert status_color(status.value) is not None
