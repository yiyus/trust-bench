import math
import warnings

from trust_bench.core.metrics import basin_rate, mad, order, rate


def test_order_estimates_two_for_a_quadratically_convergent_sequence():
    errors = [0.9]
    for _ in range(20):
        errors.append(errors[-1] ** 2)

    assert math.isclose(order(errors), 2.0, abs_tol=1e-6)


def test_rate_estimates_r_for_a_linearly_convergent_sequence():
    r = 0.7
    errors = [1.0]
    for _ in range(60):
        errors.append(errors[-1] * r)

    assert math.isclose(rate(errors), r, abs_tol=1e-6)


def test_order_returns_nan_when_too_few_points_survive_the_floor():
    errors = [1e-3]
    for _ in range(10):
        errors.append(errors[-1] ** 2)

    assert math.isnan(order(errors))


def test_order_ignores_a_plateau_in_the_error_sequence():
    errors = [1.0, 0.5, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.015]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate = order(errors)

    assert math.isclose(estimate, 1.0, abs_tol=1e-3)


def test_order_returns_nan_when_plateaus_leave_too_few_valid_estimates():
    errors = [1.0, 0.5, 0.5, 0.5]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate = order(errors)

    assert math.isnan(estimate)


def test_basin_rate_matches_a_hand_computed_fraction_on_a_fixture_set():
    distances_to_opt = [1e-9, 0.3, 1e-10, None, 2.5]

    assert math.isclose(basin_rate(distances_to_opt, tol=1e-6), 0.4)


def test_basin_rate_returns_nan_for_an_empty_set_of_starts():
    assert math.isnan(basin_rate([], tol=1e-6))


def test_mad_matches_a_hand_computed_value():
    # median=3.0; absolute deviations=[2,1,0,1,97]; median of those=1.0;
    # scaled by 1.4826 (the normal-consistent constant this project
    # already uses elsewhere, e.g. robust_loss.py's irls_tukey) for a
    # MAD directly comparable to a standard deviation.
    values = [1.0, 2.0, 3.0, 4.0, 100.0]

    assert math.isclose(mad(values), 1.4826, abs_tol=1e-6)


def test_mad_is_zero_for_identical_values():
    assert mad([5.0, 5.0, 5.0]) == 0.0
