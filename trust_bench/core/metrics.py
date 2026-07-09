import numpy as np

_MIN_LOG_RATIO = 1e-14


def order(errors, floor=1e-12):
    """Estimate empirical convergence order from an error sequence.

    p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1}), median over the tail above
    `floor`. A plateau (two consecutive equal errors) makes some `log(e_k/
    e_{k-1})` zero; those steps are excluded rather than divided by, since a
    near-zero denominator makes the ratio meaningless, not merely noisy.
    NaN when too few points survive the floor, or too few order estimates
    survive the denominator guard, to fit a reliable order.
    """
    e = np.asarray(errors, dtype=float)
    e = e[e > floor]
    if len(e) < 4:
        return float("nan")
    log_ratios = np.log(e[1:] / e[:-1])
    numerators, denominators = log_ratios[1:], log_ratios[:-1]
    valid = np.abs(denominators) > _MIN_LOG_RATIO
    if valid.sum() < 2:
        return float("nan")
    return float(np.median(numerators[valid] / denominators[valid]))


def rate(errors, floor=1e-12):
    """Estimate empirical linear convergence rate from an error sequence.

    r_k = e_{k+1}/e_k, median over the tail above `floor`. NaN when too few
    points survive the floor to fit a reliable rate.
    """
    e = np.asarray(errors, dtype=float)
    e = e[e > floor]
    if len(e) < 4:
        return float("nan")
    ratios = e[1:] / e[:-1]
    return float(np.median(ratios))
