from trust_bench.problems import (
    beale,
    cauchy_mle,
    expdec,
    gaussian_peak,
    helical,
    linear,
    logistic,
    michaelis_menten,
    noisy_expdec,
    powell,
    quadratic,
    rastrigin,
    rosenbrock,
)

CANONICAL_PROBLEMS = [
    rosenbrock.PROBLEM,
    beale.PROBLEM,
    powell.PROBLEM,
    helical.PROBLEM,
    expdec.PROBLEM,
    quadratic.PROBLEM,
    linear.PROBLEM,
]

# Smooth, moderately-conditioned nonlinear-regression/MLE problems from
# ordinary curve-fitting practice, chosen to be typical rather than to
# stress a specific failure mode (unlike CANONICAL_PROBLEMS' own MGH
# test functions, each built specifically to be hard for a solver).
TYPICAL_PROBLEMS = [
    noisy_expdec.PROBLEM,
    logistic.PROBLEM,
    michaelis_menten.PROBLEM,
    gaussian_peak.PROBLEM,
]

# Genuine black-box scalar objectives (kind="scalar", jacobian=None):
# no residual vector to exploit, unlike every problem above, all of
# which are nonlinear-least-squares fits underneath even when solved
# via BFGS. This is BFGS's actual niche.
SCALAR_PROBLEMS = [
    rastrigin.PROBLEM,
    cauchy_mle.PROBLEM,
]
