# Comparability methodology

Decisions that keep a cross-backend comparison meaningful, enforced by
the runner/config layer and tested per backend. See Section 7 of
`docs/plans/trust-bench.md` for the full list this document expands.

## Tolerance mapping

`RunConfig.tolerance` is a single intent-level value. Each backend maps
it onto its own native stopping parameters; leaving it `None` means
the backend falls back to its own native default rather than any value
this project chooses, and the two defaults are not comparable (see
below).

### scipy

`least_squares` (`lm`/`trf`/`dogbox`) exposes three independent stopping
criteria - `ftol` (cost change), `xtol` (step size), `gtol` (gradient
projection) - applied simultaneously. A single `RunConfig.tolerance`
value is written to all three (`trust_bench/backends/scipy_backend.py`).
Left `None`, scipy's own default is `ftol=xtol=gtol=1e-8`.

`minimize` methods each expose a different subset of `{ftol, xtol,
gtol}` (`_MINIMIZE_TOLERANCE_PARAMS` in the same file, derived from
`scipy.optimize.show_options`); a single `RunConfig.tolerance` value is
applied to every parameter the chosen method accepts.

### trust-apl

`trust`'s Newton-region solver (`Newton.aplo`) does not expose separate
step/cost/gradient thresholds. Its convergence check (`Newton.aplo`'s
`C` function) stops when either:

- the cost itself drops below an absolute threshold (`tc`), or
- a relative change metric `r` - the minimum of relative parameter-space
  change and relative cost change between the last two accepted
  iterates (`Newton.aplo`'s `T` function) - drops below a relative
  threshold (`tr`), once `r` is positive (a negative `r` means the step
  made things worse, not converged).

`solve.dyalog` maps a single `RunConfig.tolerance` value onto both `tc`
(`cfg.tolc`) and `tr` (`cfg.tolr`) uniformly. Left `None`, both fall
back to `⎕CT` (Dyalog's comparison tolerance, measured `1e-14` in this
environment).

### Why an equal numeric value is not an equal test

scipy's thresholds are independent, absolute-scale comparisons against
the current step/cost/gradient. `trust`'s `r` is a relative,
stall-detection metric computed from consecutive accepted iterates, not
an absolute distance to a target. The same numeric `tolerance` value
therefore does not exercise the same stopping condition in each backend.

Confirmed directly (`rosenbrock.PROBLEM`, method `"lm"`, `tolerance`
from `1e-2` to `1e-14`): scipy reaches `dist_to_opt == 0.0` at every
value in that range, unaffected by tightening it further. `trust-apl`
degrades smoothly as `tolerance` loosens (`0.078` at `1e-2`, `6.2e-5` at
`1e-6`) and plateaus at a floor of `1.55e-7` from `1e-10` onward (the
`⎕CT` floor its own iteration cannot resolve below). A precision or
iteration-count difference measured under an equal `RunConfig.tolerance`
is not, by itself, evidence of a capability difference between the two
libraries: it may equally reflect this mapping mismatch. Section 9's
capability studies that report precision across backends should be read
with this in mind; `trust_bench.studies.tolerance_comparability` exists
to make the comparison explicit rather than leave it as an unstated
side effect of `tolerance=None`.

## Status classification: CONVERGED vs STALLED

`trust`'s own API never asserts convergence: `Newton.aplo` returns
`iter`/`cost`/`rel`/`dnorm`/`p` and leaves interpretation to the caller.
Of the two conditions in `Newton.aplo`'s own termination check that
aren't iteration-limit or damping-saturation (`FAILED`/`MAX_ITER`), only
one is a genuine near-optimality guarantee: the cost dropping below an
absolute threshold (`tc>c`). The other - the relative-change metric `r`
stalling (`(r>0)∧tr>r`) - only means the last accepted step stopped
changing much; on a problem whose residual doesn't vanish at the true
optimum (e.g. the `large_residual`/`outliers` families, where cost is
large by construction even at `x_star`), it is the *only* criterion that
ever fires, and does so at genuine convergence. At extreme conditioning
in `ill_conditioning` (high `kappa`, both `BFGS` and `trust-exact`), the
same criterion can fire from a point nowhere near the optimum, once
damping has grown large without quite crossing the `FAILED` threshold -
not because any Hessian involved is indefinite (`ill_conditioning`'s is
never indefinite; see the capability boundary sections below), but
because solving the damped-Newton system itself loses numerical
precision at that condition number.

`solve.dyalog` (`backends_ext/apl/solve.dyalog`) resolves this by adding
a second, harness-level near-optimality signal: the final gradient norm
(`grad_norm_final`, already computed for every method regardless of
whether it drove the search). A relative-change stall is reported as
`CONVERGED` only when the cost criterion holds (`r.cost<cfg.tolc`) or
the gradient norm is small (`<1e-2`); otherwise it is reported as the
distinct `RunStatus.STALLED`. The `1e-2` threshold is a harness
heuristic, not a value `trust` itself uses anywhere: measured directly,
genuine convergences (including the large-residual case above) leave a
gradient norm at or below `1.6e-6`, while the indefinite-Hessian stalls
this distinguishes leave one at `4.8` or higher - six orders of
magnitude of margin either side of `1e-2` across every case measured so
far.

This gradient-norm check does not apply to a bounded request
(`req.bounds` set): at an active-boundary optimum, the unconstrained
gradient is genuinely nonzero by construction (measured, `quadratic`'s
`active_at_boundary` scenario: `grad_norm_final≈0.5`), not a stall.
Distinguishing a real stall from a genuine bounded convergence needs a
KKT-aware (projected) gradient check this harness does not compute. A
bounded request's relative-change stall is reported `CONVERGED`
unconditionally, exactly as before this fix - a known limitation, not a
claim this fix resolves bounded near-optimality detection.

## Capability boundary: trust-apl's BFGS under ill-conditioning and scale

`trust-apl`'s BFGS engine is measurably less robust to ill-conditioning
and dimensionality than scipy's BFGS, confirmed directly by sweeping
both. In both cases the underlying Hessian involved is never indefinite
(`ill_conditioning`'s is the constant, positive-definite `a.T@a`;
`dimensionality`'s is discussed below) - this is a numerical-precision
boundary, not the indefinite-Hessian one the next section covers.

- `ill_conditioning` (`kappa` from `1` to `1e8`): `trust-apl BFGS`'s cost
  at the reported solution grows from machine precision at `kappa≤1e3`
  to `~1.1e6` at `kappa=1e4` (`STALLED`, not a genuine convergence -
  see above) and `~2.7e14` by `kappa=1e8` (`FAILED`). scipy's own BFGS
  stays at `cost_final==0.0` from `kappa=1e6` onward across the same
  range.
- `dimensionality` (`n` from `2` to `1000`): `trust-apl BFGS` reaches
  `MAX_ITER` at `n=100` with a worse residual than scipy's own BFGS at
  the same `n` (`1.71` vs `0.07`), and times out entirely at `n=1000`
  (`tests/backends/test_apl_difficulty_families.py::
  test_solve_bfgs_completes_without_a_timeout_at_a_practical_dimension`
  pins the `n=100` case; the harness's own subprocess timeout, not a
  study assertion, is what stops `n=1000` from completing).

`trust-apl trust-exact`, by contrast, tracks scipy's own trust-exact
closely across both sweeps - `dimensionality` converges at `n=1000` on
both backends
(`tests/backends/test_apl_difficulty_families.py::
test_solve_trust_exact_converges_at_n_1000`), and `ill_conditioning`
only shows the same stall pattern at `kappa=1e7`/`1e8`, well past where
BFGS already fails. The gap is BFGS-specific, not a general weakness of
`trust-apl`'s Newton-region engine.

## Trust-exact and Hessian indefiniteness

Unlike `ill_conditioning`'s numerical-precision boundary above, Hessian
indefiniteness is a separate, genuine property of the underlying maths
that `trust-exact`'s Newton-region engine must navigate, confirmed
directly (`tests/problems/test_families.py::
test_dimensionality_hessian_is_indefinite_partway_between_the_start_and_the_optimum`,
`tests/problems/test_typical.py::
test_hessian_indefiniteness_at_the_standard_start_matches_trust_exacts_known_fragility`),
not only inferred from solver behaviour:

- A nonlinear-least-squares problem's true Hessian is `J.T@J` plus a
  residual-weighted correction term that vanishes only where the
  residual itself is zero. Away from a zero-residual point, that
  correction can flip an eigenvalue's sign.
- `dimensionality` (generalised Rosenbrock, zero residual at `x*`):
  indefinite partway between its standard start and the optimum
  (measured, `n=10`: minimum eigenvalue `-194` at the midpoint) - but
  mildly enough that `trust-exact` still converges reliably through
  `n=1000` (see the section above).
- `noisy_expdec`/`gaussian_peak` (the typical study, nonzero residual
  even at `x*` since both fit noisy data): indefinite already at the
  standard start (measured minimum eigenvalue `-82`/`-8`). Before
  vendored `trust` commit `05f9010`, `trust-apl trust-exact` genuinely
  failed here (`MAX_ITER`, `dist_to_opt` in the thousands to billions):
  `Newton.aplo` accepted a step whenever its actual outcome improved
  the cost, even if the quadratic model that produced it had predicted
  a negative error decrement - a sign the damping wasn't yet enough to
  counteract the indefinite Hessian. `05f9010` rejects any such step
  outright, and `trust-apl trust-exact` now converges cleanly on both
  problems (`tests/studies/test_typical_study_apl.py::
  test_trust_apls_trust_exact_converges_despite_an_indefinite_hessian_away_from_the_optimum`).
  `logistic`/`michaelis_menten`, the typical study's other two problems,
  have an always-PSD Hessian (canonical-link-shaped likelihoods) and
  were never affected either way.

`ill_conditioning`'s fragility, despite looking similar in its symptom
(`trust-exact` failing at an extreme parameter value), is a different
mechanism entirely (numerical precision loss, not indefiniteness - see
above) and is not affected by this fix.
