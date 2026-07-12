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
ever fires, and does so at genuine convergence. On a problem where
`trust-exact`'s true Hessian is indefinite far from the optimum (high
`kappa` in `ill_conditioning`, large `n` in `dimensionality`), the same
criterion can fire from a point nowhere near the optimum, once damping
has grown large without quite crossing the `FAILED` threshold.

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
both:

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

Out of scope here: whether `trust-exact`'s own indefinite-Hessian
fragility, independently confirmed on two non-adversarial curve-fitting
problems in `tests/studies/test_typical_study_apl.py`, warrants a wider
capability note of its own. Left for separate consideration rather than
folded into this one.
