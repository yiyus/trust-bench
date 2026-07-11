# Comparability methodology

Decisions that keep a cross-backend comparison meaningful, enforced by
the runner/config layer and tested per backend. See Section 7 of
`docs/plans/trust-bench.md` for the full list this document expands;
this file currently covers tolerance mapping only.

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
