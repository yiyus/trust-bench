# trust-bench

Optimisation-solver comparison harness. Benchmarks the Dyalog APL library
[`trust`](https://github.com/yiyus/trust) (trust-region Newton, BFGS and
Levenberg-Marquardt, robust losses, box constraints) against SciPy across a
set of difficulty studies (large residuals, ill-conditioning, robust losses,
bounds, scaling, dimensionality, derivative source), producing comparison
tables, plots, and an HTML report.

See `docs/plans/trust-bench.md` for the full design and `docs/methodology.md`
for the comparability rules behind the metrics.

## Setup

```
git clone --recurse-submodules <repo-url>
cd trust-bench
pip install -e .[dev]
```

The APL backend additionally requires `dyalogscript` (Dyalog APL) on `PATH`.
Without it, APL-backed studies and tests are skipped automatically; the SciPy
backend has no extra dependencies.

## Usage

Run a report (trust-apl only, skipping the slow studies, writing
`report.html` and appending to `results/*.jsonl` - all on by default):

```
trust-bench report --output-dir reports
```

Also run SciPy, for a side-by-side comparison:

```
trust-bench report --output-dir reports --scipy
```

Run the complete report, including the slow studies:

```
trust-bench report --output-dir reports --full
```

Run a subset of studies:

```
trust-bench report --only baseline scaling
```

Check for regressions or drift against a prior report's own output
directory (folds a "Longitudinal comparison" section into `report.html`):

```
trust-bench report --output-dir reports-new reports-old
```

Diff two existing report directories directly, without rerunning studies:

```
trust-bench compare reports-old reports-new --html
```

## Tests

```
make test       # fast subset
make coverage   # full suite with coverage
make lint
```
