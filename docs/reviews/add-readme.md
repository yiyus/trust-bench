# Review: branch `add-readme` - Add project README

## Summary

Single-commit, docs-only change (`a782a71`, `README.md`, 52 lines). No
issue number and no tests apply, per the user's own note that this
follows a manual process rather than the red/green TDD workflow. Every
factual claim in the README was checked directly against the current
repository state; no issues found.

## Findings

No issues found.

### Note: every claim verified directly against the repository

- Backend names in the CLI examples (`scipy`, `trust-apl`) match
  `SciPyBackend.name`/`APLBackend.name` (`scipy_backend.py:96`,
  `apl_backend.py:268`).
- `Makefile` has exactly the three targets referenced (`test`, `lint`,
  `coverage`); `make test` runs `pytest -m "not slow"` (the "fast
  subset" the README calls it) and `make coverage` runs the unmarked,
  full suite (the "full suite with coverage" the README calls it),
  consistent with `pyproject.toml`'s own `slow` marker description.
- `git clone --recurse-submodules` matches `.gitmodules`'s one
  submodule (`backends_ext/apl/trust`); `pip install -e .[dev]` matches
  the `dev` extra in `pyproject.toml`.
- CLI flags (`--output-dir`, `--html`, `--backends`, `--only`,
  `--skip-slow`) all exist in `cli.py`'s `build_parser()` with matching
  semantics; a plain `trust-bench report` genuinely defaults to
  SciPy-only, matching the README's claim.
- `docs/plans/trust-bench.md` and `docs/methodology.md`, both
  referenced, exist.
- Writing style: UK spelling, no em/en-dashes, no bold, no emoji -
  consistent with CLAUDE.md's conventions.

### Note: study list is representative, not exhaustive

The seven difficulty studies named (large residuals, ill-conditioning,
robust losses, bounds, scaling, dimensionality, derivative source) are a
subset of `trust_bench/studies/`'s actual modules (also includes
`baseline`, `scalar_cost`, `tolerance_comparability`, `typical`). Read
as a one-line project summary rather than a full study catalogue, this
is accurate as written; not a correctness issue.

## Verification

- Tests: not applicable (docs-only change, no test surface).
- Every command and path referenced in the README was checked directly
  against the repository (Makefile, `pyproject.toml`, `cli.py`,
  `.gitmodules`, backend `name` attributes, referenced doc files) as
  detailed above.

## Recommendation

Approve
