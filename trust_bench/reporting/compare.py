import pandas as pd

# Section 6 of docs/plans/trust-bench.md: Tier-1 metrics are algorithm
# properties expected to be stable run to run, so any change flags a
# real behavioural difference. `status` is included alongside the
# numeric fields for the same reason: a run that stops converging is a
# regression even when the numeric fields it did report look close.
TIER1_METRICS = ["dist_to_opt", "cost_gap", "grad_norm_final", "status"]

DEFAULT_KEY_COLUMNS = ["problem_id", "backend", "method", "start"]

REGRESSION = "regression"
DRIFT = "drift"
STABLE = "stable"


def _numeric_changed(baseline, candidate, tol):
    if baseline is None or candidate is None:
        return baseline != candidate
    try:
        return abs(float(baseline) - float(candidate)) > tol
    except (TypeError, ValueError):
        return baseline != candidate


def _dedupe_latest(df, key_columns):
    """Collapses to one row per `key_columns`, keeping the row with the
    latest `timestamp`. `results/*.jsonl` is an append-only log where the
    same key can be recorded more than once - a single report run can
    revisit a key (e.g. a basin-rate sweep re-probing a start already
    covered by the main sweep), and running `report` twice on one commit
    appends again by design - so "this key's result" means its most
    recent recording, not every recording ever made. Matching on a
    non-unique key would otherwise cross-join: two duplicated rows on
    each side of a merge produce four, not two.
    """
    return df.sort_values("timestamp").drop_duplicates(subset=list(key_columns), keep="last")


def compare(baseline, candidate, key_columns=DEFAULT_KEY_COLUMNS, tier1_tol=1e-9):
    """Diff two loaded result DataFrames (`storage.load`'s output),
    matched on `key_columns`. Tier-1 metrics are expected stable (Section
    6): any change beyond `tier1_tol` classifies the row as a
    "regression". Tier-3's `timing.median` is expected to change
    (Section 8): absent a Tier-1 regression, any change classifies the
    row as "drift" and carries the candidate's `machine_fingerprint`/
    `backend_version` so drifted rows can be grouped by environment. A
    Tier-1 regression on a row takes priority over drift on that same
    row: a genuine behavioural change must not be masked by expected
    timing movement. Each side is collapsed to its latest row per key
    before matching (see `_dedupe_latest`).
    """
    baseline = _dedupe_latest(baseline, key_columns)
    candidate = _dedupe_latest(candidate, key_columns)
    merged = baseline.merge(candidate, on=list(key_columns), suffixes=("_baseline", "_candidate"))

    rows = []
    for _, row in merged.iterrows():
        changed_tier1 = [
            metric
            for metric in TIER1_METRICS
            if _numeric_changed(row[f"{metric}_baseline"], row[f"{metric}_candidate"], tier1_tol)
        ]

        timing_baseline = row["timing_baseline"]
        timing_candidate = row["timing_candidate"]
        median_baseline = timing_baseline["median"] if timing_baseline else None
        median_candidate = timing_candidate["median"] if timing_candidate else None
        drifted = _numeric_changed(median_baseline, median_candidate, 0.0)

        if changed_tier1:
            classification = REGRESSION
        elif drifted:
            classification = DRIFT
        else:
            classification = STABLE

        provenance_candidate = row["provenance_candidate"]
        entry = {column: row[column] for column in key_columns}
        entry["classification"] = classification
        entry["changed_tier1_metrics"] = changed_tier1
        entry["timing_median_baseline"] = median_baseline
        entry["timing_median_candidate"] = median_candidate
        entry["machine_fingerprint"] = provenance_candidate["machine_fingerprint"]
        entry["backend_version"] = provenance_candidate["backend_version"]
        rows.append(entry)

    return pd.DataFrame(rows)


def drift_summary(compared):
    """Rows classified `drift`, grouped by `machine_fingerprint` and
    `backend_version` (Section 8: drift is reported as a trend per
    environment, not as a flat count).
    """
    drift = compared[compared["classification"] == DRIFT]
    return drift.groupby(["machine_fingerprint", "backend_version"]).size().reset_index(name="n_drifted")


def compare_with_provenance(baseline, candidate, key_columns=DEFAULT_KEY_COLUMNS, tier1_tol=1e-9):
    """`compare()`'s classification, plus every `EnvProvenance` field for
    both sides (prefixed `baseline_`/`candidate_`): a regression or
    drift needs its full environment to be attributable, not only the
    machine_fingerprint/backend_version `compare()` itself needs for
    `drift_summary`'s grouping. Provenance is joined back onto the
    classification explicitly by `key_columns`, not by row position, so
    this stays correct regardless of how `compare()`'s own internal
    merge orders its output. Both sides are deduplicated to their latest
    row per key up front (see `_dedupe_latest`), so this second, separate
    merge stays one-to-one with `compare()`'s own instead of compounding
    any duplication into a cross-join.
    """
    baseline = _dedupe_latest(baseline, key_columns)
    candidate = _dedupe_latest(candidate, key_columns)
    classified = compare(baseline, candidate, key_columns=key_columns, tier1_tol=tier1_tol)
    merged = baseline.merge(candidate, on=list(key_columns), suffixes=("_baseline", "_candidate"))
    provenance_rows = []
    for _, row in merged.iterrows():
        entry = {column: row[column] for column in key_columns}
        entry.update({f"baseline_{field}": value for field, value in row["provenance_baseline"].items()})
        entry.update({f"candidate_{field}": value for field, value in row["provenance_candidate"].items()})
        provenance_rows.append(entry)
    provenance = pd.DataFrame(provenance_rows)
    return classified.merge(provenance, on=list(key_columns))


def classification_counts(compared, group_column="backend"):
    """Row counts per (`group_column`, `classification`) - the graph
    trust-bench compare's report shows alongside the full-provenance
    table.
    """
    return compared.groupby([group_column, "classification"]).size().reset_index(name="count")
