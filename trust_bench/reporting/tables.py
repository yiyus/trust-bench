import pandas as pd

# Neither status carries a genuine dist_to_opt/coverage-worthy result:
# "ERROR" is a harness-side crash or timeout, "UNSUPPORTED" is a
# declared-unsupported rejection recorded here instead of raising (see
# results_to_dataframe below). Shared so every consumer that filters a
# results_to_dataframe table for "did this backend actually produce
# something" excludes both, not just the historical "ERROR" alone.
NON_RESULT_STATUSES = frozenset({"ERROR", "UNSUPPORTED"})

_METRIC_FIELDS = [
    "status",
    "message",
    "cost_final",
    "dist_to_opt",
    "cost_gap",
    "grad_norm_final",
    "n_iter",
    "n_feval",
    "n_jeval",
    "n_heval",
]

# message is populated (str(exception)) rather than blanked below: it's
# the only thing that distinguishes one declared-unsupported rejection
# from another in a table full of otherwise-identical UNSUPPORTED rows.
_BLANKED_ON_EXCEPTION = _METRIC_FIELDS[2:]


def results_to_dataframe(results, key_names):
    """Flattens a study's sweep() dict (keyed by a tuple matching
    key_names) into a comparison table: one row per result, one column
    per sweep key and per Tier-1/Tier-2 RunResult field.

    A value may be a raised exception instead of a RunResult (e.g.
    bounded.py's infeasible-start scenario, or scaling.py's own
    x_scale="jac" probe against a backend with no adaptive equivalent):
    every except-ValueError block in this project's studies catches
    exactly a backend's own declared-unsupported or rejected-input
    rejection - the expected, passing outcome of the sweep's own probe,
    not a genuine crash (which would propagate rather than land here).
    That row gets status "UNSUPPORTED", its exception message kept
    (rather than discarded), and every other metric field left blank.
    """
    rows = []
    for key, result in results.items():
        row = dict(zip(key_names, key))
        if isinstance(result, BaseException):
            row["status"] = "UNSUPPORTED"
            row["message"] = str(result)
            for field in _BLANKED_ON_EXCEPTION:
                row[field] = None
        else:
            row["status"] = result.status.value
            for field in _METRIC_FIELDS[1:]:
                row[field] = getattr(result, field)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*key_names, *_METRIC_FIELDS])


def save_table(df, path):
    df.to_csv(path, index=False)
