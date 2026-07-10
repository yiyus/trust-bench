import pandas as pd

_METRIC_FIELDS = [
    "status",
    "cost_final",
    "dist_to_opt",
    "cost_gap",
    "grad_norm_final",
    "n_iter",
    "n_feval",
    "n_jeval",
    "n_heval",
]


def results_to_dataframe(results, key_names):
    """Flattens a study's sweep() dict (keyed by a tuple matching
    key_names) into a comparison table: one row per RunResult, one
    column per sweep key and per Tier-1/Tier-2 RunResult field.
    """
    rows = []
    for key, result in results.items():
        row = dict(zip(key_names, key))
        row["status"] = result.status.value
        for field in _METRIC_FIELDS[1:]:
            row[field] = getattr(result, field)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*key_names, *_METRIC_FIELDS])


def save_table(df, path):
    df.to_csv(path, index=False)
