import json

import pandas as pd


def append(result, path):
    """Append a RunResult to a JSONL file, one record per line. Never
    truncates or overwrites an existing file; creates it if absent.
    """
    with open(path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def load(path):
    """Load a JSONL result file into a DataFrame, one row per record.
    Nested fields (provenance, timing, config) stay as dicts, one
    object-dtype column each, not flattened into dotted columns.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
