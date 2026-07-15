#!/usr/bin/env bash
set -euo pipefail

input="$1"
output="$2"
apl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_sources.sh
source "$apl_dir/_sources.sh"

script="$(mktemp -t trust-bench-apl-harness.XXXXXX.dyalog)"
trap 'rm -f "$script"' EXIT

{
  _apl_write_sources "$apl_dir"
  cat "$apl_dir/run.dyalog"
} > "$script"

dyalogscript DYALOG_INITSESSION=1 "$script" "$input" "$output"
