#!/usr/bin/env bash
set -euo pipefail

input="$1"
output="$2"
apl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trust_source="$apl_dir/trust/APLSource"

script="$(mktemp -t trust-bench-apl-harness.XXXXXX.dyalog)"
trap 'rm -f "$script"' EXIT

{
  echo "⎕SE.Link.Import '#' '$trust_source'"
  cat "$apl_dir/problems/rosenbrock.dyalog"
  cat "$apl_dir/null.dyalog"
  cat "$apl_dir/error_result.dyalog"
  cat "$apl_dir/solve.dyalog"
  cat "$apl_dir/run.dyalog"
} > "$script"

dyalogscript DYALOG_INITSESSION=1 "$script" "$input" "$output"
