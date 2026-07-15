#!/usr/bin/env bash
set -euo pipefail

apl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_sources.sh
source "$apl_dir/_sources.sh"

script="$(mktemp -t trust-bench-apl-session.XXXXXX.dyalog)"
trap 'rm -f "$script"' EXIT

{
  _apl_write_sources "$apl_dir"
  cat "$apl_dir/session.dyalog"
} > "$script"

# dyalogscript's stdout is fully buffered, not line-buffered, when
# connected to a pipe (confirmed directly: without this, a response sits
# in the interpreter's own buffer and is never actually written until it
# fills or the process exits, which a synchronous one-request-at-a-time
# protocol can't tolerate). stdbuf -oL forces line buffering via
# LD_PRELOAD, inherited through the dyalogscript/dyalog exec chain.
stdbuf -oL dyalogscript DYALOG_INITSESSION=1 "$script"
