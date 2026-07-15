# Shared combined-script assembly for both run_harness.sh (one-shot,
# file-based) and run_session.sh (persistent, stdin/stdout-based) -
# everything except each one's own driver file (run.dyalog /
# session.dyalog respectively), so a new problem family only needs
# registering in this one list, not copied by hand into two scripts.
_apl_write_sources() {
  local apl_dir="$1"
  local trust_source="$apl_dir/trust/APLSource"
  echo "⎕SE.Link.Import '#' '$trust_source'"
  cat "$apl_dir/problems/rosenbrock.dyalog"
  cat "$apl_dir/problems/beale.dyalog"
  cat "$apl_dir/problems/powell.dyalog"
  cat "$apl_dir/problems/helical.dyalog"
  cat "$apl_dir/problems/expdec.dyalog"
  cat "$apl_dir/problems/quadratic.dyalog"
  cat "$apl_dir/problems/linear.dyalog"
  cat "$apl_dir/problems/noisy_expdec.dyalog"
  cat "$apl_dir/problems/logistic.dyalog"
  cat "$apl_dir/problems/michaelis_menten.dyalog"
  cat "$apl_dir/problems/gaussian_peak.dyalog"
  cat "$apl_dir/problems/scaling.dyalog"
  cat "$apl_dir/problems/ill_conditioned.dyalog"
  cat "$apl_dir/problems/large_residual.dyalog"
  cat "$apl_dir/problems/outliers.dyalog"
  cat "$apl_dir/problems/dimensionality.dyalog"
  cat "$apl_dir/dispatch.dyalog"
  cat "$apl_dir/null.dyalog"
  cat "$apl_dir/error_result.dyalog"
  cat "$apl_dir/solve.dyalog"
  cat "$apl_dir/evaluate.dyalog"
}
