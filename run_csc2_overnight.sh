#!/usr/bin/env bash
# CSC-2 overnight run. Ordered by the dependency structure, not by cost:
#
#   E0  validate readouts FIRST (rule R5 — CSC calibrated through a readout it
#       later disqualified, and two headline numbers reversed)
#   E4  external positive control; gates E3. If our geometry cannot reproduce a
#       settled literature result, no CSC-2 number means anything.
#   E1  is CSC's null readout-specific? Determines how CSC's result is stated.
#   E3  the main event: does the advantage grow with hierarchy depth?
#
# Each stage writes its own artifact, so a failure part-way leaves everything
# before it intact and inspectable.
set -u
W=${WORKERS:-20}
LOG=${LOG_DIR:-/tmp/csc2}
mkdir -p "$LOG"

run () {
  local name=$1; shift
  echo "=== $(date -Is)  START $name ==="
  if uv run python -m "$@" --workers "$W" > "$LOG/$name.log" 2>&1; then
    echo "=== $(date -Is)  DONE  $name ==="
    tail -12 "$LOG/$name.log"
  else
    echo "=== $(date -Is)  FAILED $name (continuing; later stages may be affected) ==="
    tail -25 "$LOG/$name.log"
  fi
}

run e0 experiments.csc2.e0_head_validation.run_e0
run e4 experiments.csc2.e4_positive_control.run_e4
run e1 experiments.csc2.e1_readout_scale.run_e1
run e3 experiments.csc2.e3_hierarchy.run_e3
echo "=== $(date -Is)  ALL STAGES COMPLETE ==="
