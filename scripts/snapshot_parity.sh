#!/usr/bin/env bash
# Snapshot golden-reference workflow: render a fixed scenario set offscreen and
# prove rendering parity against captured baselines.
#
#   scripts/snapshot_parity.sh capture   -> render golden refs to snapshots/baseline/
#   scripts/snapshot_parity.sh check     -> render to snapshots/current/ and diff vs baseline
#
# Each scenario is one deterministic frame (fixed --seed and --size). The probe
# JSON is compared numerically and the PNG pixel-by-pixel, so a refactor that
# changes nothing about the scene must reproduce the baseline exactly.
#
# Requires the binary to be built already: `cargo build` first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="target/debug/lane_lunacy"
SIZE="1280x720"
SEED="42"

if [ ! -x "$BIN" ]; then
    echo "binary not found: $BIN (run 'cargo build' first)" >&2
    exit 1
fi

render_scenarios() {
    local out_dir="$1"
    local name
    for name in noon_clear midnight_rain dusk; do
        case "$name" in
            noon_clear)     args=(--time 12 --weather clear) ;;
            midnight_rain)  args=(--time 0 --weather rain) ;;
            dusk)           args=(--time 18 --weather clear) ;;
        esac
        echo ">> render $name"
        "$BIN" --snapshot "$out_dir/$name.png" --size "$SIZE" --seed "$SEED" --gpu 0 \
            --terrain-detail med "${args[@]}" >"$out_dir/$name.log"
    done
}

case "${1:-}" in
    capture)
        mkdir -p snapshots/baseline
        render_scenarios snapshots/baseline
        echo "baselines captured under snapshots/baseline/"
        ;;
    check)
        mkdir -p snapshots/current
        render_scenarios snapshots/current
        local_fail=0
        for name in noon_clear midnight_rain dusk; do
            if ! python3 scripts/parity_diff.py \
                "snapshots/baseline/$name" "snapshots/current/$name"; then
                local_fail=1
            fi
        done
        if [ "$local_fail" -eq 0 ]; then
            echo "PARITY OK: all snapshots match their baselines"
        else
            echo "PARITY FAILED: see diffs above" >&2
        fi
        exit "$local_fail"
        ;;
    *)
        echo "usage: $0 capture|check" >&2
        exit 1
        ;;
esac
