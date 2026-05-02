#!/usr/bin/env bash
# Rebuild all hand-rolled-pipeline kernels. Skips legacy IRON-Python
# kernels (those still ship as committed binaries from the prior path).
#
# Run from anywhere; this script resolves paths from its own location.
# Usage:
#   bash kernels/aie2p/build_all.sh         # rebuild everything
#   bash kernels/aie2p/build_all.sh asym3   # filter by name substring

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER="${1:-}"

failed=()
built=()
skipped=()

for kdir in "$DIR"/*/; do
    name="$(basename "$kdir")"
    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
        continue
    fi
    if [ ! -x "$kdir/build.sh" ]; then
        skipped+=("$name (no build.sh)")
        continue
    fi
    echo "=== Building $name ==="
    if (cd "$kdir" && bash build.sh) ; then
        built+=("$name")
    else
        failed+=("$name")
    fi
done

echo
echo "=== Summary ==="
echo "Built (${#built[@]}): ${built[*]:-none}"
echo "Skipped (${#skipped[@]}): ${skipped[*]:-none}"
echo "Failed (${#failed[@]}): ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
