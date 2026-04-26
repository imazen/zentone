#!/usr/bin/env bash
# Rebuild the C++ reference binaries and regenerate the golden CSV files.
#
# Run this when:
# - You update one of the extraction .cpp files
# - You want to refresh against a new upstream commit
# - You suspect a compiler difference is masking a regression

set -euo pipefail

cd "$(dirname "$0")"

mkdir -p golden

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra -Wpedantic}

build_and_run() {
    local name=$1
    echo "==> $name"
    $CXX $CXXFLAGS "${name}.cpp" -o "${name}"
    ./"${name}" > "golden/${name}.csv"
    rm -f "${name}"
    echo "    wrote golden/${name}.csv ($(wc -l < "golden/${name}.csv") lines)"
}

build_and_run libultrahdr_reinhard
build_and_run libultrahdr_apply_gain
build_and_run libultrahdr_compute_gain
build_and_run libultrahdr_luminance
build_and_run libultrahdr_gamut
build_and_run libultrahdr_hlg_ootf
build_and_run libavif_apply_gain
build_and_run libplacebo_bt2390
build_and_run darktable_filmic

echo ""
echo "Golden files updated. Review the diff and commit if intentional:"
echo "  git -C .. diff reference-checks/golden/"
