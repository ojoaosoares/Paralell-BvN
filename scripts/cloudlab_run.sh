#!/usr/bin/env bash
# ==============================================================================
# One-shot CloudLab driver: prepares a fresh bare-metal node, checks correctness,
# runs the benchmark suite and packs the results into a single tarball.
#
# Usage (on the CloudLab node, inside tmux/screen so an SSH drop does not kill it):
#   ./scripts/cloudlab_run.sh quick   # ~15-30 min: N in {1024, 2048}, uniform + bursty, 3 reps
#   ./scripts/cloudlab_run.sh full    # ~1-3 h:     N in {256..2048}, all patterns, 10 reps
#
# Environment overrides (REPS, N_LIST, PATTERNS) are passed to run_cloudlab_bench.sh.
# ==============================================================================
set -euo pipefail

MODE="${1:-quick}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

case "${MODE}" in
    quick)
        export REPS="${REPS:-3}"
        export N_LIST="${N_LIST:-1024 2048}"
        export PATTERNS="${PATTERNS:-uniform bursty}"
        ;;
    full)
        export REPS="${REPS:-10}"
        export N_LIST="${N_LIST:-256 512 1024 2048}"
        export PATTERNS="${PATTERNS:-uniform skewed bursty}"
        ;;
    *)
        echo "Usage: $0 [quick|full]" >&2
        exit 1
        ;;
esac

echo "=== [A] Installing build and profiling tools (apt) ==="
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        build-essential cmake numactl hwloc linux-tools-common > /dev/null
    # perf/cpupower must match the running kernel; not every image has the package.
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "linux-tools-$(uname -r)" > /dev/null 2>&1 \
        || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq linux-tools-generic > /dev/null 2>&1 \
        || echo "  linux-tools for $(uname -r) unavailable; perf/cpupower steps will be skipped."
else
    echo "  apt-get not found; install cmake, g++ (>= 9), numactl, hwloc and perf manually."
fi

echo "=== [B] Unlocking hardware counters ==="
sudo sysctl -q kernel.perf_event_paranoid=-1 || echo "  could not set perf_event_paranoid"
sudo sysctl -q kernel.kptr_restrict=0 || true

echo "=== [C] Node summary ==="
lscpu | grep -E '^(Model name|Socket|Core|Thread|NUMA node\(s\)|L1d|L2|L3)' || true
numactl --hardware 2> /dev/null | head -3 || true
g++ --version | head -1

echo "=== [D] Building (Release, -march=native) ==="
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "${BUILD_DIR}" -j"$(nproc)" > /dev/null
BIN="${BUILD_DIR}/bvn_bench"
PHYS_CORES="$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)"

echo "=== [E] Correctness smoke test (${PHYS_CORES} threads, exact verification) ==="
fail=0
for N in 512 2048; do
    for P in uniform skewed bursty; do
        for M in seq omp_naive omp_arch omp_block; do
            [ "${M}" = omp_naive ] && [ "${N}" -gt 1024 ] && continue
            out="$(OMP_PLACES=cores OMP_PROC_BIND=spread timeout 120 "${BIN}" --mode "${M}" --n "${N}" --d "${N}" \
                --pattern "${P}" --threads "${PHYS_CORES}" --csv 2>&1)" || true
            case "${out}" in
                *PASS) ;;
                *) echo "  FAIL: ${M} ${P} N=${N}: ${out}"; fail=$((fail + 1)) ;;
            esac
        done
        out="$(OMP_PLACES=cores OMP_PROC_BIND=spread timeout 120 "${BIN}" --mode omp_arch --n "${N}" --d "${N}" \
            --pattern "${P}" --threads "${PHYS_CORES}" --par-walk-min 1 --domains 2 --csv 2>&1)" || true
        case "${out}" in
            *PASS) ;;
            *) echo "  FAIL: omp_arch+par-walk+domains ${P} N=${N}: ${out}"; fail=$((fail + 1)) ;;
        esac
    done
done
if [ "${fail}" -ne 0 ]; then
    echo "Smoke test failed (${fail} cases); not running the benchmark." >&2
    exit 2
fi
echo "  all smoke cases PASS"

echo "=== [F] Benchmark suite (${MODE}: REPS=${REPS}, N_LIST=${N_LIST}, PATTERNS=${PATTERNS}) ==="
start="$(date +%s)"
"${ROOT_DIR}/scripts/run_cloudlab_bench.sh"
echo "  suite took $(( ($(date +%s) - start) / 60 )) min"

echo "=== [G] Packing results ==="
RESULTS_DIR="$(ls -d "${ROOT_DIR}"/results_* | sort | tail -1)"
HOST_TAG="$(hostname -s)"
TARBALL="${ROOT_DIR}/bvn_${HOST_TAG}_${MODE}_$(date +%Y%m%d_%H%M%S).tar.gz"
git -C "${ROOT_DIR}" rev-parse HEAD > "${RESULTS_DIR}/git_commit.txt" 2> /dev/null || true
tar -czf "${TARBALL}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")"
echo "Results: ${TARBALL}"
echo "Copy to your machine with:"
echo "  scp $(whoami)@$(hostname -f):${TARBALL} ."
