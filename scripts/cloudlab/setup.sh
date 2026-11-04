#!/usr/bin/env bash
# ==============================================================================
# CloudLab node setup for Parallel-BvN
# Invoked by profile.py at every boot (as root). Safe to re-run.
#   --run-bench      run scripts/run_cloudlab_bench.sh after building
#   --perf-governor  pin the CPU frequency governor to "performance"
# ==============================================================================
set -euo pipefail

REPO_DIR="/local/repository"
LOG_DIR="/local/logs"
SETUP_LOG="${LOG_DIR}/bvn-setup.log"
BENCH_LOG="${LOG_DIR}/bvn-bench.log"
MARKER="/local/.bvn-deps-installed"

RUN_BENCH=0
PERF_GOVERNOR=0
for arg in "$@"; do
    case "${arg}" in
        --run-bench) RUN_BENCH=1 ;;
        --perf-governor) PERF_GOVERNOR=1 ;;
        "") ;;
        *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${SETUP_LOG}") 2>&1
echo "=== Parallel-BvN setup started at $(date -Is) on $(hostname) ==="

# ------------------------------------------------------------------------------
# 1. Dependencies (only on first boot)
# ------------------------------------------------------------------------------
if [[ ! -f "${MARKER}" ]]; then
    echo "[1/4] Installing dependencies..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y -q \
        build-essential cmake git python3 \
        numactl hwloc linux-tools-common "linux-tools-$(uname -r)" || \
    apt-get install -y -q build-essential cmake git python3 numactl hwloc linux-tools-common
    touch "${MARKER}"
else
    echo "[1/4] Dependencies already installed, skipping."
fi

# ------------------------------------------------------------------------------
# 2. Kernel / CPU settings for reproducible measurements (reset on each boot)
# ------------------------------------------------------------------------------
echo "[2/4] Configuring perf access and CPU governor..."
sysctl -w kernel.perf_event_paranoid=-1 || true
sysctl -w kernel.kptr_restrict=0 || true
if [[ "${PERF_GOVERNOR}" -eq 1 ]]; then
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "${gov}" ]] && echo performance > "${gov}" || true
    done
fi

# ------------------------------------------------------------------------------
# 3. Build (Release)
# ------------------------------------------------------------------------------
echo "[3/4] Building bvn_bench..."
cmake -S "${REPO_DIR}" -B "${REPO_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${REPO_DIR}/build" -j"$(nproc)"

# Every experiment member must be able to build, run and write results here.
chmod -R a+rwX "${REPO_DIR}"

# ------------------------------------------------------------------------------
# 4. Record the topology next to the logs (NUMA layout differs between nodes)
# ------------------------------------------------------------------------------
echo "[4/4] Recording hardware topology..."
lscpu > "${LOG_DIR}/lscpu.txt" || true
numactl --hardware > "${LOG_DIR}/numactl_hardware.txt" || true
grep -E "Model name|Socket|NUMA node\(s\)|^CPU\(s\)" "${LOG_DIR}/lscpu.txt" || true

echo "=== Setup finished at $(date -Is). Binary: ${REPO_DIR}/build/bvn_bench ==="

if [[ "${RUN_BENCH}" -eq 1 ]]; then
    echo "Starting benchmark suite in background, log: ${BENCH_LOG}"
    nohup bash -c "cd '${REPO_DIR}' && ./scripts/run_cloudlab_bench.sh" > "${BENCH_LOG}" 2>&1 &
fi
