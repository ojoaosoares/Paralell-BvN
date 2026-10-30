#!/usr/bin/env bash
# ==============================================================================
# CloudLab Bare-Metal Architectural Benchmark Suite
# Parallel Birkhoff-von Neumann (Eulerian Divide-and-Conquer)
#
# Tunables (environment variables):
#   REPS      repetitions per configuration (default: 10)
#   N_LIST    switch sizes, D = N            (default: "256 512 1024 2048")
#   PATTERNS  traffic patterns               (default: "uniform skewed bursty")
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results_$(date +%Y%m%d_%H%M%S)"

REPS="${REPS:-10}"
read -r -a N_LIST <<< "${N_LIST:-256 512 1024 2048}"
read -r -a PATTERNS <<< "${PATTERNS:-uniform skewed bursty}"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

echo "[1/6] Collecting Hardware & NUMA Topology..."
lscpu > "${RESULTS_DIR}/lscpu.txt" || true
if command -v numactl &> /dev/null; then
    numactl --hardware > "${RESULTS_DIR}/numactl_hardware.txt" || true
fi
if command -v lstopo-no-graphics &> /dev/null; then
    lstopo-no-graphics > "${RESULTS_DIR}/hwloc_topology.txt" || true
fi

echo "[2/6] Pinning CPU frequency governor to 'performance' (best effort)..."
if command -v cpupower &> /dev/null; then
    sudo -n cpupower frequency-set -g performance > /dev/null 2>&1 \
        || echo "  cpupower failed (no sudo?); timings may include DVFS noise."
else
    echo "  cpupower not found; timings may include DVFS noise."
fi
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "${RESULTS_DIR}/governor.txt" 2>/dev/null || true

echo "[3/6] Building Release Binary (-O3 -march=native -fopenmp)..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

BIN="${BUILD_DIR}/bvn_bench"

# nproc counts SMT siblings; OMP_PLACES=cores binds one thread per physical core,
# so thread counts above the physical core count would oversubscribe places.
PHYS_CORES="$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)"
echo "  Physical cores: ${PHYS_CORES} (logical: $(nproc))"

THREAD_LIST=(1 2 4)
for t in 8 12 16 24 32 48 64 96 128; do
    if [ "$t" -le "$PHYS_CORES" ]; then
        THREAD_LIST+=("$t")
    fi
done
if [ "${THREAD_LIST[-1]}" -ne "$PHYS_CORES" ]; then
    THREAD_LIST+=("$PHYS_CORES")
fi

CSV_HEADER="rep,binding,mode,pattern,N,D,threads,cutoff,phase1_ms,phase2_3_ms,phase4_ms,total_ms,schedules,frame_eff,verified"

# run_case <csv> <rep> <binding> <bvn_bench args...>
# binding: none | spread | close  (sets OMP_PLACES/OMP_PROC_BIND accordingly)
run_case() {
    local csv="$1" rep="$2" binding="$3"
    shift 3
    local line
    if [ "${binding}" = "none" ]; then
        line="$(OMP_PROC_BIND=false "${BIN}" "$@" --csv)" || true
    else
        line="$(OMP_PLACES=cores OMP_PROC_BIND="${binding}" "${BIN}" "$@" --csv)" || true
    fi
    if [ -z "${line}" ]; then
        line="$*,TIMEOUT_OR_CRASH"
    fi
    echo "${rep},${binding},${line}" >> "${csv}"
}

SCALING_CSV="${RESULTS_DIR}/scaling_results.csv"
echo "${CSV_HEADER}" > "${SCALING_CSV}"

echo "[4/6] Strong Scaling & Affinity Experiments (${REPS} reps, N in ${N_LIST[*]})..."
for N in "${N_LIST[@]}"; do
    D="${N}"
    for P in "${PATTERNS[@]}"; do
        echo "  N=${N} pattern=${P}"
        for ((r = 0; r < REPS; ++r)); do
            run_case "${SCALING_CSV}" "$r" none --mode seq --n "${N}" --d "${D}" --pattern "${P}"
            for T in "${THREAD_LIST[@]}"; do
                # Naive OpenMP (no affinity, unaligned cursors, no cutoff)
                run_case "${SCALING_CSV}" "$r" none \
                    --mode omp_naive --n "${N}" --d "${D}" --pattern "${P}" --threads "${T}"
                for BIND in spread close; do
                    run_case "${SCALING_CSV}" "$r" "${BIND}" \
                        --mode omp_arch --n "${N}" --d "${D}" --pattern "${P}" --threads "${T}" --cutoff 8
                    run_case "${SCALING_CSV}" "$r" "${BIND}" \
                        --mode omp_block --n "${N}" --d "${D}" --pattern "${P}" --threads "${T}" --cutoff 8
                done
            done
        done
    done
done

CUTOFF_CSV="${RESULTS_DIR}/cutoff_sensitivity.csv"
echo "${CSV_HEADER}" > "${CUTOFF_CSV}"

echo "[5/6] Task Cut-off Granularity Sweep..."
for N in 512 2048; do
    for CUTOFF in 2 4 8 16 32 64 128; do
        for ((r = 0; r < REPS; ++r)); do
            run_case "${CUTOFF_CSV}" "$r" spread \
                --mode omp_arch --n "${N}" --d "${N}" --threads "${PHYS_CORES}" --cutoff "${CUTOFF}"
        done
    done
done

echo "[6/6] Collecting Hardware Performance Counters (perf stat)..."
if command -v perf &> /dev/null; then
    # Generic LLC-* events are often unsupported on AMD; keep only events this CPU exposes.
    CANDIDATE_EVENTS=(cycles instructions cache-references cache-misses
                      L1-dcache-loads L1-dcache-load-misses LLC-loads LLC-load-misses
                      dTLB-loads dTLB-load-misses stalled-cycles-backend)
    PERF_EVENTS=""
    for ev in "${CANDIDATE_EVENTS[@]}"; do
        out="$(perf stat -x, -e "${ev}" true 2>&1 || true)"
        if [[ "${out}" == *",${ev}"* && "${out}" != *"not supported"* && "${out}" != *"not counted"* ]]; then
            PERF_EVENTS="${PERF_EVENTS:+${PERF_EVENTS},}${ev}"
        fi
    done
    echo "  perf events: ${PERF_EVENTS:-<none>}"
fi
if [ -z "${PERF_EVENTS:-}" ]; then
    echo "  No usable perf events (perf missing or perf_event_paranoid too strict); skipping."
    echo "  (on CloudLab: sudo sysctl kernel.perf_event_paranoid=-1)"
else
    for N in 512 2048; do
        for MODE in seq omp_naive omp_arch omp_block; do
            OMP_PLACES=cores OMP_PROC_BIND=spread \
                perf stat -r "${REPS}" -e "${PERF_EVENTS}" -o "${RESULTS_DIR}/perf_${MODE}_n${N}.txt" \
                "${BIN}" --mode "${MODE}" --n "${N}" --d "${N}" --threads "${PHYS_CORES}" --cutoff 8 --no-verify \
                > /dev/null || true
        done
    done
fi

echo "Done! All results saved in: ${RESULTS_DIR}"
