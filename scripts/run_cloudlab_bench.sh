#!/usr/bin/env bash
# ==============================================================================
# CloudLab Bare-Metal Architectural Benchmark Suite
# Parallel Birkhoff-von Neumann (Eulerian Divide-and-Conquer)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

echo "[1/5] Collecting Hardware & NUMA Topology..."
lscpu > "${RESULTS_DIR}/lscpu.txt" || true
if command -v numactl &> /dev/null; then
    numactl --hardware > "${RESULTS_DIR}/numactl_hardware.txt" || true
fi
if command -v lstopo-no-graphics &> /dev/null; then
    lstopo-no-graphics > "${RESULTS_DIR}/hwloc_topology.txt" || true
fi

echo "[2/5] Building Release Binary (-O3 -march=native -fopenmp)..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

BIN="${BUILD_DIR}/bvn_bench"
MAX_CORES="$(nproc)"

# Build thread list dynamically up to MAX_CORES
THREAD_LIST=(1 2 4)
for t in 8 12 16 24 32 48 64 128; do
    if [ "$t" -le "$MAX_CORES" ]; then
        THREAD_LIST+=("$t")
    fi
done

SCALING_CSV="${RESULTS_DIR}/scaling_results.csv"
echo "mode,pattern,N,D,threads,cutoff,binding,phase1_ms,phase2_3_ms,phase4_ms,total_ms,verified" > "${SCALING_CSV}"

echo "[3/5] Running Strong Scaling & NUMA Affinity Experiments..."
for N in 256 512; do
    for D in 256 512; do
        # Baseline Sequential
        LINE="$("${BIN}" --mode seq --n "${N}" --d "${D}" --csv)"
        echo "${LINE%,*},none,${LINE##*,}" >> "${SCALING_CSV}"

        for T in "${THREAD_LIST[@]}"; do
            # 1. Naive OpenMP (no affinity, unaligned cursors, no cutoff)
            OMP_PROC_BIND=false "${BIN}" --mode omp_naive --n "${N}" --d "${D}" --threads "${T}" --csv \
                | sed "s/,PASS$/,none,PASS/;s/,FAIL$/,none,FAIL/" >> "${SCALING_CSV}"

            # 2. Architecture-Aware OpenMP with spread binding (maps subtrees across sockets/CCDs)
            OMP_PLACES=cores OMP_PROC_BIND=spread "${BIN}" --mode omp_arch --n "${N}" --d "${D}" --threads "${T}" --cutoff 8 --csv \
                | sed "s/,PASS$/,spread,PASS/;s/,FAIL$/,spread,FAIL/" >> "${SCALING_CSV}"

            # 3. Architecture-Aware OpenMP with close binding (packs threads into same CCX/socket first)
            OMP_PLACES=cores OMP_PROC_BIND=close "${BIN}" --mode omp_arch --n "${N}" --d "${D}" --threads "${T}" --cutoff 8 --csv \
                | sed "s/,PASS$/,close,PASS/;s/,FAIL$/,close,FAIL/" >> "${SCALING_CSV}"
        done
    done
done

CUTOFF_CSV="${RESULTS_DIR}/cutoff_sensitivity.csv"
echo "mode,pattern,N,D,threads,cutoff,phase1_ms,phase2_3_ms,phase4_ms,total_ms,verified" > "${CUTOFF_CSV}"

echo "[4/5] Running Task Cut-off Granularity Sweep..."
for CUTOFF in 1 2 4 8 16 32 64; do
    OMP_PLACES=cores OMP_PROC_BIND=spread \
        "${BIN}" --mode omp_arch --n 512 --d 512 --threads "${MAX_CORES}" --cutoff "${CUTOFF}" --csv >> "${CUTOFF_CSV}"
done

echo "[5/5] Collecting Hardware Performance Counters (perf stat)..."
if command -v perf &> /dev/null; then
    PERF_EVENTS="cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"
    for MODE in seq omp_naive omp_arch; do
        OMP_PLACES=cores OMP_PROC_BIND=spread \
            perf stat -e "${PERF_EVENTS}" -o "${RESULTS_DIR}/perf_${MODE}.txt" \
            "${BIN}" --mode "${MODE}" --n 512 --d 512 --threads "${MAX_CORES}" --cutoff 8 --no-verify || true
    done
else
    echo "perf not found or not permitted; skipping hardware counter collection."
fi

echo "Done! All results saved in: ${RESULTS_DIR}"
