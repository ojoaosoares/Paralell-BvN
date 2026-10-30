#pragma once

#include "bvn/types.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

namespace bvn {

constexpr int kMaxRecursionDepth = 20;

/**
 * Pre-allocated per-thread scratchpad workspace using 64-bit Bitmask Adjacency + Hardware TZCNT/BLSR
 * to eliminate dynamic heap allocations and pointer/index chasing across the recursion tree.
 */
struct EulerWorkspace {
    int num_words = 0;
    std::vector<uint64_t> adj_L;      // Level-1 Bitmask adjacency for Left vertices (N * num_words)
    std::vector<uint64_t> adj_R;      // Level-1 Bitmask adjacency for Right vertices (N * num_words)
    std::vector<uint64_t> summary_L;  // Level-0 Summary Bitmask of non-zero words for Left vertices (size N)
    std::vector<uint64_t> summary_R;  // Level-0 Summary Bitmask of non-zero words for Right vertices (size N)
    std::vector<uint64_t> has_heavy_L;// L1 Bitset (num_words uint64_t = 256B for N=2048) marking rows u with heavy odd edges
    std::vector<uint64_t> red_mask;   // Used only for vertices u where has_heavy_L bit is set (N * num_words)
    std::vector<WeightedEdge> heavy_odd_edges; // Odd edges with w >= 3 (typically < 1% of edges)
    std::vector<WeightedEdge> level_red[kMaxRecursionDepth];
    std::vector<WeightedEdge> level_blue[kMaxRecursionDepth];

    // Zero-contention thread-local telemetry counters
    int64_t splits = 0;
    int64_t sym_pruned = 0;
    int64_t unit_odd_fast = 0;
    int64_t heavy_odd = 0;
    int64_t deg2_calls = 0;
    int64_t edges_scanned = 0;
    int64_t odd_edges_walked = 0;
    double active_time_ms = 0.0;

    void reset_counters() {
        splits = 0;
        sym_pruned = 0;
        unit_odd_fast = 0;
        heavy_odd = 0;
        deg2_calls = 0;
        edges_scanned = 0;
        odd_edges_walked = 0;
        active_time_ms = 0.0;
    }

    void ensure_capacity(int n, int max_csr_deg, int max_level_deg, int max_edges) {
        (void)max_csr_deg;
        num_words = (n + 63) >> 6;
        const size_t mask_size = static_cast<size_t>(n) * num_words;
        if (adj_L.size() < mask_size) adj_L.assign(mask_size, 0ULL);
        if (adj_R.size() < mask_size) adj_R.assign(mask_size, 0ULL);
        if (summary_L.size() < static_cast<size_t>(n)) summary_L.assign(n, 0ULL);
        if (summary_R.size() < static_cast<size_t>(n)) summary_R.assign(n, 0ULL);
        if (has_heavy_L.size() < static_cast<size_t>(num_words)) has_heavy_L.assign(num_words, 0ULL);
        if (red_mask.size() < mask_size) red_mask.assign(mask_size, 0ULL);
        if (heavy_odd_edges.capacity() < static_cast<size_t>(n * 4)) {
            heavy_odd_edges.reserve(static_cast<size_t>(n * 4));
        }

        int d = 0;
        for (int cur_deg = max_level_deg; cur_deg > 1 && d < kMaxRecursionDepth; cur_deg >>= 1, ++d) {
            size_t cap = static_cast<size_t>(std::min(max_edges, n * (cur_deg >> 1)));
            if (level_red[d].capacity() < cap) level_red[d].reserve(cap);
            if (level_blue[d].capacity() < cap) level_blue[d].reserve(cap);
        }
    }
};

/**
 * Returns the calling thread's thread-local EulerWorkspace.
 */
EulerWorkspace& get_thread_workspace();

/**
 * Visits every EulerWorkspace ever created (one per OS thread, including nested-team threads).
 * Must not run concurrently with parallel regions that use the workspaces.
 */
void for_each_thread_workspace(const std::function<void(EulerWorkspace&)>& fn);

/**
 * Bitmask + TZCNT/BLSR Eulerian split using a pre-allocated workspace.
 * Returns true if the split is 100% symmetric (E_odd == 0, so Red and Blue subgraphs are identical).
 */
bool euler_split_sequential_ws(
    const std::vector<WeightedEdge>& edges,
    int n,
    EulerWorkspace& ws,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
);

/**
 * Recursive sequential decomposition using a pre-allocated Thread-Local EulerWorkspace.
 */
void decompose_recursive_seq_ws(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int depth,
    int leaf_offset,
    EulerWorkspace& ws,
    std::vector<ScheduleLeaf>& out_leaves
);

/**
 * Baseline Sequential Eulerian BvN Decomposition.
 */
std::vector<ScheduleLeaf> decompose_seq(
    const Matrix& regularized_matrix,
    int n,
    int degree
);

/**
 * Naive OpenMP Eulerian BvN Decomposition (unaligned cursors, unit-edge expansion, no task cutoff).
 */
std::vector<ScheduleLeaf> decompose_omp_naive(
    const Matrix& regularized_matrix,
    int n,
    int degree,
    int num_threads
);

/**
 * Architecture-Aware OpenMP Eulerian BvN Decomposition
 * (alignas(64) cursors, parity pre-halving, Bitmask TZCNT Workspace, calibrated task cut-off).
 */
std::vector<ScheduleLeaf> decompose_omp_arch(
    const Matrix& regularized_matrix,
    int n,
    int degree,
    const Config& cfg,
    PipelineStats* stats = nullptr
);

/**
 * Dispatcher for Phases 2 & 3 according to cfg.mode.
 */
std::vector<ScheduleLeaf> decompose_bvn(
    const Matrix& regularized_matrix,
    int n,
    int degree,
    const Config& cfg,
    PipelineStats* stats = nullptr
);

/**
 * Dynamically selects the optimal spatial block factor K (splitting N x N into K x K sub-matrices
 * of dimension N_b = N / K) based on switch port count N, OpenMP thread count, and L1d cache size.
 */
int select_optimal_block_factor(
    int n,
    int num_threads,
    int requested_k = 0,
    const Matrix* demand_matrix = nullptr,
    int target_degree = 0
);

/**
 * Executes the Spatial Block-Diagonal BvN Pipeline (OMP_BLOCK):
 * 1. Phase 1: Parallel sub-matrix extraction (N_b x N_b) + independent local phantom padding per block.
 * 2. Phase 2 & 3: Parallel L1-resident Bitmask Eulerian decomposition across K^2 sub-matrices
 *    and Latin-square diagonal schedule fusion into global N-port permutations.
 */
std::vector<ScheduleLeaf> run_block_bvn_pipeline(
    const Matrix& demand_matrix,
    const Config& cfg,
    PipelineStats& stats,
    double& out_phase1_ms,
    double& out_phase2_3_ms
);

} // namespace bvn
