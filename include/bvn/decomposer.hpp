#pragma once

#include "bvn/types.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace bvn {

constexpr int kMaxRecursionDepth = 20;

/**
 * Pre-allocated per-thread scratchpad workspace using 64-bit Bitmask Adjacency + Hardware TZCNT/BLSR
 * to eliminate dynamic heap allocations and pointer/index chasing across the recursion tree.
 */
struct EulerWorkspace {
    int num_words = 0;
    std::vector<uint64_t> adj_L;      // Bitmask adjacency for Left vertices (N * num_words)
    std::vector<uint64_t> adj_R;      // Bitmask adjacency for Right vertices (N * num_words)
    std::vector<uint64_t> red_mask;   // Bitmask of odd edges colored Red (N * num_words)
    std::vector<int> word_head_L;     // Active 64-bit word cursor for Left vertices (size N)
    std::vector<int> word_head_R;     // Active 64-bit word cursor for Right vertices (size N)
    std::vector<WeightedEdge> level_red[kMaxRecursionDepth];
    std::vector<WeightedEdge> level_blue[kMaxRecursionDepth];

    void ensure_capacity(int n, int max_csr_deg, int max_level_deg, int max_edges) {
        (void)max_csr_deg;
        num_words = (n + 63) >> 6;
        const size_t mask_size = static_cast<size_t>(n) * num_words;
        if (adj_L.size() < mask_size) adj_L.assign(mask_size, 0ULL);
        if (adj_R.size() < mask_size) adj_R.assign(mask_size, 0ULL);
        if (red_mask.size() < mask_size) red_mask.assign(mask_size, 0ULL);
        if (static_cast<int>(word_head_L.size()) < n) word_head_L.resize(n, 0);
        if (static_cast<int>(word_head_R.size()) < n) word_head_R.resize(n, 0);

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
    const std::vector<int>& regularized_matrix,
    int n,
    int degree
);

/**
 * Naive OpenMP Eulerian BvN Decomposition (unaligned cursors, unit-edge expansion, no task cutoff).
 */
std::vector<ScheduleLeaf> decompose_omp_naive(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    int num_threads
);

/**
 * Architecture-Aware OpenMP Eulerian BvN Decomposition
 * (alignas(64) cursors, parity pre-halving, Bitmask TZCNT Workspace, calibrated task cut-off).
 */
std::vector<ScheduleLeaf> decompose_omp_arch(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
);

/**
 * Dispatcher for Phases 2 & 3 according to cfg.mode.
 */
std::vector<ScheduleLeaf> decompose_bvn(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
);

} // namespace bvn
