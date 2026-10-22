#pragma once

#include "bvn/types.hpp"
#include <cstdint>
#include <vector>

namespace bvn {

constexpr int kMaxRecursionDepth = 20;

/**
 * Pre-allocated per-thread scratchpad workspace to eliminate dynamic heap allocations
 * (malloc/free contention and page faults) across the recursion tree.
 */
struct EulerWorkspace {
    std::vector<int> deg;
    std::vector<uint8_t> odd_color;
    std::vector<int> row_ptr;
    std::vector<FlatEdge> adj;
    std::vector<int> fill_ptr;
    std::vector<int> head;
    std::vector<uint8_t> used;
    std::vector<WeightedEdge> level_red[kMaxRecursionDepth];
    std::vector<WeightedEdge> level_blue[kMaxRecursionDepth];

    void ensure_capacity(int n, int max_csr_deg, int max_level_deg, int max_edges) {
        int sub_max = std::min(max_edges, n * max_csr_deg);
        if (static_cast<int>(deg.size()) < 2 * n) deg.resize(2 * n, 0);
        if (static_cast<int>(odd_color.size()) < sub_max) odd_color.resize(sub_max);
        if (static_cast<int>(row_ptr.size()) < 2 * n + 1) row_ptr.resize(2 * n + 1);
        if (static_cast<int>(adj.size()) < 2 * sub_max) adj.resize(2 * sub_max);
        if (static_cast<int>(fill_ptr.size()) < 2 * n) fill_ptr.resize(2 * n);
        if (static_cast<int>(head.size()) < 2 * n) head.resize(2 * n);
        if (static_cast<int>(used.size()) < sub_max) used.resize(sub_max);
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
 * Sequential Eulerian split using a pre-allocated workspace (zero heap allocations for CSR buffers).
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
 * (alignas(64) cursors, parity pre-halving, Thread-Local Workspace, calibrated task cut-off).
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
