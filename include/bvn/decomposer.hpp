#pragma once

#include "bvn/types.hpp"
#include <vector>

namespace bvn {

/**
 * Sequential Eulerian split for a D-regular subgraph represented as WeightedEdges.
 * Used by the sequential baseline and inside disjoint subtrees below the task cut-off.
 */
void euler_split_sequential(
    const std::vector<WeightedEdge>& edges,
    int n,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
);

/**
 * Recursive sequential decomposition from degree D down to degree 1 leaves.
 */
void decompose_recursive_seq(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int leaf_offset,
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
 * (alignas(64) cursors, parity pre-halving, Flat CSR, calibrated task cut-off).
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
