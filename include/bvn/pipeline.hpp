#pragma once

#include "bvn/types.hpp"
#include <vector>

namespace bvn {

/**
 * Generates a synthetic N x N integer traffic demand matrix M where max row/col sum <= target_degree.
 */
std::vector<int> generate_traffic_matrix(const Config& cfg, int64_t& out_total_packets);

/**
 * Phase 1: Regularization (Padding)
 * Transforms an arbitrary traffic demand matrix M (N x N) with max line sum <= D
 * into a D-regular bipartite multigraph M_reg (N x N) where every row and column sums to D (a power of 2).
 */
std::vector<int> phase1_regularize(
    const std::vector<int>& demand_matrix,
    int n,
    int target_degree,
    ExecMode mode,
    int& out_regularized_degree,
    int64_t& out_phantom_added
);

/**
 * Phases 2 & 3: Eulerian Divide-and-Conquer & Degree-1 Leaf Extraction
 * Recursively decomposes a D-regular bipartite multigraph into D permutation schedules of degree 1.
 */
std::vector<ScheduleLeaf> phase2_3_euler_decompose(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
);

/**
 * Phase 4: Cleanup & Post-Processing
 * Filters out phantom traffic inserted in Phase 1 by comparing the D leaf permutations against
 * the original demand matrix M, replacing dummy connections with -1 (no-op).
 */
void phase4_cleanup_phantom(
    const std::vector<int>& original_demand_matrix,
    int n,
    ExecMode mode,
    std::vector<ScheduleLeaf>& schedules
);

/**
 * Strict mathematical verifier:
 * 1. Verifies each schedule leaf is conflict-free (at most one input connected to each output).
 * 2. Verifies the sum of all schedules equals the exact original traffic matrix M_ij.
 */
bool verify_decomposition(
    const std::vector<int>& original_demand_matrix,
    int n,
    const std::vector<ScheduleLeaf>& schedules
);

} // namespace bvn
