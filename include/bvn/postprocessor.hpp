#pragma once

#include "bvn/types.hpp"
#include <vector>

namespace bvn {

/**
 * Phase 4: Cleanup & Post-Processing
 * Filters out phantom traffic inserted in Phase 1 by comparing the D leaf permutations against
 * the original demand matrix M, replacing dummy connections with -1 (no-op).
 */
void cleanup_phantom_traffic(
    const Matrix& original_demand_matrix,
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
    const Matrix& original_demand_matrix,
    int n,
    const std::vector<ScheduleLeaf>& schedules
);

} // namespace bvn
