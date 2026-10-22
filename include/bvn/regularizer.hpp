#pragma once

#include "bvn/types.hpp"
#include <vector>

namespace bvn {

/**
 * Phase 1: Regularization (Padding)
 * Transforms an arbitrary traffic demand matrix M (N x N) with max line sum <= D
 * into a D-regular bipartite multigraph M_reg (N x N) where every row and column sums to D (a power of 2).
 */
std::vector<int> regularize_matrix(
    const std::vector<int>& demand_matrix,
    int n,
    int target_degree,
    ExecMode mode,
    int& out_regularized_degree,
    int64_t& out_phantom_added
);

} // namespace bvn
