#pragma once

#include "bvn/types.hpp"
#include <vector>

namespace bvn {

/**
 * Generates a synthetic N x N integer traffic demand matrix M where max row/col sum <= target_degree.
 */
Matrix generate_traffic_matrix(const Config& cfg, int64_t& out_total_packets);

} // namespace bvn
