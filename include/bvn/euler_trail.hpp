#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bvn {

/**
 * Represents an open Eulerian walk segment resulting from concurrent thread edge claiming.
 */
struct OpenTrail {
    int start_v;
    int end_v;
    std::vector<int> edges;
};

/**
 * Stitches and colors open trails in the super-graph H without copying edge vectors.
 * Every vertex in H is guaranteed to have even degree because the underlying bipartite graph
 * is Eulerian and all closed trails & internal trail vertices consume pairs of edges.
 */
void color_open_trails(
    const std::vector<OpenTrail>& open_trails,
    int num_vertices,
    std::vector<uint8_t>& edge_color
);

} // namespace bvn
