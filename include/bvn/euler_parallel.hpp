#pragma once

#include "bvn/decomposer.hpp"
#include "bvn/types.hpp"
#include <cstdint>
#include <vector>

namespace bvn {

/**
 * Off by default (Config::par_walk_min_edges = 0): it trades work efficiency for parallelism.
 * The serial TZCNT/BLSR walk keeps 1 bit per (u, v) slot in L2/L3, while this partition needs
 * ~16 B per odd edge plus pairing sorts, ~13x more per edge on a 6-core Zen 2 (N = 2048), so it
 * only pays off when many more threads than that would otherwise idle at the tree top.
 */

/**
 * Parallel balanced 2-coloring of the odd-residual subgraph (replaces the serial Euler walk).
 *
 * Instead of walking an Euler circuit, every vertex pairs up its incident odd edges
 * (each vertex has an even number of them). Paired edges must get different colors, and since
 * each edge has exactly one partner at its L endpoint and one at its R endpoint, the pairing
 * relation is a union of even cycles: alternating colors along each cycle gives every vertex
 * exactly half Red / half Blue, which is all the Euler split needs.
 *
 * Phases (all data-parallel through `omp taskloop`, so it can run inside the task tree):
 *   1. Pairing: consecutive odd edges of each vertex are paired. Edges grouped by L endpoint
 *      (grouped_by_u) pair on the L side from run offsets alone; otherwise, and always on the
 *      R side, a counting sort by endpoint comes first.
 *   2. Segments: tasks start at unclaimed edges and extend along the cycle in both directions,
 *      claiming edges with CAS and assigning colors relative to the segment start. A segment
 *      stops where it meets an edge claimed by another segment and records that boundary.
 *   3. Stitching: each boundary says "these two segments' colors differ across this link";
 *      a sequential union-find with parity picks a flip bit per segment (#segments << #edges).
 *   4. Final color = relative color XOR segment flip.
 *
 * ou/ov: L and R endpoints of the E odd edges. color[k] = 0 (Red) or 1 (Blue).
 * Must be called from inside a parallel region (or it runs on one thread).
 */
void color_odd_edges_parallel(
    int n,
    int E,
    const uint16_t* ou,
    const uint16_t* ov,
    bool grouped_by_u,
    uint8_t* color,
    int num_tasks
);

/**
 * Data-parallel Euler split of a weighted edge list (same contract as euler_split_sequential_ws):
 * even parts halve directly, odd residuals are colored with color_odd_edges_parallel.
 * Returns true if the split is symmetric (no odd edges); then only out_red is filled.
 */
bool euler_split_parallel(
    const std::vector<WeightedEdge>& edges,
    int n,
    int num_tasks,
    EulerWorkspace& ws,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
);

} // namespace bvn
