#include "bvn/pipeline.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <omp.h>
#include <vector>

namespace bvn {

namespace {

struct OpenTrail {
    int start_v;
    int end_v;
    std::vector<int> edges;
};

/**
 * Helper to stitch and color open trails in the super-graph H without copying edge vectors.
 * Every vertex in H is guaranteed to have even degree because the underlying bipartite graph
 * is Eulerian and all closed trails & internal trail vertices consume pairs of edges.
 */
void color_open_trails(
    const std::vector<OpenTrail>& open_trails,
    int num_vertices,
    std::vector<uint8_t>& edge_color
) {
    if (open_trails.empty()) return;

    const int K = static_cast<int>(open_trails.size());
    struct SuperAdj {
        int to;
        int trail_id;
        bool forward; // true: start_v -> end_v, false: end_v -> start_v
    };

    std::vector<int> deg(num_vertices, 0);
    std::vector<int> active_vertices;
    for (int t = 0; t < K; ++t) {
        if (deg[open_trails[t].start_v] == 0) active_vertices.push_back(open_trails[t].start_v);
        deg[open_trails[t].start_v]++;
        if (deg[open_trails[t].end_v] == 0) active_vertices.push_back(open_trails[t].end_v);
        deg[open_trails[t].end_v]++;
    }

    std::vector<int> offset(num_vertices + 1, 0);
    int total_inc = 0;
    for (int v : active_vertices) {
        offset[v] = total_inc;
        total_inc += deg[v];
    }
    std::vector<int> head = offset;
    std::vector<int> end_offset(num_vertices, 0);
    for (int v : active_vertices) {
        end_offset[v] = offset[v] + deg[v];
    }

    std::vector<SuperAdj> adj(total_inc);
    std::vector<int> fill_ptr = offset;
    for (int t = 0; t < K; ++t) {
        int u = open_trails[t].start_v;
        int v = open_trails[t].end_v;
        adj[fill_ptr[u]++] = {v, t, true};
        adj[fill_ptr[v]++] = {u, t, false};
    }

    std::vector<uint8_t> used_trail(K, 0);

    // Extract closed cycles of super-edges in H
    for (int root : active_vertices) {
        while (head[root] < end_offset[root]) {
            int curr = root;
            uint8_t cur_color = 0;
            bool walked_any = false;

            while (true) {
                int chosen_trail = -1;
                int next_v = -1;
                bool forward = true;

                while (head[curr] < end_offset[curr]) {
                    const auto& item = adj[head[curr]++];
                    if (!used_trail[item.trail_id]) {
                        used_trail[item.trail_id] = 1;
                        chosen_trail = item.trail_id;
                        next_v = item.to;
                        forward = item.forward;
                        break;
                    }
                }

                if (chosen_trail == -1) {
                    break;
                }

                walked_any = true;
                const auto& edges = open_trails[chosen_trail].edges;
                if (forward) {
                    for (size_t i = 0; i < edges.size(); ++i) {
                        edge_color[edges[i]] = cur_color;
                        cur_color ^= 1;
                    }
                } else {
                    for (size_t i = edges.size(); i > 0; --i) {
                        edge_color[edges[i - 1]] = cur_color;
                        cur_color ^= 1;
                    }
                }

                curr = next_v;
                if (curr == root) {
                    break;
                }
            }
            (void)walked_any;
        }
    }
}

/**
 * Sequential Eulerian split for a D-regular subgraph represented as WeightedEdges.
 * Used as baseline (ExecMode::SEQ) and below the task cut-off in ExecMode::OMP_ARCH.
 */
void euler_split_sequential(
    const std::vector<WeightedEdge>& edges,
    int n,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
) {
    // Count odd-weight edges per vertex to build Flat CSR directly
    std::vector<int> deg(2 * n, 0);
    int E = 0;
    for (const auto& e : edges) {
        if (e.w & 1) {
            deg[e.u]++;
            deg[n + e.v]++;
            ++E;
        }
    }

    std::vector<uint8_t> odd_color(E, 0);

    if (E > 0) {
        std::vector<int> row_ptr(2 * n + 1, 0);
        for (int i = 0; i < 2 * n; ++i) {
            row_ptr[i + 1] = row_ptr[i] + deg[i];
        }

        std::vector<FlatEdge> adj(2 * E);
        std::vector<int> fill_ptr(row_ptr.begin(), row_ptr.begin() + 2 * n);
        int odd_id = 0;
        for (const auto& e : edges) {
            if (e.w & 1) {
                int u = e.u;
                int v = n + e.v;
                adj[fill_ptr[u]++] = {v, odd_id};
                adj[fill_ptr[v]++] = {u, odd_id};
                ++odd_id;
            }
        }

        std::vector<int> head(row_ptr.begin(), row_ptr.begin() + 2 * n);
        std::vector<uint8_t> used(E, 0);

        for (int root = 0; root < n; ++root) {
            while (head[root] < row_ptr[root + 1]) {
                int curr = root;
                uint8_t color = 0;
                while (true) {
                    int chosen_id = -1;
                    int next_v = -1;
                    while (head[curr] < row_ptr[curr + 1]) {
                        const FlatEdge& fe = adj[head[curr]++];
                        if (!used[fe.edge_id]) {
                            used[fe.edge_id] = 1;
                            chosen_id = fe.edge_id;
                            next_v = fe.to;
                            break;
                        }
                    }
                    if (chosen_id == -1) break;
                    odd_color[chosen_id] = color;
                    color ^= 1;
                    curr = next_v;
                    if (curr == root) break;
                }
            }
        }
    }

    // Single linear O(E) pass combining half-weights and colored odd remainders (preserves uniqueness, zero sorting!)
    out_red.clear();
    out_blue.clear();
    out_red.reserve(edges.size());
    out_blue.reserve(edges.size());

    int odd_id = 0;
    for (const auto& e : edges) {
        int half = e.w >> 1;
        int w_red = half;
        int w_blue = half;
        if (e.w & 1) {
            if (odd_color[odd_id] == 0) {
                ++w_red;
            } else {
                ++w_blue;
            }
            ++odd_id;
        }
        if (w_red > 0) out_red.push_back({e.u, e.v, w_red});
        if (w_blue > 0) out_blue.push_back({e.u, e.v, w_blue});
    }
}

void decompose_recursive_seq(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int leaf_offset,
    std::vector<ScheduleLeaf>& out_leaves
) {
    if (degree == 1) {
        ScheduleLeaf& leaf = out_leaves[leaf_offset];
        leaf.weight = 1;
        leaf.perm.assign(n, -1);
        for (const auto& e : edges) {
            if (e.w > 0) {
                leaf.perm[e.u] = e.v;
            }
        }
        return;
    }

    std::vector<WeightedEdge> red_edges;
    std::vector<WeightedEdge> blue_edges;
    euler_split_sequential(edges, n, red_edges, blue_edges);

    int half_deg = degree >> 1;
    decompose_recursive_seq(red_edges, n, half_deg, leaf_offset, out_leaves);
    decompose_recursive_seq(blue_edges, n, half_deg, leaf_offset + half_deg, out_leaves);
}

// ============================================================================
// NAIVE OPENMP IMPLEMENTATION (Exposes False Sharing & Task Overhead)
// ============================================================================

void euler_split_omp_naive(
    const std::vector<WeightedEdge>& unit_edges,
    int n,
    bool parallel_walk,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
) {
    const int E = static_cast<int>(unit_edges.size());
    std::vector<int> deg(2 * n, 0);
    for (int id = 0; id < E; ++id) {
        deg[unit_edges[id].u]++;
        deg[n + unit_edges[id].v]++;
    }

    std::vector<int> row_ptr(2 * n + 1, 0);
    for (int i = 0; i < 2 * n; ++i) {
        row_ptr[i + 1] = row_ptr[i] + deg[i];
    }

    std::vector<FlatEdge> adj(2 * E);
    std::vector<int> fill_ptr(row_ptr.begin(), row_ptr.begin() + 2 * n);
    for (int id = 0; id < E; ++id) {
        int u = unit_edges[id].u;
        int v = n + unit_edges[id].v;
        adj[fill_ptr[u]++] = {v, id};
        adj[fill_ptr[v]++] = {u, id};
    }

    // Naive: unaligned contiguous int array -> 16 vertices share a single 64B cache line!
    std::vector<int> head(row_ptr.begin(), row_ptr.begin() + 2 * n);
    std::vector<uint8_t> claimed(E, 0);
    std::vector<uint8_t> edge_color(E, 0);

    auto claim_next = [&](int x, int& out_to) -> int {
        const int end = row_ptr[x + 1];
        while (true) {
            int pos;
            #pragma omp atomic capture
            pos = head[x]++;
            if (pos >= end) return -1;
            int eid = adj[pos].edge_id;
            uint8_t prev = __atomic_exchange_n(&claimed[eid], 1, __ATOMIC_ACQ_REL);
            if (prev == 0) {
                out_to = adj[pos].to;
                return eid;
            }
        }
    };

    std::vector<OpenTrail> global_open_trails;

    #pragma omp parallel if(parallel_walk)
    {
        std::vector<OpenTrail> local_open;
        std::vector<int> trail_edges;

        #pragma omp for schedule(dynamic, 4)
        for (int root = 0; root < 2 * n; ++root) {
            while (true) {
                int next_v = -1;
                int first_e = claim_next(root, next_v);
                if (first_e == -1) break;

                trail_edges.clear();
                trail_edges.push_back(first_e);
                int curr = next_v;

                while (true) {
                    int to_v = -1;
                    int eid = claim_next(curr, to_v);
                    if (eid == -1) break;
                    trail_edges.push_back(eid);
                    curr = to_v;
                    if (curr == root) break;
                }

                if (curr == root) {
                    for (size_t i = 0; i < trail_edges.size(); ++i) {
                        edge_color[trail_edges[i]] = static_cast<uint8_t>(i & 1);
                    }
                } else {
                    local_open.push_back({root, curr, trail_edges});
                }
            }
        }

        if (!local_open.empty()) {
            #pragma omp critical
            {
                for (auto& t : local_open) {
                    global_open_trails.push_back(std::move(t));
                }
            }
        }
    }

    color_open_trails(global_open_trails, 2 * n, edge_color);

    out_red.clear();
    out_blue.clear();
    out_red.reserve(E / 2);
    out_blue.reserve(E / 2);
    for (int id = 0; id < E; ++id) {
        if (edge_color[id] == 0) {
            out_red.push_back(unit_edges[id]);
        } else {
            out_blue.push_back(unit_edges[id]);
        }
    }
}

void decompose_recursive_omp_naive(
    std::vector<WeightedEdge>& unit_edges,
    int n,
    int degree,
    int depth,
    int leaf_offset,
    std::vector<ScheduleLeaf>& out_leaves
) {
    if (degree == 1) {
        ScheduleLeaf& leaf = out_leaves[leaf_offset];
        leaf.weight = 1;
        leaf.perm.assign(n, -1);
        for (const auto& e : unit_edges) {
            leaf.perm[e.u] = e.v;
        }
        return;
    }

    std::vector<WeightedEdge> red_edges;
    std::vector<WeightedEdge> blue_edges;
    euler_split_omp_naive(unit_edges, n, (depth == 0), red_edges, blue_edges);

    int half_deg = degree >> 1;

    // Naive spawns tasks all the way down to degree == 1 without cutoff
    #pragma omp task shared(out_leaves, red_edges) firstprivate(n, half_deg, depth, leaf_offset)
    {
        decompose_recursive_omp_naive(red_edges, n, half_deg, depth + 1, leaf_offset, out_leaves);
    }

    #pragma omp task shared(out_leaves, blue_edges) firstprivate(n, half_deg, depth, leaf_offset)
    {
        decompose_recursive_omp_naive(blue_edges, n, half_deg, depth + 1, leaf_offset + half_deg, out_leaves);
    }

    #pragma omp taskwait
}

// ============================================================================
// ARCHITECTURE-AWARE OPENMP IMPLEMENTATION (OMP_ARCH)
// - alignas(64) Cache-Line Isolated Atomic Cursors (Zero False Sharing)
// - Parallel Even-Multiplicity Pre-Halving (Zero Sorting Overhead)
// - Flat CSR Adjacency Representation
// - Calibrated Task Cut-Off for L1/L2 Cache Locality
// ============================================================================

void euler_split_omp_arch(
    const std::vector<WeightedEdge>& edges,
    int n,
    bool parallel_walk,
    int num_threads,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
) {
    std::vector<int> deg(2 * n, 0);
    int E = 0;
    for (const auto& e : edges) {
        if (e.w & 1) {
            deg[e.u]++;
            deg[n + e.v]++;
            ++E;
        }
    }

    std::vector<uint8_t> edge_color(E, 0);

    if (E > 0) {
        std::vector<int> row_ptr(2 * n + 1, 0);
        for (int i = 0; i < 2 * n; ++i) {
            row_ptr[i + 1] = row_ptr[i] + deg[i];
        }

        std::vector<FlatEdge> adj(2 * E);
        std::vector<int> fill_ptr(row_ptr.begin(), row_ptr.begin() + 2 * n);
        int odd_id = 0;
        for (const auto& e : edges) {
            if (e.w & 1) {
                int u = e.u;
                int v = n + e.v;
                adj[fill_ptr[u]++] = {v, odd_id};
                adj[fill_ptr[v]++] = {u, odd_id};
                ++odd_id;
            }
        }

        // 64-byte cache-line aligned cursors: prevents cross-core and cross-socket MESI invalidations
        std::vector<AlignedCursor> cursors(2 * n);
        for (int i = 0; i < 2 * n; ++i) {
            cursors[i].head = row_ptr[i];
            cursors[i].deg = row_ptr[i + 1];
        }

        std::vector<uint8_t> claimed(E, 0);

        auto claim_next_aligned = [&](int x, int& out_to) -> int {
            const int end = cursors[x].deg;
            while (true) {
                int pos;
                #pragma omp atomic capture
                pos = cursors[x].head++;
                if (pos >= end) return -1;
                int eid = adj[pos].edge_id;
                uint8_t prev = __atomic_exchange_n(&claimed[eid], 1, __ATOMIC_ACQ_REL);
                if (prev == 0) {
                    out_to = adj[pos].to;
                    return eid;
                }
            }
        };

        std::vector<OpenTrail> global_open_trails;

        #pragma omp parallel if(parallel_walk) num_threads(num_threads)
        {
            std::vector<OpenTrail> local_open;
            std::vector<int> trail_edges;
            trail_edges.reserve(256);

            // Static/guided chunking preserves spatial locality across contiguous vertex ranges per core/CCX
            #pragma omp for schedule(guided)
            for (int root = 0; root < 2 * n; ++root) {
                while (cursors[root].head < cursors[root].deg) {
                    int next_v = -1;
                    int first_e = claim_next_aligned(root, next_v);
                    if (first_e == -1) break;

                    trail_edges.clear();
                    trail_edges.push_back(first_e);
                    int curr = next_v;

                    while (true) {
                        int to_v = -1;
                        int eid = claim_next_aligned(curr, to_v);
                        if (eid == -1) break;
                        trail_edges.push_back(eid);
                        curr = to_v;
                        if (curr == root) break;
                    }

                    if (curr == root) {
                        for (size_t i = 0; i < trail_edges.size(); ++i) {
                            edge_color[trail_edges[i]] = static_cast<uint8_t>(i & 1);
                        }
                    } else {
                        local_open.push_back({root, curr, trail_edges});
                    }
                }
            }

            if (!local_open.empty()) {
                #pragma omp critical
                {
                    for (auto& t : local_open) {
                        global_open_trails.push_back(std::move(t));
                    }
                }
            }
        }

        color_open_trails(global_open_trails, 2 * n, edge_color);
    }

    out_red.clear();
    out_blue.clear();
    out_red.reserve(edges.size());
    out_blue.reserve(edges.size());

    int odd_id = 0;
    for (const auto& e : edges) {
        int half = e.w >> 1;
        int w_red = half;
        int w_blue = half;
        if (e.w & 1) {
            if (edge_color[odd_id] == 0) {
                ++w_red;
            } else {
                ++w_blue;
            }
            ++odd_id;
        }
        if (w_red > 0) out_red.push_back({e.u, e.v, w_red});
        if (w_blue > 0) out_blue.push_back({e.u, e.v, w_blue});
    }
}

void decompose_recursive_omp_arch(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int depth,
    int cutoff_degree,
    int num_threads,
    int leaf_offset,
    std::vector<ScheduleLeaf>& out_leaves
) {
    if (degree <= cutoff_degree || degree == 1) {
        decompose_recursive_seq(edges, n, degree, leaf_offset, out_leaves);
        return;
    }

    std::vector<WeightedEdge> red_edges;
    std::vector<WeightedEdge> blue_edges;

    // Subtrees at depth >= 1 are completely disjoint in memory: split without atomic lock instructions!
    euler_split_sequential(edges, n, red_edges, blue_edges);

    int half_deg = degree >> 1;

    #pragma omp task shared(out_leaves, red_edges) firstprivate(n, half_deg, depth, cutoff_degree, num_threads, leaf_offset)
    {
        decompose_recursive_omp_arch(
            red_edges, n, half_deg, depth + 1, cutoff_degree, num_threads, leaf_offset, out_leaves
        );
    }

    // Execute right branch directly on the current thread to halve task creation overhead and preserve L1/L2 locality
    decompose_recursive_omp_arch(
        blue_edges, n, half_deg, depth + 1, cutoff_degree, num_threads, leaf_offset + half_deg, out_leaves
    );

    #pragma omp taskwait
}

} // namespace

std::vector<ScheduleLeaf> phase2_3_euler_decompose(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
) {
    std::vector<ScheduleLeaf> leaves(degree);

    if (cfg.mode == ExecMode::SEQ) {
        std::vector<WeightedEdge> initial_edges;
        initial_edges.reserve(static_cast<size_t>(n) * 16);
        for (int u = 0; u < n; ++u) {
            const int* row = &regularized_matrix[static_cast<size_t>(u) * n];
            for (int v = 0; v < n; ++v) {
                if (row[v] > 0) {
                    initial_edges.push_back({u, v, row[v]});
                }
            }
        }
        decompose_recursive_seq(initial_edges, n, degree, 0, leaves);
    } else if (cfg.mode == ExecMode::OMP_NAIVE) {
        // Naive mode expands all multiplicities into unit edges
        std::vector<WeightedEdge> unit_edges;
        unit_edges.reserve(static_cast<size_t>(n) * degree);
        for (int u = 0; u < n; ++u) {
            const int* row = &regularized_matrix[static_cast<size_t>(u) * n];
            for (int v = 0; v < n; ++v) {
                int w = row[v];
                for (int k = 0; k < w; ++k) {
                    unit_edges.push_back({u, v, 1});
                }
            }
        }
        if (degree == 1) {
            decompose_recursive_omp_naive(unit_edges, n, 1, 0, 0, leaves);
        } else {
            // Root split uses parallel walk across all threads
            omp_set_num_threads(cfg.num_threads);
            std::vector<WeightedEdge> red_edges;
            std::vector<WeightedEdge> blue_edges;
            euler_split_omp_naive(unit_edges, n, true, red_edges, blue_edges);
            int half_deg = degree >> 1;

            #pragma omp parallel num_threads(cfg.num_threads)
            {
                #pragma omp single nowait
                {
                    #pragma omp task shared(leaves, red_edges) firstprivate(n, half_deg)
                    {
                        decompose_recursive_omp_naive(red_edges, n, half_deg, 1, 0, leaves);
                    }
                    #pragma omp task shared(leaves, blue_edges) firstprivate(n, half_deg)
                    {
                        decompose_recursive_omp_naive(blue_edges, n, half_deg, 1, half_deg, leaves);
                    }
                    #pragma omp taskwait
                }
            }
        }
    } else {
        // OMP_ARCH mode: weighted multiplicity compression + aligned cursors + NUMA-friendly subtree tasks
        std::vector<WeightedEdge> initial_edges;
        initial_edges.reserve(static_cast<size_t>(n) * 16);
        for (int u = 0; u < n; ++u) {
            const int* row = &regularized_matrix[static_cast<size_t>(u) * n];
            for (int v = 0; v < n; ++v) {
                if (row[v] > 0) {
                    initial_edges.push_back({u, v, row[v]});
                }
            }
        }
        if (degree <= cfg.cutoff_degree || degree == 1) {
            decompose_recursive_seq(initial_edges, n, degree, 0, leaves);
        } else {
            // Root level (Depth 0): parallel walk with alignas(64) cursors across all threads
            std::vector<WeightedEdge> red_edges;
            std::vector<WeightedEdge> blue_edges;
            euler_split_omp_arch(initial_edges, n, true, cfg.num_threads, red_edges, blue_edges);
            int half_deg = degree >> 1;

            #pragma omp parallel num_threads(cfg.num_threads)
            {
                #pragma omp single nowait
                {
                    #pragma omp task shared(leaves, red_edges) firstprivate(n, half_deg)
                    {
                        decompose_recursive_omp_arch(
                            red_edges, n, half_deg, 1, cfg.cutoff_degree, cfg.num_threads, 0, leaves
                        );
                    }
                    decompose_recursive_omp_arch(
                        blue_edges, n, half_deg, 1, cfg.cutoff_degree, cfg.num_threads, half_deg, leaves
                    );
                    #pragma omp taskwait
                }
            }
        }
    }

    return leaves;
}

} // namespace bvn
