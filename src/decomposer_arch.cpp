#include "bvn/decomposer.hpp"
#include "bvn/euler_trail.hpp"
#include <cstdint>
#include <omp.h>
#include <vector>

namespace bvn {

namespace {

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

    // Subtrees at depth >= 1 are completely disjoint in memory: split without atomic lock instructions
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

std::vector<ScheduleLeaf> decompose_omp_arch(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
) {
    std::vector<ScheduleLeaf> leaves(degree);
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

    return leaves;
}

} // namespace bvn
