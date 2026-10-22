#include "bvn/decomposer.hpp"
#include "bvn/euler_trail.hpp"
#include <cstdint>
#include <omp.h>
#include <vector>

namespace bvn {

namespace {

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

    // Unaligned contiguous int array: 16 vertices share a single 64-byte cache line (false sharing)
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

} // namespace

std::vector<ScheduleLeaf> decompose_omp_naive(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    int num_threads
) {
    std::vector<ScheduleLeaf> leaves(degree);
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
        omp_set_num_threads(num_threads);
        std::vector<WeightedEdge> red_edges;
        std::vector<WeightedEdge> blue_edges;
        euler_split_omp_naive(unit_edges, n, true, red_edges, blue_edges);
        int half_deg = degree >> 1;

        #pragma omp parallel num_threads(num_threads)
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

    return leaves;
}

} // namespace bvn
