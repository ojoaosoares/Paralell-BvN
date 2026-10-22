#include "bvn/decomposer.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace bvn {

bool euler_split_sequential_ws(
    const std::vector<WeightedEdge>& edges,
    int n,
    EulerWorkspace& ws,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
) {
    int* deg = ws.deg.data();
    std::fill(deg, deg + 2 * n, 0);

    int E = 0;
    for (const auto& e : edges) {
        if (e.w & 1) {
            deg[e.u]++;
            deg[n + e.v]++;
            ++E;
        }
    }

    // Symmetry Pruning: when E == 0, all weights are even so Red and Blue subtrees are 100% identical!
    if (E == 0) {
        out_red.clear();
        for (const auto& e : edges) {
            int half = e.w >> 1;
            if (half > 0) out_red.push_back({e.u, e.v, half});
        }
        return true;
    }

    uint8_t* odd_color = ws.odd_color.data();

    int* row_ptr = ws.row_ptr.data();
    int* fill_ptr = ws.fill_ptr.data();
    int* head = ws.head.data();
    row_ptr[0] = 0;
    for (int i = 0; i < 2 * n; ++i) {
        row_ptr[i + 1] = row_ptr[i] + deg[i];
        fill_ptr[i] = row_ptr[i];
        head[i] = row_ptr[i];
    }

    FlatEdge* adj = ws.adj.data();
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

    uint8_t* used = ws.used.data();
    std::fill(used, used + E, 0);

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

    out_red.clear();
    out_blue.clear();

    odd_id = 0;
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
    return false;
}

void decompose_recursive_seq_ws(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int depth,
    int leaf_offset,
    EulerWorkspace& ws,
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

    auto& red_edges = ws.level_red[depth];
    auto& blue_edges = ws.level_blue[depth];
    bool symmetric = euler_split_sequential_ws(edges, n, ws, red_edges, blue_edges);

    int half_deg = degree >> 1;
    decompose_recursive_seq_ws(red_edges, n, half_deg, depth + 1, leaf_offset, ws, out_leaves);

    if (symmetric) {
        for (int i = 0; i < half_deg; ++i) {
            out_leaves[leaf_offset + half_deg + i] = out_leaves[leaf_offset + i];
        }
    } else {
        decompose_recursive_seq_ws(blue_edges, n, half_deg, depth + 1, leaf_offset + half_deg, ws, out_leaves);
    }
}

EulerWorkspace& get_thread_workspace() {
    static thread_local EulerWorkspace tls_ws;
    return tls_ws;
}

std::vector<ScheduleLeaf> decompose_seq(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree
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

    EulerWorkspace& ws = get_thread_workspace();
    ws.ensure_capacity(n, degree, degree, static_cast<int>(initial_edges.size()));

    decompose_recursive_seq_ws(initial_edges, n, degree, 0, 0, ws, leaves);
    return leaves;
}

std::vector<ScheduleLeaf> decompose_bvn(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
) {
    if (cfg.mode == ExecMode::SEQ) {
        return decompose_seq(regularized_matrix, n, degree);
    }
    if (cfg.mode == ExecMode::OMP_NAIVE) {
        return decompose_omp_naive(regularized_matrix, n, degree, cfg.num_threads);
    }
    return decompose_omp_arch(regularized_matrix, n, degree, cfg);
}

} // namespace bvn
