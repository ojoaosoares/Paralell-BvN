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
    const int W = ws.num_words;
    uint64_t* __restrict adj_L = ws.adj_L.data();
    uint64_t* __restrict adj_R = ws.adj_R.data();
    uint64_t* __restrict has_heavy_L = ws.has_heavy_L.data();
    uint64_t* __restrict red_mask = ws.red_mask.data();
    uint64_t* __restrict summary_L = ws.summary_L.data();
    uint64_t* __restrict summary_R = ws.summary_R.data();

    out_red.clear();
    out_blue.clear();
    auto& heavy_odd = ws.heavy_odd_edges;
    heavy_odd.clear();

    int E = 0;
    for (const auto& e : edges) {
        int half = e.w >> 1;
        if (e.w & 1) {
            int uw = e.v >> 6;
            int vw = e.u >> 6;
            uint64_t v_bit = 1ULL << (e.v & 63);
            adj_L[e.u * W + uw] |= v_bit;
            adj_R[e.v * W + vw] |= (1ULL << (e.u & 63));
            summary_L[e.u] |= (1ULL << uw);
            summary_R[e.v] |= (1ULL << vw);
            if (half > 0) {
                has_heavy_L[e.u >> 6] |= (1ULL << (e.u & 63));
                red_mask[e.u * W + uw] |= v_bit; // Marks that (u, v) is a heavy odd edge (w >= 3)
                heavy_odd.push_back({e.u, e.v, half});
            }
            ++E;
        } else if (half > 0) {
            out_red.push_back({e.u, e.v, half});
            out_blue.push_back({e.u, e.v, half});
        }
    }

    ws.splits++;
    ws.edges_scanned += static_cast<int64_t>(edges.size());
    ws.odd_edges_walked += E;

    // Symmetry Pruning: when E == 0, all weights are even so Red and Blue subtrees are 100% identical!
    if (E == 0) {
        ws.sym_pruned++;
        return true;
    }

    // Fast path 1: all odd edges have w == 1 (heavy_odd is empty)
    // Emit {u, v, 1} directly to out_red and {next_u, v, 1} directly to out_blue during the walk!
    if (heavy_odd.empty()) {
        ws.unit_odd_fast++;
        for (int root = 0; root < n; ++root) {
            while (summary_L[root] != 0ULL) {
                int u = root;
                while (summary_L[u] != 0ULL) {
                    uint64_t sum_u = summary_L[u];
                    int uw = __builtin_ctzll(sum_u);
                    uint64_t lmask = adj_L[u * W + uw];
                    int v_bit = __builtin_ctzll(lmask);
                    int v = (uw << 6) + v_bit;

                    uint64_t new_lmask = lmask & (lmask - 1ULL);
                    adj_L[u * W + uw] = new_lmask;
                    if (new_lmask == 0ULL) {
                        summary_L[u] = sum_u & (sum_u - 1ULL);
                    }

                    int u_word = u >> 6;
                    uint64_t new_rmask_u = adj_R[v * W + u_word] & ~(1ULL << (u & 63));
                    adj_R[v * W + u_word] = new_rmask_u;
                    if (new_rmask_u == 0ULL) {
                        summary_R[v] &= ~(1ULL << u_word);
                    }
                    out_red.push_back({static_cast<uint16_t>(u), static_cast<uint16_t>(v), 1});

                    uint64_t sum_v = summary_R[v];
                    int vw = __builtin_ctzll(sum_v);
                    uint64_t rmask = adj_R[v * W + vw];
                    int u_bit = __builtin_ctzll(rmask);
                    int next_u = (vw << 6) + u_bit;

                    uint64_t new_rmask = rmask & (rmask - 1ULL);
                    adj_R[v * W + vw] = new_rmask;
                    if (new_rmask == 0ULL) {
                        summary_R[v] = sum_v & (sum_v - 1ULL);
                    }

                    int v_word = v >> 6;
                    uint64_t new_lmask_next = adj_L[next_u * W + v_word] & ~(1ULL << (v & 63));
                    adj_L[next_u * W + v_word] = new_lmask_next;
                    if (new_lmask_next == 0ULL) {
                        summary_L[next_u] &= ~(1ULL << v_word);
                    }
                    out_blue.push_back({static_cast<uint16_t>(next_u), static_cast<uint16_t>(v), 1});

                    u = next_u;
                    if (u == root) break;
                }
            }
        }
        return false;
    }

    ws.heavy_odd++;

    // Fast path 2: ALL odd edges have w >= 3 (heavy_odd.size() == E, e.g., bursty and upper levels of skewed):
    // Zero branches inside the walk, zero red_mask touches on L->R, unconditional bit-clear on R->L!
    if (static_cast<int>(heavy_odd.size()) == E) {
        for (int root = 0; root < n; ++root) {
            while (summary_L[root] != 0ULL) {
                int u = root;
                while (summary_L[u] != 0ULL) {
                    uint64_t sum_u = summary_L[u];
                    int uw = __builtin_ctzll(sum_u);
                    uint64_t lmask = adj_L[u * W + uw];
                    int v_bit = __builtin_ctzll(lmask);
                    int v = (uw << 6) + v_bit;

                    uint64_t new_lmask = lmask & (lmask - 1ULL);
                    adj_L[u * W + uw] = new_lmask;
                    if (new_lmask == 0ULL) {
                        summary_L[u] = sum_u & (sum_u - 1ULL);
                    }

                    int u_word = u >> 6;
                    uint64_t new_rmask_u = adj_R[v * W + u_word] & ~(1ULL << (u & 63));
                    adj_R[v * W + u_word] = new_rmask_u;
                    if (new_rmask_u == 0ULL) {
                        summary_R[v] &= ~(1ULL << u_word);
                    }

                    uint64_t sum_v = summary_R[v];
                    int vw = __builtin_ctzll(sum_v);
                    uint64_t rmask = adj_R[v * W + vw];
                    int u_bit = __builtin_ctzll(rmask);
                    int next_u = (vw << 6) + u_bit;

                    uint64_t new_rmask = rmask & (rmask - 1ULL);
                    adj_R[v * W + vw] = new_rmask;
                    if (new_rmask == 0ULL) {
                        summary_R[v] = sum_v & (sum_v - 1ULL);
                    }

                    int v_word = v >> 6;
                    uint64_t v_mask_bit = 1ULL << (v & 63);
                    uint64_t new_lmask_next = adj_L[next_u * W + v_word] & ~v_mask_bit;
                    adj_L[next_u * W + v_word] = new_lmask_next;
                    if (new_lmask_next == 0ULL) {
                        summary_L[next_u] &= ~(1ULL << v_word);
                    }
                    red_mask[next_u * W + v_word] &= ~v_mask_bit;

                    u = next_u;
                    if (u == root) break;
                }
            }
        }
    } else {
        // Mixed path (both w == 1 and w >= 3 odd edges exist):
        // Uses 256-byte L1-resident has_heavy_L bitset to bypass red_mask on vertices without heavy edges!
        for (int root = 0; root < n; ++root) {
            while (summary_L[root] != 0ULL) {
                int u = root;
                while (summary_L[u] != 0ULL) {
                    uint64_t sum_u = summary_L[u];
                    int uw = __builtin_ctzll(sum_u);
                    uint64_t lmask = adj_L[u * W + uw];
                    int v_bit = __builtin_ctzll(lmask);
                    int v = (uw << 6) + v_bit;

                    uint64_t new_lmask = lmask & (lmask - 1ULL);
                    adj_L[u * W + uw] = new_lmask;
                    if (new_lmask == 0ULL) {
                        summary_L[u] = sum_u & (sum_u - 1ULL);
                    }

                    int u_word = u >> 6;
                    uint64_t u_mask_bit = 1ULL << (u & 63);
                    uint64_t new_rmask_u = adj_R[v * W + u_word] & ~u_mask_bit;
                    adj_R[v * W + u_word] = new_rmask_u;
                    if (new_rmask_u == 0ULL) {
                        summary_R[v] &= ~(1ULL << u_word);
                    }
                    if ((has_heavy_L[u_word] & u_mask_bit) == 0ULL ||
                        (red_mask[u * W + uw] & (1ULL << v_bit)) == 0ULL) {
                        out_red.push_back({static_cast<uint16_t>(u), static_cast<uint16_t>(v), 1});
                    }

                    uint64_t sum_v = summary_R[v];
                    int vw = __builtin_ctzll(sum_v);
                    uint64_t rmask = adj_R[v * W + vw];
                    int u_bit = __builtin_ctzll(rmask);
                    int next_u = (vw << 6) + u_bit;

                    uint64_t new_rmask = rmask & (rmask - 1ULL);
                    adj_R[v * W + vw] = new_rmask;
                    if (new_rmask == 0ULL) {
                        summary_R[v] = sum_v & (sum_v - 1ULL);
                    }

                    int v_word = v >> 6;
                    uint64_t v_mask_bit = 1ULL << (v & 63);
                    uint64_t new_lmask_next = adj_L[next_u * W + v_word] & ~v_mask_bit;
                    adj_L[next_u * W + v_word] = new_lmask_next;
                    if (new_lmask_next == 0ULL) {
                        summary_L[next_u] &= ~(1ULL << v_word);
                    }
                    if ((has_heavy_L[vw] & (1ULL << u_bit)) == 0ULL) {
                        out_blue.push_back({static_cast<uint16_t>(next_u), static_cast<uint16_t>(v), 1});
                    } else {
                        uint64_t& rword_next = red_mask[next_u * W + v_word];
                        if ((rword_next & v_mask_bit) == 0ULL) {
                            out_blue.push_back({static_cast<uint16_t>(next_u), static_cast<uint16_t>(v), 1});
                        } else {
                            rword_next &= ~v_mask_bit; // Colored Blue: clear heavy-red bit!
                        }
                    }

                    u = next_u;
                    if (u == root) break;
                }
            }
        }
    }

    // Emit only the heavy odd edges (w >= 3) — 100% branchless (since he.w >= 1, both Red and Blue weights are >= 1)!
    for (const auto& he : heavy_odd) {
        has_heavy_L[he.u >> 6] = 0ULL;
        uint64_t& rword = red_mask[he.u * W + (he.v >> 6)];
        uint64_t bit = 1ULL << (he.v & 63);
        int is_red = (rword & bit) != 0ULL;
        rword &= ~bit; // Leaves red_mask 100% zeroed
        out_red.push_back({he.u, he.v, he.w + is_red});
        out_blue.push_back({he.u, he.v, he.w + (1 - is_red)});
    }
    return false;
}

namespace {

void extract_degree2_leaves(
    const std::vector<WeightedEdge>& edges,
    int n,
    EulerWorkspace& ws,
    ScheduleLeaf& leaf0,
    ScheduleLeaf& leaf1
) {
    ws.deg2_calls++;
    ws.edges_scanned += static_cast<int64_t>(edges.size());

    leaf0.weight = 1;
    leaf1.weight = 1;
    if (static_cast<int>(leaf0.perm.size()) != n) leaf0.perm.resize(n);
    if (static_cast<int>(leaf1.perm.size()) != n) leaf1.perm.resize(n);

    int* __restrict p0 = leaf0.perm.data();
    int* __restrict p1 = leaf1.perm.data();

    const int W = ws.num_words;
    uint64_t* __restrict adj_L = ws.adj_L.data();
    uint64_t* __restrict adj_R = ws.adj_R.data();
    uint64_t* __restrict summary_L = ws.summary_L.data();
    uint64_t* __restrict summary_R = ws.summary_R.data();

    int E = 0;
    for (const auto& e : edges) {
        if (e.w == 2) {
            p0[e.u] = e.v;
            p1[e.u] = e.v;
        } else if (e.w == 1) {
            int uw = e.v >> 6;
            int vw = e.u >> 6;
            adj_L[e.u * W + uw] |= (1ULL << (e.v & 63));
            adj_R[e.v * W + vw] |= (1ULL << (e.u & 63));
            summary_L[e.u] |= (1ULL << uw);
            summary_R[e.v] |= (1ULL << vw);
            ++E;
        }
    }

    ws.odd_edges_walked += E;
    if (E == 0) return;

    for (int root = 0; root < n; ++root) {
        while (summary_L[root] != 0ULL) {
            int u = root;
            while (summary_L[u] != 0ULL) {
                uint64_t sum_u = summary_L[u];
                int uw = __builtin_ctzll(sum_u);
                uint64_t lmask = adj_L[u * W + uw];
                int v_bit = __builtin_ctzll(lmask);
                int v = (uw << 6) + v_bit;

                uint64_t new_lmask = lmask & (lmask - 1ULL);
                adj_L[u * W + uw] = new_lmask;
                if (new_lmask == 0ULL) {
                    summary_L[u] = sum_u & (sum_u - 1ULL);
                }

                int u_word = u >> 6;
                uint64_t new_rmask_u = adj_R[v * W + u_word] & ~(1ULL << (u & 63));
                adj_R[v * W + u_word] = new_rmask_u;
                if (new_rmask_u == 0ULL) {
                    summary_R[v] &= ~(1ULL << u_word);
                }
                p0[u] = v; // Direct write to Leaf 0 (Red permutation)

                uint64_t sum_v = summary_R[v];
                int vw = __builtin_ctzll(sum_v);
                uint64_t rmask = adj_R[v * W + vw];
                int u_bit = __builtin_ctzll(rmask);
                int next_u = (vw << 6) + u_bit;

                uint64_t new_rmask = rmask & (rmask - 1ULL);
                adj_R[v * W + vw] = new_rmask;
                if (new_rmask == 0ULL) {
                    summary_R[v] = sum_v & (sum_v - 1ULL);
                }

                int v_word = v >> 6;
                uint64_t new_lmask_next = adj_L[next_u * W + v_word] & ~(1ULL << (v & 63));
                adj_L[next_u * W + v_word] = new_lmask_next;
                if (new_lmask_next == 0ULL) {
                    summary_L[next_u] &= ~(1ULL << v_word);
                }
                p1[next_u] = v; // Direct write to Leaf 1 (Blue permutation)

                u = next_u;
                if (u == root) break;
            }
        }
    }
}

} // namespace

void decompose_recursive_seq_ws(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int depth,
    int leaf_offset,
    EulerWorkspace& ws,
    std::vector<ScheduleLeaf>& out_leaves
) {
    if (degree == 2) {
        extract_degree2_leaves(
            edges, n, ws, out_leaves[leaf_offset], out_leaves[leaf_offset + 1]
        );
        return;
    }
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
    const Matrix& regularized_matrix,
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
    const Matrix& regularized_matrix,
    int n,
    int degree,
    const Config& cfg,
    PipelineStats* stats
) {
    if (cfg.mode == ExecMode::SEQ) {
        return decompose_seq(regularized_matrix, n, degree);
    }
    if (cfg.mode == ExecMode::OMP_NAIVE) {
        return decompose_omp_naive(regularized_matrix, n, degree, cfg.num_threads);
    }
    return decompose_omp_arch(regularized_matrix, n, degree, cfg, stats);
}

} // namespace bvn
