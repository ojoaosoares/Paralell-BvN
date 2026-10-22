#include "bvn/decomposer.hpp"
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <vector>

namespace bvn {

namespace {

/**
 * Root-level (Depth 0) Architecture-Aware Parallel Split directly from the N x N regularized matrix:
 * 1. Parallel lock-free 64-bit Bitmask construction using 64-row block partitioning (disjoint uint64_t words in adj_R).
 * 2. Hardware TZCNT (__builtin_ctzll) + BLSR 2-step (L -> R -> L) Eulerian cycle walk in L1/L2 cache.
 * 3. Parallel prefix-sum construction of red_edges and blue_edges directly from the matrix (eliminating initial_edges).
 */
bool euler_root_split_omp_arch(
    const std::vector<int>& regularized_matrix,
    int n,
    int num_threads,
    std::vector<WeightedEdge>& out_red,
    std::vector<WeightedEdge>& out_blue
) {
    const int W = (n + 63) >> 6;
    const size_t mask_size = static_cast<size_t>(n) * W;

    EulerWorkspace& root_ws = get_thread_workspace();
    if (root_ws.adj_L.size() < mask_size) root_ws.adj_L.assign(mask_size, 0ULL);
    if (root_ws.adj_R.size() < mask_size) root_ws.adj_R.assign(mask_size, 0ULL);
    if (root_ws.red_mask.size() < mask_size) root_ws.red_mask.assign(mask_size, 0ULL);
    if (static_cast<int>(root_ws.summary_L.size()) < n) root_ws.summary_L.assign(n, 0ULL);
    if (static_cast<int>(root_ws.summary_R.size()) < n) root_ws.summary_R.assign(n, 0ULL);

    uint64_t* __restrict adj_L = root_ws.adj_L.data();
    uint64_t* __restrict adj_R = root_ws.adj_R.data();
    uint64_t* __restrict red_mask = root_ws.red_mask.data();
    uint64_t* __restrict summary_L = root_ws.summary_L.data();
    uint64_t* __restrict summary_R = root_ws.summary_R.data();

    int total_odd_edges = 0;
    std::vector<int> thread_red_count(num_threads, 0);
    std::vector<int> thread_blue_count(num_threads, 0);
    std::vector<int> thread_red_offset(num_threads + 1, 0);
    std::vector<int> thread_blue_offset(num_threads + 1, 0);

    #pragma omp parallel num_threads(num_threads) reduction(+:total_odd_edges)
    {
        // Step 1: Parallel Bitmask Construction
        // Chunk size of 64 rows guarantees that (u >> 6) is unique per thread,
        // so writes to adj_R[v * W + (u >> 6)] are 100% lock-free and race-free without atomics!
        #pragma omp for schedule(static, 64)
        for (int u = 0; u < n; ++u) {
            const int* __restrict row = &regularized_matrix[static_cast<size_t>(u) * n];
            uint64_t* __restrict l_row = &adj_L[u * W];
            const int u_word = u >> 6;
            const uint64_t u_bit = 1ULL << (u & 63);
            uint64_t sum_u = 0ULL;

            for (int v = 0; v < n; ++v) {
                int w = row[v];
                if (w & 1) {
                    int uw = v >> 6;
                    l_row[uw] |= (1ULL << (v & 63));
                    sum_u |= (1ULL << uw);
                    adj_R[v * W + u_word] |= u_bit;
                    ++total_odd_edges;
                }
            }
            summary_L[u] = sum_u;
        }

        // Parallel construction of summary_R across vertices v in R (100% race-free per v)
        if (total_odd_edges > 0) {
            #pragma omp for schedule(static)
            for (int v = 0; v < n; ++v) {
                const uint64_t* __restrict r_row = &adj_R[v * W];
                uint64_t sum_v = 0ULL;
                for (int w = 0; w < W; ++w) {
                    if (r_row[w] != 0ULL) {
                        sum_v |= (1ULL << w);
                    }
                }
                summary_R[v] = sum_v;
            }
        }

        // Step 2: Ultra-fast L1-resident 2-Level Hierarchical Bitmask + double TZCNT/BLSR Eulerian walk
        #pragma omp single
        {
            if (total_odd_edges > 0) {
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
                            red_mask[u * W + uw] |= (1ULL << v_bit);

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

                            u = next_u;
                            if (u == root) break;
                        }
                    }
                }
            }
        }

        // Step 3: Parallel extraction of out_red and out_blue directly from regularized_matrix + red_mask
        int tid = omp_get_thread_num();
        auto& local_red = get_thread_workspace().level_red[0];
        auto& local_blue = get_thread_workspace().level_blue[0];
        local_red.clear();
        local_blue.clear();

        if (total_odd_edges == 0) {
            #pragma omp for schedule(static)
            for (int u = 0; u < n; ++u) {
                const int* __restrict row = &regularized_matrix[static_cast<size_t>(u) * n];
                for (int v = 0; v < n; ++v) {
                    int half = row[v] >> 1;
                    if (half > 0) {
                        local_red.push_back({u, v, half});
                    }
                }
            }
        } else {
            #pragma omp for schedule(static)
            for (int u = 0; u < n; ++u) {
                const int* __restrict row = &regularized_matrix[static_cast<size_t>(u) * n];
                uint64_t* __restrict r_row = &red_mask[u * W];
                for (int v = 0; v < n; ++v) {
                    int w = row[v];
                    if (w > 0) {
                        int half = w >> 1;
                        int w_red = half;
                        int w_blue = half;
                        if (w & 1) {
                            uint64_t& rword = r_row[v >> 6];
                            uint64_t bit = 1ULL << (v & 63);
                            if (rword & bit) {
                                ++w_red;
                                rword &= ~bit;
                            } else {
                                ++w_blue;
                            }
                        }
                        if (w_red > 0) local_red.push_back({u, v, w_red});
                        if (w_blue > 0) local_blue.push_back({u, v, w_blue});
                    }
                }
            }
        }

        thread_red_count[tid] = static_cast<int>(local_red.size());
        thread_blue_count[tid] = static_cast<int>(local_blue.size());

        #pragma omp barrier

        #pragma omp single
        {
            for (int t = 0; t < num_threads; ++t) {
                thread_red_offset[t + 1] = thread_red_offset[t] + thread_red_count[t];
                thread_blue_offset[t + 1] = thread_blue_offset[t] + thread_blue_count[t];
            }
            out_red.resize(thread_red_offset[num_threads]);
            if (total_odd_edges > 0) {
                out_blue.resize(thread_blue_offset[num_threads]);
            }
        }

        std::copy(local_red.begin(), local_red.end(), out_red.begin() + thread_red_offset[tid]);
        if (total_odd_edges > 0) {
            std::copy(local_blue.begin(), local_blue.end(), out_blue.begin() + thread_blue_offset[tid]);
        }
    }

    return (total_odd_edges == 0);
}

void decompose_recursive_omp_arch(
    const std::vector<WeightedEdge>& edges,
    int n,
    int degree,
    int depth,
    int cutoff_degree,
    int leaf_offset,
    int node_id,
    std::vector<std::vector<WeightedEdge>>& node_buffers,
    std::vector<ScheduleLeaf>& out_leaves
) {
    EulerWorkspace& tls_ws = get_thread_workspace();
    tls_ws.ensure_capacity(n, degree, cutoff_degree, static_cast<int>(edges.size()));

    if (degree <= cutoff_degree || degree <= 2) {
        decompose_recursive_seq_ws(edges, n, degree, 0, leaf_offset, tls_ws, out_leaves);
        return;
    }

    int left_id = 2 * node_id + 1;
    int right_id = 2 * node_id + 2;
    std::vector<WeightedEdge>& red_edges = node_buffers[left_id];
    std::vector<WeightedEdge>& blue_edges = node_buffers[right_id];
    if (red_edges.capacity() < edges.size()) red_edges.reserve(edges.size());
    if (blue_edges.capacity() < edges.size()) blue_edges.reserve(edges.size());

    bool symmetric = euler_split_sequential_ws(edges, n, tls_ws, red_edges, blue_edges);
    int half_deg = degree >> 1;

    if (symmetric) {
        decompose_recursive_omp_arch(
            red_edges, n, half_deg, depth + 1, cutoff_degree, leaf_offset, left_id, node_buffers, out_leaves
        );
        for (int i = 0; i < half_deg; ++i) {
            out_leaves[leaf_offset + half_deg + i] = out_leaves[leaf_offset + i];
        }
    } else {
        #pragma omp task shared(out_leaves, node_buffers, red_edges) firstprivate(n, half_deg, depth, cutoff_degree, leaf_offset, left_id)
        {
            decompose_recursive_omp_arch(
                red_edges, n, half_deg, depth + 1, cutoff_degree, leaf_offset, left_id, node_buffers, out_leaves
            );
        }

        // Execute right branch directly on the current thread to halve task creation overhead and preserve L1/L2 locality
        decompose_recursive_omp_arch(
            blue_edges, n, half_deg, depth + 1, cutoff_degree, leaf_offset + half_deg, right_id, node_buffers, out_leaves
        );

        #pragma omp taskwait
    }
}

} // namespace

std::vector<ScheduleLeaf> decompose_omp_arch(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg
) {
    std::vector<ScheduleLeaf> leaves(degree);

    if (degree <= cfg.cutoff_degree || degree == 1) {
        return decompose_seq(regularized_matrix, n, degree);
    }

    // Persistent pool of edge buffers for internal task tree nodes (zero heap contention inside tasks)
    int eff_cutoff = std::max(cfg.cutoff_degree, 2);
    int num_internal_leaves = std::max(2, degree / eff_cutoff);
    int total_nodes = 2 * num_internal_leaves + 1;
    static std::vector<std::vector<WeightedEdge>> node_buffers;
    if (static_cast<int>(node_buffers.size()) < total_nodes) {
        node_buffers.resize(total_nodes);
    }

    std::vector<WeightedEdge>& red_edges = node_buffers[1];
    std::vector<WeightedEdge>& blue_edges = node_buffers[2];
    bool root_symmetric = euler_root_split_omp_arch(
        regularized_matrix, n, cfg.num_threads, red_edges, blue_edges
    );
    int half_deg = degree >> 1;

    // Pre-reserve internal task buffers in parallel across threads (NUMA first-touch & zero malloc in tasks)
    size_t root_child_edges = std::max(red_edges.size(), blue_edges.size());
    #pragma omp parallel num_threads(cfg.num_threads)
    {
        #pragma omp for schedule(static)
        for (int id = 3; id < total_nodes; ++id) {
            if (node_buffers[id].capacity() < root_child_edges) {
                node_buffers[id].reserve(root_child_edges);
            }
        }

        if (root_symmetric) {
            #pragma omp single
            {
                decompose_recursive_omp_arch(
                    red_edges, n, half_deg, 1, cfg.cutoff_degree, 0, 1, node_buffers, leaves
                );
            }
            #pragma omp for schedule(static)
            for (int i = 0; i < half_deg; ++i) {
                leaves[half_deg + i] = leaves[i];
            }
        } else {
            #pragma omp single nowait
            {
                #pragma omp task shared(leaves, node_buffers, red_edges) firstprivate(n, half_deg)
                {
                    decompose_recursive_omp_arch(
                        red_edges, n, half_deg, 1, cfg.cutoff_degree, 0, 1, node_buffers, leaves
                    );
                }
                decompose_recursive_omp_arch(
                    blue_edges, n, half_deg, 1, cfg.cutoff_degree, half_deg, 2, node_buffers, leaves
                );
                #pragma omp taskwait
            }
        }
    }

    return leaves;
}

} // namespace bvn
