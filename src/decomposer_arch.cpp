#include "bvn/decomposer.hpp"
#include "bvn/regularizer.hpp"
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
    std::vector<WeightedEdge>& out_blue,
    PipelineStats* stats
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

    double t_start = omp_get_wtime();
    double t_after_bitmask = t_start;
    double t_after_walk = t_start;

    #pragma omp parallel num_threads(num_threads) reduction(+:total_odd_edges)
    {
        // Step 1: Parallel Bitmask Construction
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
            t_after_bitmask = omp_get_wtime();
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
            t_after_walk = omp_get_wtime();
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

    double t_end = omp_get_wtime();
    if (stats) {
        stats->sub_root_bitmask_ms = (t_after_bitmask - t_start) * 1000.0;
        stats->sub_root_walk_ms = (t_after_walk - t_after_bitmask) * 1000.0;
        stats->sub_root_extract_ms = (t_end - t_after_walk) * 1000.0;
        stats->total_splits += 1;
        if (total_odd_edges == 0) stats->symmetric_pruned_splits += 1;
        stats->total_odd_edges_walked += total_odd_edges;
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
        double t0 = omp_get_wtime();
        decompose_recursive_seq_ws(edges, n, degree, 0, leaf_offset, tls_ws, out_leaves);
        tls_ws.active_time_ms += (omp_get_wtime() - t0) * 1000.0;
        return;
    }

    double t0 = omp_get_wtime();
    int left_id = 2 * node_id + 1;
    int right_id = 2 * node_id + 2;
    std::vector<WeightedEdge>& red_edges = node_buffers[left_id];
    std::vector<WeightedEdge>& blue_edges = node_buffers[right_id];
    if (red_edges.capacity() < edges.size()) red_edges.reserve(edges.size());
    if (blue_edges.capacity() < edges.size()) blue_edges.reserve(edges.size());

    bool symmetric = euler_split_sequential_ws(edges, n, tls_ws, red_edges, blue_edges);
    tls_ws.active_time_ms += (omp_get_wtime() - t0) * 1000.0;
    int half_deg = degree >> 1;

    if (symmetric) {
        decompose_recursive_omp_arch(
            red_edges, n, half_deg, depth + 1, cutoff_degree, leaf_offset, left_id, node_buffers, out_leaves
        );
        double tc = omp_get_wtime();
        for (int i = 0; i < half_deg; ++i) {
            out_leaves[leaf_offset + half_deg + i] = out_leaves[leaf_offset + i];
        }
        tls_ws.active_time_ms += (omp_get_wtime() - tc) * 1000.0;
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

void aggregate_thread_telemetry(int num_threads, double wall_ms, PipelineStats* stats) {
    if (!stats) return;
    std::vector<double> t_times(num_threads, 0.0);
    int64_t tot_splits = 0, tot_sym = 0, tot_unit = 0, tot_heavy = 0, tot_deg2 = 0, tot_scanned = 0, tot_walked = 0;

    #pragma omp parallel num_threads(num_threads) reduction(+:tot_splits,tot_sym,tot_unit,tot_heavy,tot_deg2,tot_scanned,tot_walked)
    {
        int tid = omp_get_thread_num();
        EulerWorkspace& ws = get_thread_workspace();
        t_times[tid] = ws.active_time_ms;
        tot_splits += ws.splits;
        tot_sym += ws.sym_pruned;
        tot_unit += ws.unit_odd_fast;
        tot_heavy += ws.heavy_odd;
        tot_deg2 += ws.deg2_calls;
        tot_scanned += ws.edges_scanned;
        tot_walked += ws.odd_edges_walked;
    }

    double t_min = t_times[0], t_max = t_times[0], t_sum = 0.0;
    for (int t = 0; t < num_threads; ++t) {
        t_min = std::min(t_min, t_times[t]);
        t_max = std::max(t_max, t_times[t]);
        t_sum += t_times[t];
    }
    double t_avg = t_sum / std::max(1, num_threads);

    stats->thread_min_ms = t_min;
    stats->thread_avg_ms = t_avg;
    stats->thread_max_ms = t_max;
    stats->load_imbalance_pct = (t_max > 0.0) ? ((t_max - t_avg) / t_max) * 100.0 : 0.0;
    stats->parallel_efficiency_pct = (wall_ms > 0.0) ? (t_avg / wall_ms) * 100.0 : 0.0;

    stats->total_splits += tot_splits;
    stats->symmetric_pruned_splits += tot_sym;
    stats->unit_odd_fast_splits += tot_unit;
    stats->heavy_odd_splits += tot_heavy;
    stats->deg2_leaf_calls += tot_deg2;
    stats->total_edges_scanned += tot_scanned;
    stats->total_odd_edges_walked += tot_walked;
}

} // namespace

std::vector<ScheduleLeaf> decompose_omp_arch(
    const std::vector<int>& regularized_matrix,
    int n,
    int degree,
    const Config& cfg,
    PipelineStats* stats
) {
    std::vector<ScheduleLeaf> leaves(degree);

    if (degree <= cfg.cutoff_degree || degree == 1) {
        return decompose_seq(regularized_matrix, n, degree);
    }

    int eff_cutoff = std::max(cfg.cutoff_degree, 2);
    int num_internal_leaves = std::max(2, degree / eff_cutoff);
    int total_nodes = 2 * num_internal_leaves + 1;
    static std::vector<std::vector<WeightedEdge>> node_buffers;
    if (static_cast<int>(node_buffers.size()) < total_nodes) {
        node_buffers.resize(total_nodes);
    }

    #pragma omp parallel num_threads(cfg.num_threads)
    {
        get_thread_workspace().reset_counters();
    }

    std::vector<WeightedEdge>& red_edges = node_buffers[1];
    std::vector<WeightedEdge>& blue_edges = node_buffers[2];
    bool root_symmetric = euler_root_split_omp_arch(
        regularized_matrix, n, cfg.num_threads, red_edges, blue_edges, stats
    );
    int half_deg = degree >> 1;

    double t_tree_start = omp_get_wtime();

    #pragma omp parallel num_threads(cfg.num_threads)
    {
        if (root_symmetric) {
            #pragma omp single
            {
                decompose_recursive_omp_arch(
                    red_edges, n, half_deg, 1, cfg.cutoff_degree, 0, 1, node_buffers, leaves
                );
            }
            double tc = omp_get_wtime();
            #pragma omp for schedule(static)
            for (int i = 0; i < half_deg; ++i) {
                leaves[half_deg + i] = leaves[i];
            }
            get_thread_workspace().active_time_ms += (omp_get_wtime() - tc) * 1000.0;
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

    double t_tree_end = omp_get_wtime();
    double tree_wall_ms = (t_tree_end - t_tree_start) * 1000.0;
    if (stats) {
        stats->sub_tree_tasks_ms = tree_wall_ms;
        aggregate_thread_telemetry(cfg.num_threads, tree_wall_ms, stats);
    }

    return leaves;
}

namespace {

int next_pow2_int(int x) {
    int p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

} // namespace

int select_optimal_block_factor(
    int n,
    int num_threads,
    int requested_k,
    const std::vector<int>* demand_matrix,
    int target_degree
) {
    if (requested_k > 0 && (n % requested_k == 0)) {
        return requested_k;
    }

    int k = 1;
    while ((k * k < num_threads || (n / k) > 256) && (n / (k * 2)) >= 64 && (n % (k * 2) == 0)) {
        k <<= 1;
    }
    if (k == 1 && n >= 128 && (n % 2 == 0) && num_threads > 1) {
        k = 2;
    }

    // Adaptive Anti-Inflation Check: sample 32 rows in O(N) (< 0.02 ms) to check if heavy bursts
    // inflate sub-block degrees (> D/K) across more than 2 diagonal shifts (e.g., bursty traffic).
    if (k > 2 && demand_matrix && !demand_matrix->empty() && target_degree > 0) {
        const int nb = n / k;
        const int thresh = target_degree / k;
        uint64_t inflated_shifts_mask = 0ULL;
        const int step = std::max(1, n / 32);
        for (int u = 0; u < n; u += step) {
            const int bi = u / nb;
            const int* __restrict row = &(*demand_matrix)[static_cast<size_t>(u) * n];
            for (int bj = 0; bj < k; ++bj) {
                int s = (bj - bi + k) % k;
                if (inflated_shifts_mask & (1ULL << s)) continue;
                const int* __restrict sub_row = row + bj * nb;
                int rsum = 0;
                for (int v = 0; v < nb; ++v) {
                    rsum += sub_row[v];
                }
                if (rsum > thresh) {
                    inflated_shifts_mask |= (1ULL << s);
                }
            }
        }
        if (__builtin_popcountll(inflated_shifts_mask) > 2) {
            k = std::max(2, n / 1024);
        }
    }

    return k;
}

std::vector<ScheduleLeaf> run_block_bvn_pipeline(
    const std::vector<int>& demand_matrix,
    const Config& cfg,
    PipelineStats& stats,
    double& out_phase1_ms,
    double& out_phase2_3_ms
) {
    const int n = cfg.n;
    const int K = select_optimal_block_factor(
        n, cfg.num_threads, cfg.block_factor, &demand_matrix, std::max(cfg.target_degree, n)
    );
    const int nb = n / K;
    const int num_blocks = K * K;

    double t0 = omp_get_wtime();

    static std::vector<std::vector<WeightedEdge>> block_edges;
    static std::vector<std::vector<ScheduleLeaf>> block_leaves;
    static std::vector<ScheduleLeaf> global_pool;
    if (static_cast<int>(block_edges.size()) < num_blocks) {
        block_edges.resize(num_blocks);
    }
    if (static_cast<int>(block_leaves.size()) < num_blocks) {
        block_leaves.resize(num_blocks);
    }

    // Warm-up call (target_degree == 0) pre-touches thread-local, block, and global schedule buffers
    if (cfg.target_degree <= 0) {
        int max_possible_sched = 2 * next_pow2_int(std::max(1, n));
        if (static_cast<int>(global_pool.size()) < max_possible_sched) {
            global_pool.resize(max_possible_sched);
        }
        #pragma omp parallel num_threads(cfg.num_threads)
        {
            #pragma omp for schedule(static)
            for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {
                block_edges[b_idx].reserve(static_cast<size_t>(nb) * nb);
                block_leaves[b_idx].resize(n / K);
                for (auto& lf : block_leaves[b_idx]) {
                    lf.perm.assign(nb, -1);
                }
            }
            #pragma omp for schedule(static)
            for (int s = 0; s < max_possible_sched; ++s) {
                global_pool[s].weight = 1;
                global_pool[s].perm.assign(n, -1);
            }
        }
        out_phase1_ms = 0.0;
        out_phase2_3_ms = 0.0;
        return {};
    }

    std::vector<int> block_deg(num_blocks, 0);
    std::vector<int> shift_deg(K, 0);
    int64_t total_phantom = 0;

    // Phase 1: Single-pass parallel sub-matrix extraction + Edge-Coalesced & Parity-Aligned local padding
    #pragma omp parallel num_threads(cfg.num_threads) reduction(+:total_phantom)
    {
        get_thread_workspace().reset_counters();
        static thread_local std::vector<int> row_sum;
        static thread_local std::vector<int> col_sum;
        if (static_cast<int>(row_sum.size()) < nb) row_sum.resize(nb);
        if (static_cast<int>(col_sum.size()) < nb) col_sum.resize(nb);

        int* __restrict r_ptr = row_sum.data();
        int* __restrict c_ptr = col_sum.data();

        #pragma omp for schedule(dynamic, 1)
        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {
            const int s = b_idx / K;
            const int bi = b_idx % K;
            const int bj = (bi + s) % K;
            const int row_base = bi * nb;
            const int col_base = bj * nb;

            std::fill(c_ptr, c_ptr + nb, 0);
            int max_line = 0;

            auto& edges = block_edges[b_idx];
            edges.clear();

            for (int u = 0; u < nb; ++u) {
                const int* __restrict src_row = &demand_matrix[static_cast<size_t>(row_base + u) * n + col_base];
                int rsum = 0;
                for (int v = 0; v < nb; ++v) {
                    int w = src_row[v];
                    if (w > 0) {
                        rsum += w;
                        c_ptr[v] += w;
                        edges.push_back({u, v, w});
                    }
                }
                r_ptr[u] = rsum;
                if (rsum > max_line) max_line = rsum;
            }
            for (int v = 0; v < nb; ++v) {
                if (c_ptr[v] > max_line) max_line = c_ptr[v];
            }

            if (max_line == 0) {
                block_deg[b_idx] = 0;
                continue;
            }

            const int db = next_pow2_int(max_line);
            block_deg[b_idx] = db;

            // Count active rows and active columns; only m = max(act_r, act_c) lines need degree db!
            // The remaining (nb - m) empty rows and (nb - m) empty cols stay at degree 0 (zero phantom edges!).
            int act_r = 0;
            int act_c = 0;
            for (int i = 0; i < nb; ++i) {
                if (r_ptr[i] > 0) ++act_r;
                if (c_ptr[i] > 0) ++act_c;
            }
            int need_empty_r = std::max(0, act_c - act_r);
            int need_empty_c = std::max(0, act_r - act_c);
            for (int i = 0; i < nb; ++i) {
                if (r_ptr[i] == 0) {
                    if (need_empty_r > 0) {
                        --need_empty_r; // Keep r_ptr[i] = 0 so deficit is db
                    } else {
                        r_ptr[i] = db;  // Mark as degree-0 pruned (deficit = 0)!
                    }
                }
                if (c_ptr[i] == 0) {
                    if (need_empty_c > 0) {
                        --need_empty_c; // Keep c_ptr[i] = 0 so deficit is db
                    } else {
                        c_ptr[i] = db;  // Mark as degree-0 pruned (deficit = 0)!
                    }
                }
            }

            // Pass 1: Coalesce phantom padding directly onto existing real edges (0 new edges created!)
            for (auto& e : edges) {
                int r_def = db - r_ptr[e.u];
                int c_def = db - c_ptr[e.v];
                if (r_def > 0 && c_def > 0) {
                    int add = std::min(r_def, c_def);
                    e.w += add;
                    r_ptr[e.u] += add;
                    c_ptr[e.v] += add;
                    total_phantom += add;
                }
            }

            // Pass 2: Match remaining deficits among active/activated lines
            int u = 0;
            int v = 0;
            while (u < nb && v < nb) {
                int r_def = db - r_ptr[u];
                int c_def = db - c_ptr[v];
                if (r_def == 0) {
                    ++u;
                    continue;
                }
                if (c_def == 0) {
                    ++v;
                    continue;
                }
                int add = std::min(r_def, c_def);
                edges.push_back({u, v, add});
                r_ptr[u] += add;
                c_ptr[v] += add;
                total_phantom += add;
            }
        }
    }

    int min_bdeg = 1e9, max_bdeg = 0, empty_cnt = 0, hotspot_cnt = 0;
    const int hotspot_thresh = (num_blocks < cfg.num_threads * 2)
        ? 64
        : std::max(cfg.cutoff_degree, std::max(256, cfg.target_degree / K));
    const int intra_block_cutoff = std::max(cfg.cutoff_degree, 32);

    for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {
        int db = block_deg[b_idx];
        min_bdeg = std::min(min_bdeg, db);
        max_bdeg = std::max(max_bdeg, db);
        if (db == 0) ++empty_cnt;
        if (db > hotspot_thresh) ++hotspot_cnt;
    }

    stats.total_phantom_packets = total_phantom;
    stats.empty_blocks_skipped = empty_cnt;
    stats.hotspot_blocks_tasked = hotspot_cnt;
    stats.min_block_degree = (min_bdeg == 1e9) ? 0 : min_bdeg;
    stats.max_block_degree = max_bdeg;

    double t1 = omp_get_wtime();

    static std::vector<std::vector<std::vector<WeightedEdge>>> block_node_buffers;
    if (static_cast<int>(block_node_buffers.size()) < num_blocks) {
        block_node_buffers.resize(num_blocks);
    }
    for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {
        int db = block_deg[b_idx];
        if (static_cast<int>(block_leaves[b_idx].size()) < db) {
            block_leaves[b_idx].resize(db);
        }
        if (db > hotspot_thresh) {
            int num_internal = std::max(2, db / intra_block_cutoff);
            int total_nodes = 2 * num_internal + 1;
            if (static_cast<int>(block_node_buffers[b_idx].size()) < total_nodes) {
                block_node_buffers[b_idx].resize(total_nodes);
            }
        }
    }

    double t_after_alloc = omp_get_wtime();
    double t_after_stepA = t_after_alloc;
    double sum_cleanup_ms = 0.0;
    int total_schedules = 0;
    std::vector<int> shift_offset(K + 1, 0);
    std::vector<int> sched_to_shift;
    std::vector<ScheduleLeaf> global_leaves;

    #pragma omp parallel num_threads(cfg.num_threads) reduction(+:sum_cleanup_ms)
    {
        EulerWorkspace& tls_ws = get_thread_workspace();

        // Step C.1 & C.2: Decompose each non-empty N_b x N_b block in L1 cache + local cleanup & empty-schedule compaction
        #pragma omp for schedule(dynamic, 1)
        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {
            int db = block_deg[b_idx];
            if (db == 0) continue;
            const auto& edges = block_edges[b_idx];
            auto& b_leaves = block_leaves[b_idx];

            if (db > hotspot_thresh) {
                decompose_recursive_omp_arch(
                    edges, nb, db, 0, intra_block_cutoff, 0, 0, block_node_buffers[b_idx], b_leaves
                );
            } else {
                double tb0 = omp_get_wtime();
                tls_ws.ensure_capacity(nb, db, db, static_cast<int>(edges.size()));
                decompose_recursive_seq_ws(edges, nb, db, 0, 0, tls_ws, b_leaves);
                tls_ws.active_time_ms += (omp_get_wtime() - tb0) * 1000.0;
            }

            double tc0 = omp_get_wtime();
            // Local L1-resident phantom traffic cleanup + Dead-Schedule Compaction for block b_idx
            const int s = b_idx / K;
            const int bi = b_idx % K;
            const int bj = (bi + s) % K;
            const int row_base = bi * nb;
            const int col_base = bj * nb;

            static thread_local std::vector<int> rem_row;
            static thread_local std::vector<uint8_t> sched_nonempty;
            if (static_cast<int>(rem_row.size()) < nb) rem_row.resize(nb);
            if (static_cast<int>(sched_nonempty.size()) < db) sched_nonempty.resize(db);
            int* __restrict rem_ptr = rem_row.data();
            uint8_t* __restrict ne_ptr = sched_nonempty.data();
            std::fill(ne_ptr, ne_ptr + db, 0);

            for (int u = 0; u < nb; ++u) {
                const int* __restrict orig_row = &demand_matrix[static_cast<size_t>(row_base + u) * n + col_base];
                int row_real = 0;
                for (int v = 0; v < nb; ++v) {
                    int w = orig_row[v];
                    rem_ptr[v] = w;
                    row_real += w;
                }
                if (row_real == 0) {
                    for (int d = 0; d < db; ++d) {
                        b_leaves[d].perm[u] = -1;
                    }
                    continue;
                }
                for (int d = 0; d < db; ++d) {
                    int v = b_leaves[d].perm[u];
                    if (v >= 0) {
                        if (rem_ptr[v] > 0) {
                            rem_ptr[v]--;
                            ne_ptr[d] = 1;
                            if (--row_real == 0) {
                                for (int d_rem = d + 1; d_rem < db; ++d_rem) {
                                    b_leaves[d_rem].perm[u] = -1;
                                }
                                break;
                            }
                        } else {
                            b_leaves[d].perm[u] = -1; // Filtered locally in L1 cache!
                        }
                    }
                }
            }

            // Compact non-empty schedules in O(db) pointer swaps!
            int write_d = 0;
            for (int d = 0; d < db; ++d) {
                if (ne_ptr[d]) {
                    if (write_d != d) {
                        b_leaves[write_d].perm.swap(b_leaves[d].perm);
                    }
                    ++write_d;
                }
            }
            block_deg[b_idx] = write_d;

            double tc1 = omp_get_wtime();
            double c_ms = (tc1 - tc0) * 1000.0;
            sum_cleanup_ms += c_ms;
            tls_ws.active_time_ms += c_ms;
        }

        #pragma omp single
        {
            t_after_stepA = omp_get_wtime();
            for (int s = 0; s < K; ++s) {
                int d_max = 0;
                for (int bi = 0; bi < K; ++bi) {
                    d_max = std::max(d_max, block_deg[s * K + bi]);
                }
                shift_deg[s] = d_max;
                shift_offset[s + 1] = shift_offset[s] + d_max;
            }
            total_schedules = shift_offset[K];
            stats.regularized_degree = total_schedules;
            global_leaves.resize(total_schedules);
            sched_to_shift.resize(total_schedules);
            for (int s = 0; s < K; ++s) {
                std::fill(
                    sched_to_shift.begin() + shift_offset[s],
                    sched_to_shift.begin() + shift_offset[s + 1],
                    s
                );
            }
        }

        // Step C.3: Flat 12-Thread Parallel Fusion across all total_schedules leaves (100% thread utilization!)
        double tf0 = omp_get_wtime();
        #pragma omp for schedule(static)
        for (int g_idx = 0; g_idx < total_schedules; ++g_idx) {
            const int s = sched_to_shift[g_idx];
            const int d = g_idx - shift_offset[s];
            ScheduleLeaf& g_leaf = global_leaves[g_idx];
            g_leaf.weight = 1;
            if (g_idx < static_cast<int>(global_pool.size()) &&
                static_cast<int>(global_pool[g_idx].perm.size()) == n) {
                g_leaf.perm.swap(global_pool[g_idx].perm);
            } else {
                g_leaf.perm.resize(n);
            }
            int* __restrict g_perm = g_leaf.perm.data();

            for (int bi = 0; bi < K; ++bi) {
                const int b_idx = s * K + bi;
                const int db = block_deg[b_idx];
                const int bj = (bi + s) % K;
                const int row_base = bi * nb;
                const int col_base = bj * nb;

                if (d < db) {
                    const int* __restrict b_perm = block_leaves[b_idx][d].perm.data();
                    for (int u = 0; u < nb; ++u) {
                        int local_v = b_perm[u];
                        g_perm[row_base + u] = (local_v >= 0) ? (col_base + local_v) : -1;
                    }
                } else {
                    std::fill(g_perm + row_base, g_perm + row_base + nb, -1);
                }
            }
        }
        tls_ws.active_time_ms += (omp_get_wtime() - tf0) * 1000.0;
    }

    double t2 = omp_get_wtime();
    out_phase1_ms = (t1 - t0) * 1000.0;
    out_phase2_3_ms = (t2 - t1) * 1000.0;

    stats.sub_block_alloc_ms = (t_after_alloc - t1) * 1000.0;
    stats.sub_block_cleanup_ms = sum_cleanup_ms / std::max(1, cfg.num_threads);
    stats.sub_block_decomp_ms = std::max(0.0, (t_after_stepA - t_after_alloc) * 1000.0 - stats.sub_block_cleanup_ms);
    stats.sub_block_fusion_ms = (t2 - t_after_stepA) * 1000.0;
    aggregate_thread_telemetry(cfg.num_threads, out_phase2_3_ms, &stats);

    return global_leaves;
}

} // namespace bvn
