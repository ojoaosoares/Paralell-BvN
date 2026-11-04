#include "bvn/traffic.hpp"
#include <algorithm>
#include <omp.h>
#include <random>
#include <vector>

namespace bvn {

Matrix make_first_touch_matrix(int n, int num_threads) {
    Matrix mat(static_cast<size_t>(n) * n);
    const int chunk = matrix_row_chunk(n, num_threads);
    #pragma omp parallel for num_threads(num_threads) schedule(static, chunk)
    for (int u = 0; u < n; ++u) {
        int* row = &mat[static_cast<size_t>(u) * n];
        for (int v = 0; v < n; ++v) {
            row[v] = 0;
        }
    }
    return mat;
}

Matrix first_touch_copy(const Matrix& src, int n, int num_threads) {
    Matrix dst(static_cast<size_t>(n) * n);
    const int chunk = matrix_row_chunk(n, num_threads);
    #pragma omp parallel for num_threads(num_threads) schedule(static, chunk)
    for (int u = 0; u < n; ++u) {
        const int* s = &src[static_cast<size_t>(u) * n];
        int* d = &dst[static_cast<size_t>(u) * n];
        for (int v = 0; v < n; ++v) {
            d[v] = s[v];
        }
    }
    return dst;
}

Matrix generate_traffic_matrix(const Config& cfg, int64_t& out_total_packets) {
    const int n = cfg.n;
    const int D = cfg.target_degree;
    // RNG generation is sequential, but pages are first-touched in parallel beforehand so each
    // row slab lives on the NUMA node of the thread that later processes it.
    Matrix mat = make_first_touch_matrix(n, cfg.num_threads);
    std::vector<int> row_rem(n, D);
    std::vector<int> col_rem(n, D);

    std::mt19937_64 rng(cfg.seed);

    // Target utilization ~82% of D so Regularization and Post-Processing
    // actively insert and filter non-trivial amounts of phantom traffic.
    const int target_load = std::max(1, (D * 82) / 100);

    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;

    out_total_packets = 0;

    if (cfg.pattern == TrafficPattern::UNIFORM) {
        for (int s = 0; s < target_load; ++s) {
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int u = 0; u < n; ++u) {
                if ((rng() % 10) != 0) {
                    int v = perm[u];
                    mat[static_cast<size_t>(u) * n + v] += 1;
                    row_rem[u]--;
                    col_rem[v]--;
                    out_total_packets++;
                }
            }
        }
    } else if (cfg.pattern == TrafficPattern::SKEWED) {
        for (int u = 0; u < n; ++u) {
            for (int k = 0; k < 4; ++k) {
                int v = (u + k) % n;
                int w = std::min({row_rem[u], col_rem[v], target_load / (k + 2)});
                if (w > 0) {
                    mat[static_cast<size_t>(u) * n + v] += w;
                    row_rem[u] -= w;
                    col_rem[v] -= w;
                    out_total_packets += w;
                }
            }
        }
    } else {
        int bursts = n * 8;
        std::uniform_int_distribution<int> dist_port(0, n - 1);
        for (int b = 0; b < bursts; ++b) {
            int u = dist_port(rng);
            int v = dist_port(rng);
            int max_add = std::min(row_rem[u], col_rem[v]);
            if (max_add > 0) {
                int add = std::max(1, max_add / 3);
                mat[static_cast<size_t>(u) * n + v] += add;
                row_rem[u] -= add;
                col_rem[v] -= add;
                out_total_packets += add;
            }
        }
    }

    return mat;
}

} // namespace bvn
