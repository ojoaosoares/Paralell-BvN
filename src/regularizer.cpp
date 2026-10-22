#include "bvn/regularizer.hpp"
#include <algorithm>
#include <omp.h>
#include <vector>

namespace bvn {

namespace {
int next_power_of_two(int x) {
    int p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}
} // namespace

std::vector<int> regularize_matrix(
    const std::vector<int>& demand_matrix,
    int n,
    int target_degree,
    ExecMode mode,
    int& out_regularized_degree,
    int64_t& out_phantom_added
) {
    std::vector<int> reg_mat = demand_matrix;
    std::vector<int> row_sum(n, 0);
    std::vector<int> col_sum(n, 0);

    int max_line_sum = target_degree;

    if (mode == ExecMode::SEQ || (mode == ExecMode::OMP_ARCH && n <= 1024)) {
        int* col_ptr = col_sum.data();
        for (int u = 0; u < n; ++u) {
            int rsum = 0;
            const int* row_ptr = &reg_mat[static_cast<size_t>(u) * n];
            #pragma omp simd reduction(+:rsum)
            for (int v = 0; v < n; ++v) {
                int w = row_ptr[v];
                rsum += w;
                col_ptr[v] += w;
            }
            row_sum[u] = rsum;
            if (rsum > max_line_sum) max_line_sum = rsum;
        }
        for (int v = 0; v < n; ++v) {
            if (col_sum[v] > max_line_sum) max_line_sum = col_sum[v];
        }
    } else {
        // Multi-threaded row-major SIMD accumulation (eliminates strided column cache thrashing)
        #pragma omp parallel reduction(max:max_line_sum)
        {
            std::vector<int> local_col(n, 0);

            #pragma omp for schedule(static) nowait
            for (int u = 0; u < n; ++u) {
                int rsum = 0;
                const int* row_ptr = &reg_mat[static_cast<size_t>(u) * n];
                int* lcol_ptr = local_col.data();
                #pragma omp simd reduction(+:rsum)
                for (int v = 0; v < n; ++v) {
                    int w = row_ptr[v];
                    rsum += w;
                    lcol_ptr[v] += w;
                }
                row_sum[u] = rsum;
                if (rsum > max_line_sum) max_line_sum = rsum;
            }

            #pragma omp critical
            {
                for (int v = 0; v < n; ++v) {
                    col_sum[v] += local_col[v];
                }
            }
        }

        for (int v = 0; v < n; ++v) {
            if (col_sum[v] > max_line_sum) max_line_sum = col_sum[v];
        }
    }

    const int D = next_power_of_two(max_line_sum);
    out_regularized_degree = D;

    // Two-pointer O(N) greedy deficit matching to add phantom traffic
    int64_t phantom_added = 0;
    int u = 0;
    int v = 0;
    while (u < n && v < n) {
        int r_def = D - row_sum[u];
        int c_def = D - col_sum[v];
        if (r_def == 0) {
            ++u;
            continue;
        }
        if (c_def == 0) {
            ++v;
            continue;
        }
        int add = std::min(r_def, c_def);
        reg_mat[static_cast<size_t>(u) * n + v] += add;
        row_sum[u] += add;
        col_sum[v] += add;
        phantom_added += add;
    }

    out_phantom_added = phantom_added;
    return reg_mat;
}

} // namespace bvn
