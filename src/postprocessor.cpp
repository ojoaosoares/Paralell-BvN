#include "bvn/postprocessor.hpp"
#include <algorithm>
#include <omp.h>
#include <vector>

namespace bvn {

void cleanup_phantom_traffic(
    const Matrix& original_demand_matrix,
    int n,
    ExecMode mode,
    std::vector<ScheduleLeaf>& schedules
) {
    const int num_sched = static_cast<int>(schedules.size());

    if (mode == ExecMode::SEQ) {
        std::vector<int> rem_row(n, 0);
        for (int u = 0; u < n; ++u) {
            const int* orig_row = &original_demand_matrix[static_cast<size_t>(u) * n];
            for (int v = 0; v < n; ++v) {
                rem_row[v] = orig_row[v];
            }
            for (int s = 0; s < num_sched; ++s) {
                int v = schedules[s].perm[u];
                if (v >= 0) {
                    if (rem_row[v] > 0) {
                        rem_row[v]--;
                    } else {
                        schedules[s].perm[u] = -1; // Filter phantom traffic -> no-op
                    }
                }
            }
        }
    } else {
        // Each input port u is independent: embarrassingly parallel across rows with L1-local state.
        // Same row slabs as the first-touch layout, so every thread reads node-local demand rows.
        const int chunk = matrix_row_chunk(n, omp_get_max_threads());
        #pragma omp parallel
        {
            std::vector<int> rem_row(n, 0);
            #pragma omp for schedule(static, chunk)
            for (int u = 0; u < n; ++u) {
                const int* orig_row = &original_demand_matrix[static_cast<size_t>(u) * n];
                for (int v = 0; v < n; ++v) {
                    rem_row[v] = orig_row[v];
                }
                for (int s = 0; s < num_sched; ++s) {
                    int v = schedules[s].perm[u];
                    if (v >= 0) {
                        if (rem_row[v] > 0) {
                            rem_row[v]--;
                        } else {
                            schedules[s].perm[u] = -1; // Filter phantom traffic -> no-op
                        }
                    }
                }
            }
        }
    }
}

bool verify_decomposition(
    const Matrix& original_demand_matrix,
    int n,
    const std::vector<ScheduleLeaf>& schedules
) {
    std::vector<int> reconstructed(static_cast<size_t>(n) * n, 0);
    std::vector<int> output_used(n, 0);

    for (size_t s = 0; s < schedules.size(); ++s) {
        const auto& leaf = schedules[s];
        if (static_cast<int>(leaf.perm.size()) != n) {
            return false;
        }
        std::fill(output_used.begin(), output_used.end(), 0);

        for (int u = 0; u < n; ++u) {
            int v = leaf.perm[u];
            if (v == -1) continue;
            if (v < 0 || v >= n) {
                return false;
            }
            if (output_used[v] != 0) {
                return false;
            }
            output_used[v] = 1;
            reconstructed[static_cast<size_t>(u) * n + v] += leaf.weight;
        }
    }

    for (size_t i = 0; i < reconstructed.size(); ++i) {
        if (reconstructed[i] != original_demand_matrix[i]) {
            return false;
        }
    }

    return true;
}

} // namespace bvn
