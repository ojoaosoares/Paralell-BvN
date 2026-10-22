#include "bvn/pipeline.hpp"
#include <omp.h>
#include <vector>

namespace bvn {

void phase4_cleanup_phantom(
    const std::vector<int>& original_demand_matrix,
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
        // Each input port u is independent: embarrassingly parallel across rows with L1-local state
        #pragma omp parallel
        {
            std::vector<int> rem_row(n, 0);
            #pragma omp for schedule(static)
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
    const std::vector<int>& original_demand_matrix,
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
            if (v == -1) continue; // Idle port (no-op)
            if (v < 0 || v >= n) {
                return false;
            }
            // Verify zero output port contention in this permutation matrix
            if (output_used[v] != 0) {
                return false;
            }
            output_used[v] = 1;
            reconstructed[static_cast<size_t>(u) * n + v] += leaf.weight;
        }
    }

    // Verify exact match with original chaotic traffic matrix M
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        if (reconstructed[i] != original_demand_matrix[i]) {
            return false;
        }
    }

    return true;
}

} // namespace bvn
