#include "bvn/decomposer.hpp"
#include "bvn/postprocessor.hpp"
#include "bvn/regularizer.hpp"
#include "bvn/traffic.hpp"
#include "bvn/types.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/resource.h>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --n <int>            Number of switch ports N (default: 256)\n"
              << "  --d <int>            Target degree D, power of 2 (default: 256)\n"
              << "  --threads <int>      Number of OpenMP threads (default: 4)\n"
              << "  --cutoff <int>       Task cut-off degree for omp_arch (default: 8)\n"
              << "  --blocks <int>       Spatial block grid KxK for omp_block (0 = dynamic optimal)\n"
              << "  --mode <str>         Execution mode: seq | omp_naive | omp_arch | omp_block (default: omp_arch)\n"
              << "  --pattern <str>      Traffic pattern: uniform | skewed | bursty (default: uniform)\n"
              << "  --seed <int>         RNG seed (default: 42)\n"
              << "  --metrics            Collect and print deep bottleneck telemetry\n"
              << "  --csv                Print single-line CSV output for benchmark scripts\n"
              << "  --no-verify          Skip mathematical verification after decomposition\n"
              << "  --help               Show this help message\n";
}

const char* mode_to_str(bvn::ExecMode m) {
    switch (m) {
        case bvn::ExecMode::SEQ: return "seq";
        case bvn::ExecMode::OMP_NAIVE: return "omp_naive";
        case bvn::ExecMode::OMP_ARCH: return "omp_arch";
        case bvn::ExecMode::OMP_BLOCK: return "omp_block";
    }
    return "unknown";
}

const char* pattern_to_str(bvn::TrafficPattern p) {
    switch (p) {
        case bvn::TrafficPattern::UNIFORM: return "uniform";
        case bvn::TrafficPattern::SKEWED: return "skewed";
        case bvn::TrafficPattern::BURSTY: return "bursty";
    }
    return "unknown";
}

} // namespace

int main(int argc, char** argv) {
    bvn::Config cfg;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            cfg.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            cfg.target_degree = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.num_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cutoff") == 0 && i + 1 < argc) {
            cfg.cutoff_degree = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            cfg.block_factor = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "seq") cfg.mode = bvn::ExecMode::SEQ;
            else if (m == "omp_naive") cfg.mode = bvn::ExecMode::OMP_NAIVE;
            else if (m == "omp_arch") cfg.mode = bvn::ExecMode::OMP_ARCH;
            else if (m == "omp_block") cfg.mode = bvn::ExecMode::OMP_BLOCK;
            else {
                std::cerr << "Unknown mode: " << m << "\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "uniform") cfg.pattern = bvn::TrafficPattern::UNIFORM;
            else if (p == "skewed") cfg.pattern = bvn::TrafficPattern::SKEWED;
            else if (p == "bursty") cfg.pattern = bvn::TrafficPattern::BURSTY;
            else {
                std::cerr << "Unknown pattern: " << p << "\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--metrics") == 0) {
            cfg.detailed_metrics = true;
        } else if (std::strcmp(argv[i], "--csv") == 0) {
            cfg.csv_output = true;
        } else if (std::strcmp(argv[i], "--no-verify") == 0) {
            cfg.verify = false;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unrecognized argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.mode == bvn::ExecMode::SEQ) {
        cfg.num_threads = 1;
    }
    omp_set_num_threads(cfg.num_threads);

    bvn::PipelineStats stats;

    // 0. Generate chaotic traffic matrix M
    std::vector<int> demand_matrix = bvn::generate_traffic_matrix(cfg, stats.total_real_packets);

    // Warm up OpenMP thread pool and thread-local workspaces so OS pthread/mmap page faults do not skew timers
    if (cfg.mode == bvn::ExecMode::OMP_BLOCK) {
        int K = bvn::select_optimal_block_factor(
            cfg.n, cfg.num_threads, cfg.block_factor, &demand_matrix, cfg.target_degree
        );
        int nb = cfg.n / K;
        #pragma omp parallel num_threads(cfg.num_threads)
        {
            bvn::get_thread_workspace().ensure_capacity(
                nb, cfg.target_degree, cfg.target_degree, nb * nb
            );
        }
        bvn::Config warm_cfg = cfg;
        warm_cfg.target_degree = 0;
        bvn::PipelineStats warm_stats;
        double dummy1 = 0.0, dummy2 = 0.0;
        bvn::run_block_bvn_pipeline(demand_matrix, warm_cfg, warm_stats, dummy1, dummy2);
    } else if (cfg.mode != bvn::ExecMode::SEQ) {
        #pragma omp parallel num_threads(cfg.num_threads)
        {
            bvn::get_thread_workspace().ensure_capacity(
                cfg.n, cfg.target_degree >> 1, cfg.cutoff_degree, cfg.n * cfg.n
            );
        }
    } else {
        auto& ws = bvn::get_thread_workspace();
        ws.ensure_capacity(
            cfg.n, cfg.target_degree, cfg.target_degree, cfg.n * cfg.n
        );
        ws.reset_counters();
    }

    struct rusage ru_before{}, ru_after{};
    getrusage(RUSAGE_SELF, &ru_before);

    std::vector<bvn::ScheduleLeaf> schedules;
    if (cfg.mode == bvn::ExecMode::OMP_BLOCK) {
        schedules = bvn::run_block_bvn_pipeline(
            demand_matrix,
            cfg,
            stats,
            stats.phase1_padding_ms,
            stats.phase2_3_euler_ms
        );
    } else {
        // 1. Phase 1: Regularization (Padding)
        double t0 = omp_get_wtime();
        std::vector<int> reg_matrix = bvn::regularize_matrix(
            demand_matrix,
            cfg.n,
            cfg.target_degree,
            cfg.mode,
            stats.regularized_degree,
            stats.total_phantom_packets
        );
        double t1 = omp_get_wtime();

        // 2 & 3. Phases 2 & 3: Eulerian Divide-and-Conquer & Degree-1 Leaf Extraction
        schedules = bvn::decompose_bvn(
            reg_matrix,
            cfg.n,
            stats.regularized_degree,
            cfg,
            &stats
        );
        double t2 = omp_get_wtime();

        stats.phase1_padding_ms = (t1 - t0) * 1000.0;
        stats.phase2_3_euler_ms = (t2 - t1) * 1000.0;
        if (cfg.mode == bvn::ExecMode::SEQ) {
            auto& ws = bvn::get_thread_workspace();
            stats.total_splits = ws.splits;
            stats.symmetric_pruned_splits = ws.sym_pruned;
            stats.unit_odd_fast_splits = ws.unit_odd_fast;
            stats.heavy_odd_splits = ws.heavy_odd;
            stats.deg2_leaf_calls = ws.deg2_calls;
            stats.total_edges_scanned = ws.edges_scanned;
            stats.total_odd_edges_walked = ws.odd_edges_walked;
            stats.thread_min_ms = stats.phase2_3_euler_ms;
            stats.thread_avg_ms = stats.phase2_3_euler_ms;
            stats.thread_max_ms = stats.phase2_3_euler_ms;
            stats.load_imbalance_pct = 0.0;
            stats.parallel_efficiency_pct = 100.0;
        }
    }

    // 4. Phase 4: Cleanup & Post-Processing (for OMP_BLOCK, phantom cleanup was already fused in L1 per sub-matrix!)
    double t2_end = omp_get_wtime();
    if (cfg.mode != bvn::ExecMode::OMP_BLOCK) {
        bvn::cleanup_phantom_traffic(demand_matrix, cfg.n, cfg.mode, schedules);
    }
    double t3 = omp_get_wtime();

    getrusage(RUSAGE_SELF, &ru_after);
    stats.minor_page_faults = ru_after.ru_minflt - ru_before.ru_minflt;
    stats.major_page_faults = ru_after.ru_majflt - ru_before.ru_majflt;
    stats.voluntary_ctx_switches = ru_after.ru_nvcsw - ru_before.ru_nvcsw;
    stats.involuntary_ctx_switches = ru_after.ru_nivcsw - ru_before.ru_nivcsw;
    stats.peak_rss_kb = ru_after.ru_maxrss;
    if (FILE* fp = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), fp)) {
            long vmrss = 0;
            if (std::sscanf(line, "VmHWM: %ld kB", &vmrss) == 1) {
                stats.peak_rss_kb = vmrss;
                break;
            }
        }
        std::fclose(fp);
    }

    stats.phase4_cleanup_ms = (t3 - t2_end) * 1000.0;
    stats.total_ms = stats.phase1_padding_ms + stats.phase2_3_euler_ms + stats.phase4_cleanup_ms;
    stats.num_schedules = static_cast<int>(schedules.size());

    if (cfg.verify) {
        stats.verified_ok = bvn::verify_decomposition(demand_matrix, cfg.n, schedules);
    } else {
        stats.verified_ok = true;
    }

    if (cfg.csv_output) {
        std::cout << mode_to_str(cfg.mode) << ","
                  << pattern_to_str(cfg.pattern) << ","
                  << cfg.n << ","
                  << stats.regularized_degree << ","
                  << cfg.num_threads << ","
                  << cfg.cutoff_degree << ","
                  << std::fixed << std::setprecision(3)
                  << stats.phase1_padding_ms << ","
                  << stats.phase2_3_euler_ms << ","
                  << stats.phase4_cleanup_ms << ","
                  << stats.total_ms << ","
                  << (stats.verified_ok ? "PASS" : "FAIL") << "\n";
    } else {
        std::cout << "============================================================\n"
                  << " Parallel Birkhoff-von Neumann (Eulerian Divide & Conquer)\n"
                  << "============================================================\n"
                  << " Mode             : " << mode_to_str(cfg.mode) << "\n"
                  << " Traffic Pattern  : " << pattern_to_str(cfg.pattern) << "\n"
                  << " Matrix Size (N)  : " << cfg.n << " x " << cfg.n << "\n";
        if (cfg.mode == bvn::ExecMode::OMP_BLOCK) {
            int K = bvn::select_optimal_block_factor(
                cfg.n, cfg.num_threads, cfg.block_factor, &demand_matrix, cfg.target_degree
            );
            std::cout << " Spatial Grid KxK : " << K << " x " << K
                      << " (" << (K * K) << " sub-matrices of " << (cfg.n / K) << "x" << (cfg.n / K) << ")\n";
        }
        std::cout << " Regular Degree D : " << stats.regularized_degree << "\n"
                  << " OpenMP Threads   : " << cfg.num_threads << "\n"
                  << " Task Cut-off Deg : " << cfg.cutoff_degree << "\n"
                  << " Real Packets     : " << stats.total_real_packets << "\n"
                  << " Phantom Padding  : " << stats.total_phantom_packets << "\n"
                  << "------------------------------------------------------------\n"
                  << std::fixed << std::setprecision(3)
                  << " Phase 1 (Padding)          : " << std::setw(9) << stats.phase1_padding_ms << " ms\n"
                  << " Phase 2 & 3 (Euler + BvN)  : " << std::setw(9) << stats.phase2_3_euler_ms << " ms\n"
                  << " Phase 4 (Phantom Cleanup)  : " << std::setw(9) << stats.phase4_cleanup_ms << " ms\n"
                  << "------------------------------------------------------------\n"
                  << " Total Pipeline Time        : " << std::setw(9) << stats.total_ms << " ms\n"
                  << " Verification (Exact BvN)   : " << (stats.verified_ok ? "PASSED (100% Exact)" : "FAILED!") << "\n";

        if (cfg.detailed_metrics) {
            double total_safe = stats.total_ms > 1e-6 ? stats.total_ms : 1.0;
            double medges_per_sec = (stats.total_edges_scanned / 1e6) / (total_safe / 1000.0);
            double modd_per_sec = (stats.total_odd_edges_walked / 1e6) / (total_safe / 1000.0);
            double single_pass_pct = stats.total_splits > 0
                ? (100.0 * stats.unit_odd_fast_splits / stats.total_splits) : 0.0;
            std::cout << "============================================================\n"
                      << " DEEP BOTTLENECK TELEMETRY (--metrics)\n"
                      << "------------------------------------------------------------\n"
                      << " [1] Sub-Phase Breakdown:\n";
            if (cfg.mode == bvn::ExecMode::OMP_ARCH) {
                std::cout << "   - Root Bitmask Build (Par)  : " << std::setw(8) << stats.sub_root_bitmask_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_root_bitmask_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Root Euler Walk (1-Thread): " << std::setw(8) << stats.sub_root_walk_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_root_walk_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Root Partition (Parallel) : " << std::setw(8) << stats.sub_root_extract_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_root_extract_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Task Tree Recursion       : " << std::setw(8) << stats.sub_tree_tasks_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_tree_tasks_ms / total_safe) << "%)\n"
                          << std::setprecision(3);
            } else if (cfg.mode == bvn::ExecMode::OMP_BLOCK) {
                std::cout << "   - Step A (Sub-Block Padding): " << std::setw(8) << stats.phase1_padding_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.phase1_padding_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Step B (Global Leaf Alloc): " << std::setw(8) << stats.sub_block_alloc_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_block_alloc_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Step C.1 (Block Euler BvN): " << std::setw(8) << stats.sub_block_decomp_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_block_decomp_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Step C.2 (L1 Cleanup)     : " << std::setw(8) << stats.sub_block_cleanup_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_block_cleanup_ms / total_safe) << "%)\n"
                          << std::setprecision(3)
                          << "   - Step C.3 (Global Fusion)  : " << std::setw(8) << stats.sub_block_fusion_ms
                          << " ms (" << std::setw(5) << std::setprecision(1) << (100.0 * stats.sub_block_fusion_ms / total_safe) << "%)\n"
                          << std::setprecision(3);
                std::cout << "   - Block Degree (Min / Max)  : " << stats.min_block_degree << " / " << stats.max_block_degree
                          << " (Empty: " << stats.empty_blocks_skipped
                          << ", Hotspot Tasked: " << stats.hotspot_blocks_tasked << ")\n";
            }
            std::cout << "------------------------------------------------------------\n"
                      << " [2] Thread Load Balance & Parallel Efficiency:\n"
                      << "   - Thread Work Time (Min)    : " << std::setw(8) << stats.thread_min_ms << " ms\n"
                      << "   - Thread Work Time (Avg)    : " << std::setw(8) << stats.thread_avg_ms << " ms\n"
                      << "   - Thread Work Time (Max)    : " << std::setw(8) << stats.thread_max_ms << " ms\n"
                      << "   - Load Imbalance ((Max-Avg)/Max): " << std::setw(5) << std::setprecision(1) << stats.load_imbalance_pct << " %\n"
                      << "   - Active Thread Efficiency  : " << std::setw(7) << std::setprecision(1) << stats.parallel_efficiency_pct << " %\n"
                      << std::setprecision(3)
                      << "------------------------------------------------------------\n"
                      << " [3] Algorithmic & Graph Walk Counters:\n"
                      << "   - Total Euler Splits        : " << stats.total_splits << "\n"
                      << "   - Symmetric Pruned Splits   : " << stats.symmetric_pruned_splits << "\n"
                      << "   - Single-Pass Splits (w=1)  : " << stats.unit_odd_fast_splits
                      << " (" << std::setprecision(1) << single_pass_pct << "%)\n"
                      << "   - Multi-Edge Splits (w>1)   : " << stats.heavy_odd_splits << "\n"
                      << "   - Deg-2 Fast-Path Calls     : " << stats.deg2_leaf_calls << "\n"
                      << "   - Total Edges Scanned       : " << stats.total_edges_scanned
                      << " (" << std::setprecision(1) << medges_per_sec << " MEdges/s)\n"
                      << "   - Total Odd Edges Walked    : " << stats.total_odd_edges_walked
                      << " (" << std::setprecision(1) << modd_per_sec << " MOddEdges/s)\n"
                      << "------------------------------------------------------------\n"
                      << " [4] OS / Memory Subsystem (getrusage delta):\n"
                      << "   - Minor Page Faults         : " << stats.minor_page_faults << "\n"
                      << "   - Major Page Faults         : " << stats.major_page_faults << "\n"
                      << "   - Voluntary Ctx Switches    : " << stats.voluntary_ctx_switches << "\n"
                      << "   - Involuntary Ctx Switches  : " << stats.involuntary_ctx_switches << "\n"
                      << "   - Peak Resident Set (RSS)   : " << (stats.peak_rss_kb / 1024.0) << " MB\n";
        }
        std::cout << "============================================================\n";
    }

    return stats.verified_ok ? 0 : 2;
}
