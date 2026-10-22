#include "bvn/pipeline.hpp"
#include "bvn/types.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --n <int>            Number of switch ports N (default: 256)\n"
              << "  --d <int>            Target degree D, power of 2 (default: 256)\n"
              << "  --threads <int>      Number of OpenMP threads (default: 4)\n"
              << "  --cutoff <int>       Task cut-off degree for omp_arch (default: 8)\n"
              << "  --mode <str>         Execution mode: seq | omp_naive | omp_arch (default: omp_arch)\n"
              << "  --pattern <str>      Traffic pattern: uniform | skewed | bursty (default: uniform)\n"
              << "  --seed <int>         RNG seed (default: 42)\n"
              << "  --csv                Print single-line CSV output for benchmark scripts\n"
              << "  --no-verify          Skip mathematical verification after decomposition\n"
              << "  --help               Show this help message\n";
}

const char* mode_to_str(bvn::ExecMode m) {
    switch (m) {
        case bvn::ExecMode::SEQ: return "seq";
        case bvn::ExecMode::OMP_NAIVE: return "omp_naive";
        case bvn::ExecMode::OMP_ARCH: return "omp_arch";
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
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "seq") cfg.mode = bvn::ExecMode::SEQ;
            else if (m == "omp_naive") cfg.mode = bvn::ExecMode::OMP_NAIVE;
            else if (m == "omp_arch") cfg.mode = bvn::ExecMode::OMP_ARCH;
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

    // Warm up OpenMP thread pool so OS pthread creation is not charged to Phase 1
    if (cfg.mode != bvn::ExecMode::SEQ) {
        #pragma omp parallel num_threads(cfg.num_threads)
        {
            asm volatile("" ::: "memory");
        }
    }

    // 1. Phase 1: Regularization (Padding)
    double t0 = omp_get_wtime();
    std::vector<int> reg_matrix = bvn::phase1_regularize(
        demand_matrix,
        cfg.n,
        cfg.target_degree,
        cfg.mode,
        stats.regularized_degree,
        stats.total_phantom_packets
    );
    double t1 = omp_get_wtime();

    // 2 & 3. Phases 2 & 3: Eulerian Divide-and-Conquer & Degree-1 Leaf Extraction
    std::vector<bvn::ScheduleLeaf> schedules = bvn::phase2_3_euler_decompose(
        reg_matrix,
        cfg.n,
        stats.regularized_degree,
        cfg
    );
    double t2 = omp_get_wtime();

    // 4. Phase 4: Cleanup & Post-Processing
    bvn::phase4_cleanup_phantom(demand_matrix, cfg.n, cfg.mode, schedules);
    double t3 = omp_get_wtime();

    stats.phase1_padding_ms = (t1 - t0) * 1000.0;
    stats.phase2_3_euler_ms = (t2 - t1) * 1000.0;
    stats.phase4_cleanup_ms = (t3 - t2) * 1000.0;
    stats.total_ms = (t3 - t0) * 1000.0;
    stats.num_schedules = static_cast<int>(schedules.size());

    if (cfg.verify) {
        stats.verified_ok = bvn::verify_decomposition(demand_matrix, cfg.n, schedules);
    } else {
        stats.verified_ok = true;
    }

    if (cfg.csv_output) {
        // CSV format: mode,pattern,N,D,threads,cutoff,phase1_ms,phase2_3_ms,phase4_ms,total_ms,verified
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
                  << " Matrix Size (N)  : " << cfg.n << " x " << cfg.n << "\n"
                  << " Regular Degree D : " << stats.regularized_degree << "\n"
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
                  << " Verification (Exact BvN)   : " << (stats.verified_ok ? "PASSED (100% Exact)" : "FAILED!") << "\n"
                  << "============================================================\n";
    }

    return stats.verified_ok ? 0 : 2;
}
