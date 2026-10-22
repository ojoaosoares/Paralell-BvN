#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bvn {

/**
 * Execution mode for Phase 2 & 3 (Eulerian Divide-and-Conquer + Leaf Extraction)
 */
enum class ExecMode {
    SEQ,        // Sequential baseline
    OMP_NAIVE,  // Naive OpenMP: unaligned cursors (false sharing), unit-edge expansion, no task cutoff
    OMP_ARCH,   // Architecture-Aware OpenMP: 2-level bitmask, double TZCNT/BLSR, task cutoff
    OMP_BLOCK   // Dynamic Spatial Block-Diagonal Decomposition (K x K sub-matrices in L1 cache)
};

/**
 * Traffic matrix pattern for synthetic benchmarks
 */
enum class TrafficPattern {
    UNIFORM,    // Uniform random traffic across all ports
    SKEWED,     // Hotspot/diagonal skewed traffic (heavy contention on specific ports)
    BURSTY      // Sparse high-intensity bursts
};

/**
 * Cache-line aligned vertex cursor to prevent False Sharing (64-byte cache line on x86_64/ARM64)
 * across multi-socket NUMA domains and L1/L2/L3 caches during atomic edge claiming.
 */
struct alignas(64) AlignedCursor {
    int head = 0;
    int deg = 0;
    uint8_t padding[64 - 2 * sizeof(int)]{};
};

/**
 * Compact edge representation in Flat CSR format
 */
struct FlatEdge {
    int to;       // Destination vertex (0..N-1 in opposite partition)
    int edge_id;  // Unique undirected edge identifier (0..E-1)
};

/**
 * Weighted edge entry (u -> v with multiplicity w), packed into 8 bytes (single 64-bit register)
 */
struct alignas(8) WeightedEdge {
    uint16_t u;
    uint16_t v;
    int32_t w;

    WeightedEdge() = default;
    constexpr WeightedEdge(int u_, int v_, int w_)
        : u(static_cast<uint16_t>(u_)), v(static_cast<uint16_t>(v_)), w(w_) {}
};

/**
 * A single permutation schedule (leaf of the recursion tree, degree 1).
 * perm[u] = v means input port u is routed to output port v.
 * perm[u] = -1 means no-op (idle port after phantom traffic cleanup in Phase 4).
 */
struct ScheduleLeaf {
    int weight = 1;
    std::vector<int> perm; // size N
};

/**
 * Configuration parameters for the BvN pipeline
 */
struct Config {
    int n = 256;                                  // Number of switch ports (N x N matrix)
    int target_degree = 256;                      // Regularized degree D (power of 2)
    int num_threads = 4;                          // OpenMP thread count
    int cutoff_degree = 8;                        // Task cut-off degree D_min for OMP_ARCH
    int block_factor = 0;                         // Spatial grid KxK blocks (0 = dynamic optimal selection)
    uint64_t seed = 42;                           // RNG seed for reproducibility
    ExecMode mode = ExecMode::OMP_ARCH;           // Execution mode
    TrafficPattern pattern = TrafficPattern::UNIFORM;
    bool verify = true;                           // Run strict mathematical verification
    bool csv_output = false;                      // Output single-line CSV for scripts
    bool detailed_metrics = false;                // Collect and print deep microarchitectural & algorithmic telemetry
};

/**
 * Timing and deep telemetry statistics collected across the 4 pipeline phases
 */
struct PipelineStats {
    double phase1_padding_ms = 0.0;
    double phase2_3_euler_ms = 0.0;
    double phase4_cleanup_ms = 0.0;
    double total_ms = 0.0;
    int regularized_degree = 0;
    int num_schedules = 0;
    int64_t total_real_packets = 0;
    int64_t total_phantom_packets = 0;
    bool verified_ok = false;

    // Deep Sub-Phase Breakdown (ms)
    double sub_root_bitmask_ms = 0.0;
    double sub_root_walk_ms = 0.0;
    double sub_root_extract_ms = 0.0;
    double sub_tree_tasks_ms = 0.0;
    double sub_block_alloc_ms = 0.0;
    double sub_block_decomp_ms = 0.0;
    double sub_block_cleanup_ms = 0.0;
    double sub_block_fusion_ms = 0.0;

    // Thread Utilization & Load Balancing
    double thread_min_ms = 0.0;
    double thread_avg_ms = 0.0;
    double thread_max_ms = 0.0;
    double load_imbalance_pct = 0.0;
    double parallel_efficiency_pct = 0.0;

    // Algorithmic Graph & Split Counters
    int64_t total_splits = 0;
    int64_t symmetric_pruned_splits = 0;
    int64_t unit_odd_fast_splits = 0;
    int64_t heavy_odd_splits = 0;
    int64_t deg2_leaf_calls = 0;
    int64_t total_edges_scanned = 0;
    int64_t total_odd_edges_walked = 0;
    int empty_blocks_skipped = 0;
    int hotspot_blocks_tasked = 0;
    int min_block_degree = 0;
    int max_block_degree = 0;

    // OS / Memory / Hardware Resource Telemetry (getrusage)
    long minor_page_faults = 0;
    long major_page_faults = 0;
    long voluntary_ctx_switches = 0;
    long involuntary_ctx_switches = 0;
    long peak_rss_kb = 0;
};

} // namespace bvn
