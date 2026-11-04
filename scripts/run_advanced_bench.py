#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import statistics
import datetime
import re

# Repository root, wherever it is cloned (e.g. ~/Paralell-BvN or /local/repository on CloudLab)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
BIN = os.path.join(BUILD_DIR, "bvn_bench")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(ROOT_DIR, f"results_advanced_{TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"=== Starting Advanced Architectural Benchmark Suite ===")
print(f"Directory: {RESULTS_DIR}")
print(f"Platform: 2x Intel Xeon Gold 6142 (64 cores, 2 NUMA sockets), 376GB RAM")
print("=" * 60)

report_lines = []
report_lines.append("# Advanced Architectural Benchmark Report: Parallel Birkhoff-von Neumann")
report_lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
report_lines.append(f"**Platform:** 2x Intel Xeon Gold 6142 (64 vCPUs @ 2.60GHz, 2 Sockets, 2 NUMA nodes), 376GB RAM, Ubuntu 22.04  \n")

# ------------------------------------------------------------------------------
# BATTERY 1: Hardware Performance Counters (perf stat)
# ------------------------------------------------------------------------------
print("\n[Battery 1/5] Hardware Performance Counters (perf stat) at N=2048, D=2048...")
perf_csv = os.path.join(RESULTS_DIR, "battery1_perf_counters.csv")
with open(perf_csv, "w") as f:
    f.write("mode,cycles,instructions,ipc,l1_loads,l1_misses,l1_miss_rate,llc_loads,llc_misses,llc_miss_rate,branch_miss_rate,time_s\n")

report_lines.append("## 1. Hardware Performance Counters (`perf stat` at N=2048, D=2048, Threads=64)")
report_lines.append("| Mode | Time (ms) | IPC | L1 D-Cache Miss % | LLC (L3) Miss % | Branch Miss % | Instructions (M) |")
report_lines.append("|---|---|---|---|---|---|---|")

PERF_EVENTS = "cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses"

for mode in ["seq", "omp_naive", "omp_arch", "omp_block"]:
    threads = 1 if mode == "seq" else 64
    env = os.environ.copy()
    if mode in ["omp_arch", "omp_block"]:
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "spread"
    elif mode == "omp_naive":
        env["OMP_PROC_BIND"] = "false"

    cmd = [
        "perf", "stat", "-e", PERF_EVENTS,
        BIN, "--mode", mode, "--n", "2048", "--d", "2048", "--threads", str(threads), "--no-verify", "--csv"
    ]

    # Run 5 reps and average
    reps = 5
    cycles_list, insn_list, l1_loads_list, l1_miss_list = [], [], [], []
    llc_loads_list, llc_miss_list, br_list, br_miss_list, time_list = [], [], [], [], []

    for r in range(reps):
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Parse CSV output from bvn_bench
        for line in res.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 10:
                try:
                    time_list.append(float(parts[9]))
                except ValueError:
                    pass

        # Parse perf stat from stderr
        lines = res.stderr.splitlines()
        metrics = {}
        for l in lines:
            l = l.strip()
            for ev in PERF_EVENTS.split(","):
                if ev in l:
                    m = re.search(r"([\d,]+)\s+" + re.escape(ev), l)
                    if m:
                        val = int(m.group(1).replace(",", ""))
                        metrics[ev] = val
        if "cycles" in metrics and "instructions" in metrics:
            cycles_list.append(metrics["cycles"])
            insn_list.append(metrics["instructions"])
            l1_loads_list.append(metrics.get("L1-dcache-loads", 1))
            l1_miss_list.append(metrics.get("L1-dcache-load-misses", 0))
            llc_loads_list.append(metrics.get("LLC-loads", 1))
            llc_miss_list.append(metrics.get("LLC-load-misses", 0))
            br_list.append(metrics.get("branch-instructions", 1))
            br_miss_list.append(metrics.get("branch-misses", 0))

    avg_time = statistics.mean(time_list) if time_list else 0.0
    avg_cyc = statistics.mean(cycles_list) if cycles_list else 1
    avg_ins = statistics.mean(insn_list) if insn_list else 0
    avg_l1_ld = statistics.mean(l1_loads_list) if l1_loads_list else 1
    avg_l1_ms = statistics.mean(l1_miss_list) if l1_miss_list else 0
    avg_llc_ld = statistics.mean(llc_loads_list) if llc_loads_list else 1
    avg_llc_ms = statistics.mean(llc_miss_list) if llc_miss_list else 0
    avg_br = statistics.mean(br_list) if br_list else 1
    avg_br_ms = statistics.mean(br_miss_list) if br_miss_list else 0

    ipc = avg_ins / avg_cyc if avg_cyc > 0 else 0.0
    l1_rate = (avg_l1_ms / avg_l1_ld) * 100.0 if avg_l1_ld > 0 else 0.0
    llc_rate = (avg_llc_ms / avg_llc_ld) * 100.0 if avg_llc_ld > 0 else 0.0
    br_rate = (avg_br_ms / avg_br) * 100.0 if avg_br > 0 else 0.0

    with open(perf_csv, "a") as f:
        f.write(f"{mode},{avg_cyc:.0f},{avg_ins:.0f},{ipc:.3f},{avg_l1_ld:.0f},{avg_l1_ms:.0f},{l1_rate:.2f}%,{avg_llc_ld:.0f},{avg_llc_ms:.0f},{llc_rate:.2f}%,{br_rate:.2f}%,{avg_time/1000.0:.4f}\n")

    report_lines.append(f"| `{mode}` | {avg_time:8.2f} | {ipc:5.2f} | {l1_rate:6.2f}% | {llc_rate:6.2f}% | {br_rate:5.2f}% | {avg_ins/1e6:7.1f}M |")
    print(f"  [{mode:<10}] Time: {avg_time:7.2f} ms | IPC: {ipc:4.2f} | L1-miss: {l1_rate:5.2f}% | LLC-miss: {llc_rate:5.2f}%")

# ------------------------------------------------------------------------------
# BATTERY 2: NUMA Topology & Thread Affinity
# ------------------------------------------------------------------------------
print("\n[Battery 2/5] NUMA Topology & Thread Affinity at N=2048, D=2048...")
numa_csv = os.path.join(RESULTS_DIR, "battery2_numa_affinity.csv")
with open(numa_csv, "w") as f:
    f.write("mode,affinity_policy,threads,mean_total_ms,std_total_ms,min_ms,max_ms\n")

report_lines.append("\n## 2. NUMA Topology & Thread Affinity (N=2048, D=2048)")
report_lines.append("| Policy | Threads | Mode | Mean Total (ms) | Std Dev (ms) | Description |")
report_lines.append("|---|---|---|---|---|---|")

policies = [
    ("spread", ["OMP_PLACES=cores", "OMP_PROC_BIND=spread"], "Spread threads evenly across sockets"),
    ("close", ["OMP_PLACES=cores", "OMP_PROC_BIND=close"], "Pack threads into Socket 0 first"),
    ("numa0_local", ["numactl", "--cpunodebind=0", "--membind=0"], "Confined strictly to NUMA Node 0 (Local RAM & L3)"),
    ("numa_interleave", ["numactl", "--interleave=all"], "Interleaved memory allocation across NUMA Node 0 & 1")
]

for pol_name, pol_args, pol_desc in policies:
    for threads in [16, 32, 64]:
        if pol_name == "numa0_local" and threads > 32:
            continue # Node 0 has only 32 vCPUs
        for mode in ["omp_arch", "omp_block"]:
            env = os.environ.copy()
            prefix_cmd = []
            if pol_name in ["spread", "close"]:
                env["OMP_PLACES"] = "cores"
                env["OMP_PROC_BIND"] = pol_name
            elif pol_name.startswith("numa"):
                prefix_cmd = pol_args
                env["OMP_PLACES"] = "cores"
                env["OMP_PROC_BIND"] = "spread"

            cmd = prefix_cmd + [
                BIN, "--mode", mode, "--n", "2048", "--d", "2048", "--threads", str(threads), "--no-verify", "--csv"
            ]

            times = []
            for _ in range(15):
                res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = res.stdout.strip()
                if out:
                    parts = out.split(",")
                    if len(parts) >= 10:
                        try:
                            times.append(float(parts[9]))
                        except ValueError:
                            pass

            if times:
                m = statistics.mean(times)
                s = statistics.stdev(times) if len(times) > 1 else 0.0
                with open(numa_csv, "a") as f:
                    f.write(f"{mode},{pol_name},{threads},{m:.4f},{s:.4f},{min(times):.4f},{max(times):.4f}\n")
                report_lines.append(f"| `{pol_name}` | {threads} | `{mode}` | {m:8.2f} | {s:5.2f} | {pol_desc} |")
                print(f"  [{pol_name:<15}] T={threads:<2} {mode:<10} -> {m:7.2f} ms (+/- {s:4.2f})")

# ------------------------------------------------------------------------------
# BATTERY 3: Sparsity / Duty-Cycle Sweep (D << N)
# ------------------------------------------------------------------------------
print("\n[Battery 3/5] Sparsity / Duty-Cycle Sweep (Fix N=4096, vary D)...")
sparsity_csv = os.path.join(RESULTS_DIR, "battery3_sparsity_duty_cycle.csv")
with open(sparsity_csv, "w") as f:
    f.write("mode,N,D,duty_cycle,mean_phase1_ms,mean_euler_ms,mean_cleanup_ms,mean_total_ms\n")

report_lines.append("\n## 3. Sparsity / Duty-Cycle Sweep (Fix N=4096, Threads=64)")
report_lines.append("| D | Load Ratio (D/N) | Mode | Phase 1 Padding (ms) | Phase 2&3 Decomp (ms) | Phase 4 Cleanup (ms) | Total (ms) |")
report_lines.append("|---|---|---|---|---|---|---|")

for D in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    ratio = f"{(D / 4096.0) * 100:.1f}%"
    modes_to_test = ["omp_block"]
    if D <= 2048:
        modes_to_test.append("omp_arch")

    for mode in modes_to_test:
        env = os.environ.copy()
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "spread"

        cmd = [
            BIN, "--mode", mode, "--n", "4096", "--d", str(D), "--threads", "64", "--no-verify", "--csv"
        ]

        p1_l, p2_l, p4_l, tot_l = [], [], [], []
        for _ in range(15):
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = res.stdout.strip()
            if out:
                parts = out.split(",")
                if len(parts) >= 10:
                    try:
                        p1_l.append(float(parts[6]))
                        p2_l.append(float(parts[7]))
                        p4_l.append(float(parts[8]))
                        tot_l.append(float(parts[9]))
                    except ValueError:
                        pass
        if tot_l:
            m_p1 = statistics.mean(p1_l)
            m_p2 = statistics.mean(p2_l)
            m_p4 = statistics.mean(p4_l)
            m_tot = statistics.mean(tot_l)
            with open(sparsity_csv, "a") as f:
                f.write(f"{mode},4096,{D},{ratio},{m_p1:.4f},{m_p2:.4f},{m_p4:.4f},{m_tot:.4f}\n")
            report_lines.append(f"| {D} | {ratio} | `{mode}` | {m_p1:6.2f} | {m_p2:6.2f} | {m_p4:6.2f} | **{m_tot:7.2f}** |")
            print(f"  [D={D:<4} ({ratio:>5})] {mode:<10} -> Total: {m_tot:7.2f} ms (P1: {m_p1:5.2f}, Decomp: {m_p2:6.2f}, Clean: {m_p4:5.2f})")

# ------------------------------------------------------------------------------
# BATTERY 4: Spatial Block Grid (K x K) Sensitivity for omp_block
# ------------------------------------------------------------------------------
print("\n[Battery 4/5] Block Grid (KxK) Sensitivity for omp_block at N=4096, D=4096...")
blocks_csv = os.path.join(RESULTS_DIR, "battery4_block_grid_sensitivity.csv")
with open(blocks_csv, "w") as f:
    f.write("K,block_size_B,mean_total_ms,std_total_ms,min_ms,max_ms,cache_level_fit\n")

report_lines.append("\n## 4. Spatial Block Grid ($K \\times K$) Sensitivity for `omp_block` (N=4096, D=4096, Threads=64)")
report_lines.append("| Grid K | Sub-Matrix Size B | Mean Total (ms) | Std Dev (ms) | Min (ms) | Max (ms) | Memory Hierarchy Fit |")
report_lines.append("|---|---|---|---|---|---|---|")

grid_info = {
    2: "2048 x 2048 (~16MB / LLC)",
    4: "1024 x 1024 (~4MB / LLC)",
    8: "512 x 512 (~1MB / L2 Cache)",
    16: "256 x 256 (~256KB / L2 Cache)",
    32: "128 x 128 (~64KB / L1-L2 boundary)",
    64: "64 x 64 (~16KB / 100% L1-Data Cache)"
}

for K in [2, 4, 8, 16, 32, 64]:
    B = 4096 // K
    env = os.environ.copy()
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "spread"

    cmd = [
        BIN, "--mode", "omp_block", "--n", "4096", "--d", "4096", "--threads", "64", "--blocks", str(K), "--no-verify", "--csv"
    ]

    times = []
    for _ in range(20):
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = res.stdout.strip()
        if out:
            parts = out.split(",")
            if len(parts) >= 10:
                try:
                    times.append(float(parts[9]))
                except ValueError:
                    pass
    if times:
        m = statistics.mean(times)
        s = statistics.stdev(times) if len(times) > 1 else 0.0
        fit = grid_info.get(K, "N/A")
        with open(blocks_csv, "a") as f:
            f.write(f"{K},{B},{m:.4f},{s:.4f},{min(times):.4f},{max(times):.4f},{fit}\n")
        report_lines.append(f"| {K} | {B} x {B} | **{m:7.2f}** | {s:5.2f} | {min(times):7.2f} | {max(times):7.2f} | {fit} |")
        print(f"  [K={K:<2} (Sub-matrix {B}x{B})] -> {m:7.2f} ms (+/- {s:4.2f}) [{fit}]")

# ------------------------------------------------------------------------------
# BATTERY 5: Weak Scaling
# ------------------------------------------------------------------------------
print("\n[Battery 5/5] Weak Scaling (Constant Work per Thread)...")
weak_csv = os.path.join(RESULTS_DIR, "battery5_weak_scaling.csv")
with open(weak_csv, "w") as f:
    f.write("mode,threads,N,D,mean_total_ms,std_total_ms,weak_efficiency\n")

report_lines.append("\n## 5. Weak Scaling (Constant Load per Thread: $N=512\\sqrt{T}$)")
report_lines.append("| Threads | Problem Size N | Mode | Mean Total (ms) | Std Dev (ms) | Weak Efficiency $E = T_1 / T_p$ |")
report_lines.append("|---|---|---|---|---|---|")

weak_configs = [
    (1, 512),
    (4, 1024),
    (16, 2048),
    (64, 4096)
]

base_weak = {}

for T, N in weak_configs:
    modes = ["omp_block", "omp_arch"]
    if T <= 16:
        modes.append("omp_naive")

    for mode in modes:
        env = os.environ.copy()
        if mode in ["omp_arch", "omp_block"]:
            env["OMP_PLACES"] = "cores"
            env["OMP_PROC_BIND"] = "spread"
        elif mode == "omp_naive":
            env["OMP_PROC_BIND"] = "false"

        cmd = [
            BIN, "--mode", mode, "--n", str(N), "--d", str(N), "--threads", str(T), "--no-verify", "--csv"
        ]

        times = []
        for _ in range(20):
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = res.stdout.strip()
            if out:
                parts = out.split(",")
                if len(parts) >= 10:
                    try:
                        times.append(float(parts[9]))
                    except ValueError:
                        pass
        if times:
            m = statistics.mean(times)
            s = statistics.stdev(times) if len(times) > 1 else 0.0
            if T == 1:
                base_weak[mode] = m
            eff = (base_weak.get(mode, m) / m) * 100.0 if m > 0 else 0.0
            with open(weak_csv, "a") as f:
                f.write(f"{mode},{T},{N},{N},{m:.4f},{s:.4f},{eff:.2f}%\n")
            report_lines.append(f"| {T} | {N} x {N} | `{mode}` | {m:8.2f} | {s:5.2f} | **{eff:6.1f}%** |")
            print(f"  [T={T:<2} N={N:<4}] {mode:<10} -> {m:7.2f} ms (Weak Efficiency: {eff:5.1f}%)")

# Write final report
report_md = os.path.join(RESULTS_DIR, "ADVANCED_REPORT.md")
with open(report_md, "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\n[DONE] All 5 advanced benchmark batteries finished successfully!")
print(f"Summary Report: {report_md}")
