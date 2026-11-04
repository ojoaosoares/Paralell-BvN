#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import statistics
import datetime

REPS = 30
# Repository root, wherever it is cloned (e.g. ~/Paralell-BvN or /local/repository on CloudLab)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
BIN = os.path.join(BUILD_DIR, "bvn_bench")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(ROOT_DIR, f"results_statistical_{TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_CSV = os.path.join(RESULTS_DIR, "raw_runs.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_stats.csv")
REPORT_MD = os.path.join(RESULTS_DIR, "REPORT.md")

print(f"=== Birkhoff-von Neumann Statistical Benchmark Suite ===")
print(f"Machine: 64 vCPUs (2x Xeon Gold 6142), 376GB RAM")
print(f"Repetitions per test: {REPS}")
print(f"Results Directory: {RESULTS_DIR}")
print("=" * 60)

raw_file = open(RAW_CSV, "w")
raw_file.write("rep,mode,pattern,N,D,threads,cutoff,phase1_ms,phase2_3_ms,phase4_ms,total_ms,schedules,frame_eff,verified\n")
raw_file.flush()

configs = []

# 1. Mode comparison for N in [256, 512, 1024, 2048]
for N in [256, 512, 1024, 2048]:
    D = N
    configs.append(("seq", "uniform", N, D, 1, 8, "Exp1_ModeComp"))
    configs.append(("omp_naive", "uniform", N, D, 64, 8, "Exp1_ModeComp"))
    configs.append(("omp_arch", "uniform", N, D, 64, 8, "Exp1_ModeComp"))
    configs.append(("omp_block", "uniform", N, D, 64, 8, "Exp1_ModeComp"))

# 2. Extreme scale for omp_block and omp_arch
for N in [4096, 8192, 16384]:
    D = N
    if N <= 4096:
        configs.append(("omp_arch", "uniform", N, D, 64, 16, "Exp2_ExtremeScale"))
    configs.append(("omp_block", "uniform", N, D, 64, 8, "Exp2_ExtremeScale"))

# 3. Strong Scaling at N=2048
for t in [1, 2, 4, 8, 16, 32, 64]:
    configs.append(("omp_naive", "uniform", 2048, 2048, t, 8, "Exp3_StrongScaling"))
    configs.append(("omp_arch", "uniform", 2048, 2048, t, 8, "Exp3_StrongScaling"))
    configs.append(("omp_block", "uniform", 2048, 2048, t, 8, "Exp3_StrongScaling"))

# 4. Traffic Patterns at N=1024
for pat in ["skewed", "bursty"]:
    configs.append(("omp_arch", pat, 1024, 1024, 64, 8, "Exp4_Patterns"))
    configs.append(("omp_block", pat, 1024, 1024, 64, 8, "Exp4_Patterns"))

# Deduplicate configs while preserving order
seen = set()
unique_configs = []
for c in configs:
    key = (c[0], c[1], c[2], c[3], c[4], c[5])
    if key not in seen:
        seen.add(key)
        unique_configs.append(c)

data = {}
total_tasks = len(unique_configs)
print(f"Total test configurations: {total_tasks} ({total_tasks * REPS} total benchmark invocations)\n")

for idx, (mode, pattern, N, D, threads, cutoff, exp_group) in enumerate(unique_configs, 1):
    tag = f"[{idx}/{total_tasks}] {mode} N={N} D={D} T={threads} {pattern}"
    print(f"{tag:<55} ... running {REPS} reps", flush=True)

    env = os.environ.copy()
    if mode in ["omp_arch", "omp_block"]:
        env["OMP_PLACES"] = "cores"
        env["OMP_PROC_BIND"] = "spread"
    elif mode == "omp_naive":
        env["OMP_PROC_BIND"] = "false"

    cmd = [
        BIN,
        "--mode", mode,
        "--pattern", pattern,
        "--n", str(N),
        "--d", str(D),
        "--threads", str(threads),
        "--cutoff", str(cutoff),
        "--no-verify",
        "--csv"
    ]

    key = (mode, pattern, N, D, threads, cutoff)
    data[key] = {
        "exp_group": exp_group,
        "phase1": [],
        "phase2_3": [],
        "phase4": [],
        "total": [],
        "status": "PASS"
    }

    # Warmup run (discarded)
    try:
        subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    except Exception as e:
        print(f"Warmup error for {tag}: {e}")

    for rep in range(1, REPS + 1):
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = res.stdout.strip()
        if not out or res.returncode != 0:
            print(f"Error in rep {rep} for {tag}: returncode={res.returncode}, stderr={res.stderr.strip()}")
            continue
        raw_file.write(f"{rep},{out}\n")
        raw_file.flush()
        parts = out.split(",")
        # Format: mode,pattern,N,D,threads,cutoff,phase1_ms,phase2_3_ms,phase4_ms,total_ms,schedules,frame_eff,verified
        if len(parts) >= 11:
            try:
                p1 = float(parts[6])
                p2_3 = float(parts[7])
                p4 = float(parts[8])
                tot = float(parts[9])
                status = parts[-1]
                data[key]["phase1"].append(p1)
                data[key]["phase2_3"].append(p2_3)
                data[key]["phase4"].append(p4)
                data[key]["total"].append(tot)
                if status != "PASS":
                    data[key]["status"] = status
            except ValueError:
                pass

    if data[key]["total"]:
        m_tot = statistics.mean(data[key]["total"])
        s_tot = statistics.stdev(data[key]["total"]) if len(data[key]["total"]) > 1 else 0.0
        print(f"   -> mean: {m_tot:8.3f} ms | std: {s_tot:7.3f} ms (min: {min(data[key]['total']):8.3f}, max: {max(data[key]['total']):8.3f})")

raw_file.close()

# Compute summary stats
summary_file = open(SUMMARY_CSV, "w")
summary_file.write("exp_group,mode,pattern,N,D,threads,cutoff,samples,mean_total_ms,std_total_ms,min_total_ms,max_total_ms,mean_euler_ms,std_euler_ms,mean_cleanup_ms\n")

report_lines = []
report_lines.append("# CloudLab Benchmark Report: Parallel Birkhoff-von Neumann")
report_lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
report_lines.append(f"**Platform:** 2x Intel Xeon Gold 6142 (64 vCPUs @ 2.60GHz), 376GB RAM, Ubuntu 22.04  ")
report_lines.append(f"**Statistical Repetitions:** {REPS} runs per point (with 1 initial warm-up discarded)  \n")

report_lines.append("## 1. Full Mode Comparison (N from 256 to 2048, Threads=64)")
report_lines.append("| N | Mode | Mean Total (ms) | Std Dev (ms) | Min (ms) | Max (ms) | Speedup vs Seq |")
report_lines.append("|---|---|---|---|---|---|---|")

seq_means = {}
for (mode, pattern, N, D, threads, cutoff), info in data.items():
    if mode == "seq" and pattern == "uniform":
        if info["total"]:
            seq_means[N] = statistics.mean(info["total"])

for (mode, pattern, N, D, threads, cutoff), info in data.items():
    if not info["total"]: continue
    m_tot = statistics.mean(info["total"])
    s_tot = statistics.stdev(info["total"]) if len(info["total"]) > 1 else 0.0
    mn_tot = min(info["total"])
    mx_tot = max(info["total"])
    m_eul = statistics.mean(info["phase2_3"])
    s_eul = statistics.stdev(info["phase2_3"]) if len(info["phase2_3"]) > 1 else 0.0
    m_clean = statistics.mean(info["phase4"])
    summary_file.write(f"{info['exp_group']},{mode},{pattern},{N},{D},{threads},{cutoff},{len(info['total'])},{m_tot:.4f},{s_tot:.4f},{mn_tot:.4f},{mx_tot:.4f},{m_eul:.4f},{s_eul:.4f},{m_clean:.4f}\n")

    if info["exp_group"] == "Exp1_ModeComp":
        seq_m = seq_means.get(N, m_tot)
        spd = f"{seq_m / m_tot:6.2f}x" if m_tot > 0 else "N/A"
        report_lines.append(f"| {N} | `{mode}` | {m_tot:8.3f} | {s_tot:6.3f} | {mn_tot:8.3f} | {mx_tot:8.3f} | **{spd}** |")

report_lines.append("\n## 2. Extreme Scale Breakdown (omp_block vs omp_arch, Threads=64)")
report_lines.append("| N | Mode | Mean Total (ms) | Euler/Decomp (ms) | Cleanup (ms) | Std Dev (ms) |")
report_lines.append("|---|---|---|---|---|---|")
for (mode, pattern, N, D, threads, cutoff), info in data.items():
    if info["exp_group"] == "Exp2_ExtremeScale":
        if not info["total"]: continue
        m_tot = statistics.mean(info["total"])
        s_tot = statistics.stdev(info["total"]) if len(info["total"]) > 1 else 0.0
        m_eul = statistics.mean(info["phase2_3"])
        m_cl = statistics.mean(info["phase4"])
        report_lines.append(f"| {N} | `{mode}` | {m_tot:8.3f} | {m_eul:8.3f} | {m_cl:8.3f} | {s_tot:6.3f} |")

report_lines.append("\n## 3. Strong Scaling at N=2048, D=2048")
report_lines.append("| Threads | Mode | Mean Total (ms) | Std Dev (ms) | Min (ms) | Max (ms) | Speedup vs 1-thread |")
report_lines.append("|---|---|---|---|---|---|---|")
base_t1 = {}
for (mode, pattern, N, D, threads, cutoff), info in data.items():
    if info["exp_group"] == "Exp3_StrongScaling" and threads == 1 and info["total"]:
        base_t1[mode] = statistics.mean(info["total"])

for (mode, pattern, N, D, threads, cutoff), info in data.items():
    if info["exp_group"] == "Exp3_StrongScaling":
        if not info["total"]: continue
        m_tot = statistics.mean(info["total"])
        s_tot = statistics.stdev(info["total"]) if len(info["total"]) > 1 else 0.0
        mn_tot = min(info["total"])
        mx_tot = max(info["total"])
        t1 = base_t1.get(mode, m_tot)
        spd = f"{t1 / m_tot:5.2f}x" if m_tot > 0 else "N/A"
        report_lines.append(f"| {threads} | `{mode}` | {m_tot:8.3f} | {s_tot:6.3f} | {mn_tot:8.3f} | {mx_tot:8.3f} | **{spd}** |")

report_lines.append("\n## 4. Traffic Pattern Sensitivity (N=1024, Threads=64)")
report_lines.append("| Pattern | Mode | Mean Total (ms) | Std Dev (ms) | Min (ms) | Max (ms) |")
report_lines.append("|---|---|---|---|---|---|")
for (mode, pattern, N, D, threads, cutoff), info in data.items():
    if info["exp_group"] in ["Exp4_Patterns", "Exp1_ModeComp"] and N == 1024 and mode in ["omp_arch", "omp_block"]:
        if not info["total"]: continue
        m_tot = statistics.mean(info["total"])
        s_tot = statistics.stdev(info["total"]) if len(info["total"]) > 1 else 0.0
        report_lines.append(f"| {pattern} | `{mode}` | {m_tot:8.3f} | {s_tot:6.3f} | {min(info['total']):8.3f} | {max(info['total']):8.3f} |")

summary_file.close()

with open(REPORT_MD, "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\n[DONE] Statistical benchmark completed successfully!")
print(f"Raw data: {RAW_CSV}")
print(f"Summary data: {SUMMARY_CSV}")
print(f"Markdown report: {REPORT_MD}")
