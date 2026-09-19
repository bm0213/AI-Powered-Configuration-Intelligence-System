r"""
generate_data.py
-----------------
Creates a synthetic dataset that mimics real execution logs from a
configurable/randomized test system (the kind SanDisk described in the
problem statement).

v2 UPDATE: added Environment columns (node/hardware conditions) and
Log-derived columns (error/warning/timeout counts observed during the
run). These give the root-cause analysis step (Q5/Q6 in the brief)
something real to chew on beyond just config + randomization.

Why synthetic data with "hidden patterns"?
Because right now you don't have real SanDisk logs. A dashboard built on
100% random data would find nothing (because there's nothing there).
So this script quietly embeds a handful of realistic cause-and-effect
patterns into the noise. Your dashboard's job in later steps is to
DISCOVER these patterns automatically -- which proves your analytics
pipeline actually works, and gives you a truthful demo.

Run (from the backend/ folder):
    python generate_data.py

Output:
    ../data/raw/execution_logs.csv  (55,000 rows x 170 columns)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 55000  # within your requested 50,000-60,000 range

# ----------------------------------------------------------------------
# 1. CONFIGURATION PARAMETERS (things engineers explicitly set)
# ----------------------------------------------------------------------
cache_policy = np.random.choice(["Adaptive", "Static", "LRU", "FIFO"], N, p=[0.3, 0.3, 0.25, 0.15])
scheduler = np.random.choice(["Dynamic", "Static", "RoundRobin"], N, p=[0.35, 0.35, 0.3])
cache_size_mb = np.random.choice([64, 128, 256, 512, 1024], N)
memory_alloc_mb = np.random.choice([512, 1024, 2048, 4096, 8192], N)
cpu_cores = np.random.choice([1, 2, 4, 8, 16], N)
gpu_enabled = np.random.choice([True, False], N, p=[0.4, 0.6])
compiler_opt_level = np.random.choice(["O0", "O1", "O2", "O3"], N, p=[0.1, 0.2, 0.4, 0.3])
thread_pool_size = np.random.choice([2, 4, 8, 16, 32], N)
batch_size = np.random.choice([16, 32, 64, 128, 256], N)
timeout_ms = np.random.choice([100, 500, 1000, 5000, 10000], N)
retry_count = np.random.choice([0, 1, 2, 3, 5], N)

# ~90 extra feature-flag style config switches (mostly noise, a couple matter)
feature_flags = {}
for i in range(1, 91):
    feature_flags[f"feature_flag_{i}"] = np.random.choice([True, False], N, p=[0.3, 0.7])

# One flag is deliberately risky -- it boosts throughput but raises failure risk
feature_flags["feature_flag_7"] = np.random.choice([True, False], N, p=[0.35, 0.65])

# ----------------------------------------------------------------------
# 2. RANDOMIZATION PARAMETERS (things that vary per run, not chosen by engineers)
# ----------------------------------------------------------------------
random_seed = np.random.randint(1, 100000, N)
seed_group = (random_seed % 10)  # bucket seeds into 10 groups, mirrors the brief's "Seed Group 8"
workload_type = np.random.choice(["ReadHeavy", "WriteHeavy", "Mixed", "Sequential", "Random"], N)
traffic_pattern = np.random.choice(["Burst", "Steady", "Spiky"], N)
timing_variation_ms = np.random.exponential(20, N)
env_perturbation_level = np.random.uniform(0, 1, N)
jitter_ms = np.random.exponential(5, N)

# ~45 more randomized noise variables
rand_vars = {}
for i in range(1, 46):
    rand_vars[f"rand_var_{i}"] = np.round(np.random.normal(0, 1, N), 3)

# ----------------------------------------------------------------------
# 2b. ENVIRONMENT PARAMETERS (physical/hardware conditions of the node
#     the test ran on -- NEW in this version)
# ----------------------------------------------------------------------
node_region = np.random.choice(["us-east", "us-west", "eu-central", "ap-south"], N)

# Planted pattern E: seed_group 8's runs happen to land on hotter nodes
# (simulates a real scenario: a batch of runs scheduled onto flaky hardware)
base_temp = np.random.normal(45, 5, N)
host_temperature_c = base_temp + np.where(seed_group == 8, 15, 0) + np.random.normal(0, 2, N)
host_temperature_c = np.clip(host_temperature_c, 20, None)

voltage_v = np.clip(np.random.normal(12, 0.3, N) + np.where(seed_group == 8, 0.4, 0), 10, 14)
cpu_load_pct = np.clip(np.random.uniform(10, 90, N), 0, 100)
memory_pressure_pct = np.clip(
    np.random.uniform(5, 95, N) + np.where(memory_alloc_mb <= 1024, 15, 0), 0, 100
)
ambient_humidity_pct = np.clip(np.random.normal(45, 10, N), 5, 95)

# ----------------------------------------------------------------------
# 3. EMBEDDED GROUND-TRUTH PATTERNS (this is what your dashboard should "discover")
# ----------------------------------------------------------------------
# Base failure probability
fail_prob = np.full(N, 0.06)

# Pattern A: Seed Group 8 is flaky -> contributes disproportionately to failures
fail_prob += np.where(seed_group == 8, 0.30, 0.0)

# Pattern B: feature_flag_7 raises failure risk but also raises throughput
fail_prob += np.where(feature_flags["feature_flag_7"], 0.10, 0.0)

# Pattern C: low memory + high thread count is unstable
low_mem_high_thread = (memory_alloc_mb <= 1024) & (thread_pool_size >= 16)
fail_prob += np.where(low_mem_high_thread, 0.15, 0.0)

# Pattern D: Adaptive cache + Dynamic scheduler is the "golden combo" -> safer & faster
golden_combo = (cache_policy == "Adaptive") & (scheduler == "Dynamic")
fail_prob -= np.where(golden_combo, 0.03, 0.0)

# Pattern E (NEW): overheating nodes fail more, independent of seed group
overheating = host_temperature_c > 65
fail_prob += np.where(overheating, 0.08, 0.0)

# Pattern F (NEW): aggressive timeout setting fails more often under load
aggressive_timeout = timeout_ms <= 100
fail_prob += np.where(aggressive_timeout & (cpu_load_pct > 70), 0.07, 0.0)

fail_prob = np.clip(fail_prob, 0.01, 0.95)
passed = np.random.binomial(1, 1 - fail_prob)
failed_flag = 1 - passed  # convenience for generating log features below

# ----------------------------------------------------------------------
# 3b. LOG-DERIVED FEATURES (NEW -- things you'd only know from reading the
#     execution logs after the run: error/warning counts, timeouts, etc.
#     These are generated AS A CONSEQUENCE of the same patterns above,
#     the way real logs would reflect a real failure.)
# ----------------------------------------------------------------------
error_count = np.random.poisson(
    lam=0.3 + failed_flag * 2.5 + overheating * 0.5
)
warning_count = np.random.poisson(
    lam=1.0 + failed_flag * 1.5 + memory_pressure_pct / 100
)

timeout_occurred = np.random.binomial(
    1, np.clip(0.03 + np.where(aggressive_timeout & (failed_flag == 1), 0.45, 0.0), 0, 1)
).astype(bool)

memory_error_occurred = np.random.binomial(
    1, np.clip(0.02 + np.where(low_mem_high_thread & (failed_flag == 1), 0.5, 0.0), 0, 1)
).astype(bool)

scheduler_error_occurred = np.random.binomial(
    1, np.clip(0.02 + np.where(feature_flags["feature_flag_7"] & (failed_flag == 1), 0.35, 0.0), 0, 1)
).astype(bool)

retry_attempts_logged = np.random.poisson(
    lam=np.clip(retry_count * (0.3 + failed_flag * 1.2), 0.1, None)
)

# ----------------------------------------------------------------------
# 4. OUTCOME METRICS (also shaped by the same patterns, so they're consistent)
# ----------------------------------------------------------------------
base_exec_time = 500 - (cpu_cores * 10) - (thread_pool_size * 2) + timing_variation_ms
base_exec_time += np.where(golden_combo, -60, 0)      # golden combo is faster
execution_time_ms = np.clip(base_exec_time + np.random.normal(0, 25, N), 50, None)

base_throughput = 1000 + cpu_cores * 40 + cache_size_mb * 0.3
base_throughput += np.where(golden_combo, 220, 0)               # ~22% lift, matches the brief's example
base_throughput += np.where(feature_flags["feature_flag_7"], 180, 0)  # flag_7 boosts throughput too
throughput = np.clip(base_throughput + np.random.normal(0, 40, N), 100, None)

accuracy = np.clip(np.random.normal(0.95, 0.03, N) - fail_prob * 0.2, 0.5, 1.0)
resource_consumption_pct = np.clip(np.random.normal(55, 15, N) + thread_pool_size * 0.5, 5, 100)
reliability_score = np.clip((1 - fail_prob) * 100 + np.random.normal(0, 3, N), 0, 100)

# ----------------------------------------------------------------------
# 5. ASSEMBLE THE DATAFRAME
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "run_id": [f"RUN-{i:06d}" for i in range(N)],
    "cache_policy": cache_policy,
    "scheduler": scheduler,
    "cache_size_mb": cache_size_mb,
    "memory_alloc_mb": memory_alloc_mb,
    "cpu_cores": cpu_cores,
    "gpu_enabled": gpu_enabled,
    "compiler_opt_level": compiler_opt_level,
    "thread_pool_size": thread_pool_size,
    "batch_size": batch_size,
    "timeout_ms": timeout_ms,
    "retry_count": retry_count,
    **feature_flags,
    "random_seed": random_seed,
    "seed_group": seed_group,
    "workload_type": workload_type,
    "traffic_pattern": traffic_pattern,
    "timing_variation_ms": np.round(timing_variation_ms, 2),
    "env_perturbation_level": np.round(env_perturbation_level, 3),
    "jitter_ms": np.round(jitter_ms, 2),
    **rand_vars,
    # Environment (NEW)
    "node_region": node_region,
    "host_temperature_c": np.round(host_temperature_c, 2),
    "voltage_v": np.round(voltage_v, 3),
    "cpu_load_pct": np.round(cpu_load_pct, 2),
    "memory_pressure_pct": np.round(memory_pressure_pct, 2),
    "ambient_humidity_pct": np.round(ambient_humidity_pct, 2),
    # Log-derived (NEW)
    "error_count": error_count,
    "warning_count": warning_count,
    "timeout_occurred": timeout_occurred,
    "memory_error_occurred": memory_error_occurred,
    "scheduler_error_occurred": scheduler_error_occurred,
    "retry_attempts_logged": retry_attempts_logged,
    # Outcomes
    "passed": passed,
    "execution_time_ms": np.round(execution_time_ms, 2),
    "throughput": np.round(throughput, 2),
    "accuracy": np.round(accuracy, 4),
    "resource_consumption_pct": np.round(resource_consumption_pct, 2),
    "reliability_score": np.round(reliability_score, 2),
})

out_path = "../data/raw/execution_logs.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} rows x {len(df.columns)} columns -> {out_path}")
print(f"Overall pass rate: {df['passed'].mean():.1%}")
print(f"Seed group 8 failure rate: {(1 - df[df.seed_group==8]['passed'].mean()):.1%}")
print(f"Other seed groups failure rate: {(1 - df[df.seed_group!=8]['passed'].mean()):.1%}")
print(f"Overheating (>65C) failure rate: {(1 - df[df.host_temperature_c>65]['passed'].mean()):.1%}")
print(f"Non-overheating failure rate: {(1 - df[df.host_temperature_c<=65]['passed'].mean()):.1%}")
print(f"Avg error_count when FAIL: {df[df.passed==0]['error_count'].mean():.2f}")
print(f"Avg error_count when PASS: {df[df.passed==1]['error_count'].mean():.2f}")
print(f"timeout_occurred rate when FAIL: {df[df.passed==0]['timeout_occurred'].mean():.1%}")
print(f"timeout_occurred rate when PASS: {df[df.passed==1]['timeout_occurred'].mean():.1%}")