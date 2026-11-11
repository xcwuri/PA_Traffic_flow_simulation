#!/usr/bin/env python3
"""Run comparable configurations and write median-time speedup/efficiency data."""
import argparse
import csv
import os
import random
import re
import statistics
import subprocess
import sys


def integer_list(value):
    # argparse helper for options such as --threads 1,2,4.
    try:
        result = [int(item) for item in value.split(",") if item.strip()]
        if not result or any(item < 1 for item in result):
            raise ValueError
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc


def run(command, env):
    # Run one simulation and extract the elapsed time printed by the C++ executable.
    completed = subprocess.run(command, env=env, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    match = re.search(r"elapsed_seconds=([0-9.eE+-]+)", completed.stdout)
    if not match:
        raise RuntimeError("Could not find elapsed_seconds in program output:\n" + completed.stdout)
    return float(match.group(1))


def main():
    # Command-line options define a workload and the parallel configurations to compare.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="./build/traffic_sim")
    parser.add_argument("--output", default="benchmarks.csv")
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1,
                        help="unrecorded launch per configuration (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=integer_list, default=[1, 2, 4])
    parser.add_argument("--ranks", type=integer_list, default=[1, 2, 4])
    parser.add_argument("--modes", default="serial,openmp,mpi,hybrid")
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--oversubscribe", action="store_true",
                        help="pass Open MPI's --oversubscribe flag before -np")
    parser.add_argument("--omp-wait-policy", type=str.upper,
                        choices=("ACTIVE", "PASSIVE"), default="PASSIVE",
                        help="OpenMP idle-worker policy (default: PASSIVE, useful for hybrid runs)")
    parser.add_argument("--raw-output", help="per-run CSV path (default: <output stem>_raw.csv)")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups nonnegative")

    requested = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if "serial" not in requested:
        parser.error("include serial mode to establish the speedup baseline")
    if any(mode not in {"serial", "openmp", "mpi", "hybrid"} for mode in requested):
        parser.error("modes must be serial, openmp, mpi, or hybrid")

    base = [args.exe, "--length", str(args.length), "--steps", str(args.steps),
            "--density", str(args.density), "--seed", str(args.seed)]
    # Encode each test as (mode, MPI ranks, OpenMP threads); serial is the baseline.
    configs = [("serial", 1, 1)]
    if "openmp" in requested:
        configs += [("openmp", 1, n) for n in args.threads]
    if "mpi" in requested:
        configs += [("mpi", n, 1) for n in args.ranks]
    if "hybrid" in requested:
        configs += [("hybrid", n, t) for n in args.ranks for t in args.threads]

    def command_for(ranks):
        command = [args.launcher]
        if args.oversubscribe:
            command.append("--oversubscribe")
        command += ["-np", str(ranks)]
        return command

    def execute(config):
        # Set OpenMP controls consistently and add mpirun only for distributed modes.
        mode, ranks, threads = config
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_WAIT_POLICY"] = args.omp_wait_policy
        if args.oversubscribe:
            # Keep CTest and direct Open MPI invocations consistent on hosts
            # whose advertised process slots are fewer than requested ranks.
            env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
        command = base + ["--mode", mode]
        if mode in {"mpi", "hybrid"}:
            command = command_for(ranks) + command
        return run(command, env)

    # Warm up once, then shuffle each measurement round to reduce order-related bias.
    order_rng = random.Random(args.seed)
    warmup_order = list(configs)
    order_rng.shuffle(warmup_order)
    for config in warmup_order:
        for _ in range(args.warmups):
            execute(config)

    samples_by_config = {config: [] for config in configs}
    raw_rows = []
    for repeat in range(1, args.repeats + 1):
        round_order = list(configs)
        order_rng.shuffle(round_order)
        for mode, ranks, threads in round_order:
            elapsed = execute((mode, ranks, threads))
            samples_by_config[(mode, ranks, threads)].append(elapsed)
            raw_rows.append({"mode": mode, "ranks": ranks, "threads": threads,
                             "workers": ranks * threads, "length": args.length,
                             "steps": args.steps, "density": args.density,
                             "seed": args.seed, "omp_wait_policy": args.omp_wait_policy,
                             "repeat": repeat,
                             "elapsed_seconds": f"{elapsed:.9g}"})

    # Summarize repeated timings by median, then derive speedup and efficiency.
    summary = []
    for mode, ranks, threads in configs:
        samples = samples_by_config[(mode, ranks, threads)]
        median = statistics.median(samples)
        summary.append({"mode": mode, "ranks": ranks, "threads": threads,
                        "workers": ranks * threads, "length": args.length,
                        "steps": args.steps, "density": args.density,
                        "seed": args.seed, "omp_wait_policy": args.omp_wait_policy,
                        "repeats": len(samples),
                        "median_seconds": median,
                        "samples_seconds": ";".join(f"{sample:.9g}" for sample in samples)})
        print(f"{mode:7s} ranks={ranks} threads={threads} median={median:.6f}s")

    serial_time = next(row["median_seconds"] for row in summary if row["mode"] == "serial")
    for row in summary:
        row["speedup"] = serial_time / row["median_seconds"] if row["median_seconds"] else 0.0
        row["efficiency"] = row["speedup"] / row["workers"]

    raw_path = args.raw_output
    if raw_path is None:
        stem, _ = os.path.splitext(args.output)
        raw_path = stem + "_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as stream:
        fields = ["mode", "ranks", "threads", "workers", "length", "steps",
                  "density", "seed", "omp_wait_policy", "repeat", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        fields = list(summary[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote per-run measurements to {raw_path}")
    print(f"Wrote median summary to {args.output}")
    print("Use tools/plot_results.py to create speedup and efficiency graphs.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        sys.exit(1)
