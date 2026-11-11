#!/usr/bin/env python3
"""Plot speedup and efficiency from tools/benchmark.py output."""
import argparse
import csv


def main():
    # Read the benchmark summary and choose the output image path.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", default="benchmarks.csv")
    parser.add_argument("--output", default="performance.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise SystemExit("Install matplotlib to plot: python3 -m pip install matplotlib") from exc

    with open(args.csv_file, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No benchmark rows found in {args.csv_file}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"serial": "#555555", "openmp": "#2878b5", "mpi": "#d95319", "hybrid": "#2a9d55"}
    markers = {"serial": "o", "openmp": "s", "mpi": "^", "hybrid": "D"}
    # Draw one curve per mode; hybrid curves are split by MPI rank count.
    for mode in ("serial", "openmp", "mpi", "hybrid"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        if not mode_rows:
            continue
        if mode == "hybrid":
            # Separate curves by rank count so equal worker totals such as
            # 1x2 and 2x1 remain visible as distinct hybrid configurations.
            groups = {}
            for row in mode_rows:
                groups.setdefault(int(row["ranks"]), []).append(row)
        else:
            groups = {None: mode_rows}
        for ranks, group in sorted(groups.items(), key=lambda item: item[0] or 0):
            group.sort(key=lambda row: (int(row["workers"]), int(row["threads"])))
            x = [int(row["workers"]) for row in group]
            name = mode.capitalize() if mode != "hybrid" else f"Hybrid ({ranks} rank{'s' if ranks != 1 else ''})"
            color = colors[mode]
            marker = markers[mode]
            linestyle = "-"
            if mode == "hybrid":
                color = "#7a4fa3" if ranks == 1 else colors[mode]
                marker = "D" if ranks == 1 else "P"
                linestyle = "--" if ranks == 1 else "-"
            axes[0].plot(x, [float(row["speedup"]) for row in group],
                         marker=marker, color=color, linestyle=linestyle, label=name)
            axes[1].plot(x, [float(row["efficiency"]) for row in group],
                         marker=marker, color=color, linestyle=linestyle, label=name)
    # Reference lines show ideal linear speedup and perfect efficiency.
    max_workers = max([int(row["workers"]) for row in rows] or [1])
    axes[0].plot([1, max_workers], [1, max_workers], "k--", alpha=0.45, label="Ideal")
    axes[0].set(title="Speedup", xlabel="Workers (ranks × threads)", ylabel="T₁ / Tₚ")
    axes[1].set(title="Parallel efficiency", xlabel="Workers (ranks × threads)", ylabel="Speedup / workers")
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.45, label="Ideal")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
        axis.set_xlim(left=1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    first = rows[0]
    fig.suptitle(f"Traffic simulation: length={first['length']}, steps={first['steps']}, "
                 f"density={first['density']}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
