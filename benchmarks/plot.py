"""Plot latency vs request rate from run_serving.py CSVs (paper Fig. 12 style).

    python benchmarks/plot.py results/e1/*.csv --out results/e1_sharegpt.png --title "OPT-1.3B, ShareGPT"

Each CSV row is one (system, rate) point. Lines are drawn per system; the y-axis is
the paper's normalized latency (seconds per output token). The rate at which a curve
shoots up is that system's capacity.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ORDER = ["paged", "orca-oracle", "orca-pow2", "orca-max"]
COLORS = {"paged": "#2b6cb0", "orca-oracle": "#dd6b20", "orca-pow2": "#d69e2e", "orca-max": "#718096"}
LABELS = {"paged": "PagedAttention (ours)", "orca-oracle": "Orca (Oracle)", "orca-pow2": "Orca (Pow2)",
          "orca-max": "Orca (Max)"}


def read(paths: list[Path]) -> dict[str, list[dict]]:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        with p.open() as fh:
            for row in csv.DictReader(fh):
                by_system[row["system"]].append(row)
    for rows in by_system.values():
        rows.sort(key=lambda r: float(r["rate"]))
    return by_system


def capacity(rows: list[dict], slo: float) -> float:
    """Highest measured rate whose normalized latency stays under `slo` s/token."""
    ok = [float(r["rate"]) for r in rows
          if not math.isnan(float(r["norm_latency_mean"])) and float(r["norm_latency_mean"]) <= slo
          and r["timed_out"] == "0"]
    return max(ok) if ok else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--ymax", type=float, default=None, help="clip the y-axis (s/token)")
    ap.add_argument("--slo", type=float, default=None,
                    help="latency target for the capacity summary (default: 4x paged's lowest latency)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read(args.csvs)
    systems = [s for s in ORDER if s in data] + [s for s in data if s not in ORDER]
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in systems:
        rows = [r for r in data[s] if not math.isnan(float(r["norm_latency_mean"]))]
        ax.plot([float(r["rate"]) for r in rows], [float(r["norm_latency_mean"]) for r in rows],
                marker="o", color=COLORS.get(s), label=LABELS.get(s, s), linewidth=2 if s == "paged" else 1.5)
    ax.set_xlabel("request rate (req/s)")
    ax.set_ylabel("normalized latency (s/token)")
    if args.ymax:
        ax.set_ylim(0, args.ymax)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(args.title, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    if "paged" in data:
        base = min(float(r["norm_latency_mean"]) for r in data["paged"] if not math.isnan(float(r["norm_latency_mean"])))
        slo = args.slo or 4 * base
        cap = {s: capacity(data[s], slo) for s in systems}
        print(f"Capacity = highest rate with normalized latency <= {slo:.4f} s/token:")
        for s in systems:
            ratio = f"  (paged is {cap['paged'] / cap[s]:.2f}x)" if s != "paged" and cap[s] > 0 else ""
            print(f"  {LABELS.get(s, s):<22} {cap[s]:g} req/s{ratio}")


if __name__ == "__main__":
    main()
