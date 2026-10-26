#!/usr/bin/env python3
"""Omnibenchmark metric collector: gather the ARI from every metrics.json.

Called by `ob run` once all metrics are computed, as

    run.py --output_dir DIR --name ID --metrics.metrics FILE [FILE ...]

Each metrics file lives at the end of a chain of `stage/module/.<hash>`
folders, and every non-default `.<hash>` folder holds a parameters.json.
Walking up that chain tells us which dataset, method and parameters
produced each number, so nothing has to be looked up by hand.

Writes to DIR:
  ari_summary.tsv  one row per run: dataset, method, parameters, ARI (Leiden, Louvain)
  all_metrics.tsv  every agreement metric, long format
  ari_barplot.png  ARI per run, Leiden and Louvain side by side
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Collect ARI across all runs")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--metrics.metrics", dest="metrics", nargs="+", type=Path, required=True)
    return p.parse_args()


def lineage(metrics_file: Path) -> dict:
    """Walk up stage/module/.<hash> folders, collecting module ids and parameters."""
    info = {}
    d = metrics_file.parent
    while d.name.startswith("."):
        module, stage = d.parent.name, d.parent.parent.name
        info[f"{stage}_module"] = module
        pfile = d / "parameters.json"
        if pfile.exists():
            for k, v in json.loads(pfile.read_text()).items():
                info[f"{stage}.{k}"] = v
        d = d.parent.parent.parent
    return info


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, long_rows = [], []
    for f in args.metrics:
        m = json.loads(f.read_text())
        meta = lineage(f)
        meta["metrics_file"] = str(f)
        rows.append({
            **meta,
            "ari_leiden": m["agreement"]["ari"]["leiden"],
            "ari_louvain": m["agreement"]["ari"]["louvain"],
            "n_clusters_leiden": m.get("n_clusters", {}).get("leiden"),
            "n_clusters_louvain": m.get("n_clusters", {}).get("louvain"),
        })
        for metric, by_algo in m["agreement"].items():
            for algo, value in by_algo.items():
                long_rows.append({**meta, "metric": metric, "clustering": algo, "value": value})

    df = pd.DataFrame(rows)

    # Modules don't all share parameters (e.g. a random baseline has a seed but
    # no n_comp), which leaves gaps; pandas then stores whole numbers as floats.
    # Turn those columns back into integers so 50 isn't shown as 50.0.
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s) and s.notna().any() and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64")

    # A short run label: method name (or module id) plus only the parameters
    # that differ between runs that actually have them, so adding
    # n_comp: [30, 50] shows up as "n_comp=50".
    method_params = [c for c in df.columns if c.startswith("methods.") and c != "methods.method_name"]
    varying = [c for c in method_params if df[c].dropna().astype(str).nunique() > 1]

    def label(r):
        name = r.get("methods.method_name")
        base = str(name) if pd.notna(name) else str(r.get("methods_module"))
        extra = ", ".join(f"{c.split('.', 1)[1]}={r[c]}" for c in varying if pd.notna(r[c]))
        ds = r.get("datasets.dataset_name")
        multi_ds = "datasets.dataset_name" in df and df["datasets.dataset_name"].nunique() > 1
        out = f"{base} ({extra})" if extra else base
        return f"{ds}: {out}" if multi_ds else out

    df.insert(0, "run", df.apply(label, axis=1))
    df = df.sort_values("ari_leiden", ascending=False)

    lead = ["run", "ari_leiden", "ari_louvain", "n_clusters_leiden", "n_clusters_louvain"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(args.output_dir / "ari_summary.tsv", sep="\t", index=False)
    pd.DataFrame(long_rows).to_csv(args.output_dir / "all_metrics.tsv", sep="\t", index=False)

    plot(df, args.output_dir / "ari_barplot.png")

    print(df[lead].to_string(index=False))


def plot(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    colors = {"Leiden": "#2a78d6", "Louvain": "#eb6834"}

    n = len(df)
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 1.2 + 0.7 * n), facecolor=surface)
    ax.set_facecolor(surface)

    y = range(n)[::-1]  # best run at the top
    for i, (name, col) in enumerate([("Leiden", "ari_leiden"), ("Louvain", "ari_louvain")]):
        pos = [yy + (h / 2 if i == 0 else -h / 2) for yy in y]
        bars = ax.barh(pos, df[col], height=h - 0.04, color=colors[name], label=name)
        for b, v in zip(bars, df[col]):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}".replace("-0.000", "0.000"), va="center", fontsize=9, color=ink)

    ax.set_yticks(list(y), df["run"], color=ink)
    ax.set_xlim(min(0, df[["ari_leiden", "ari_louvain"]].min().min() - 0.05), 1.08)
    ax.set_xlabel("Adjusted Rand index (vs. ground truth)", color=muted)
    ax.set_title("ARI per pipeline run", loc="left", color=ink, fontsize=12)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.xaxis.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", colors=muted, length=0)
    ax.tick_params(axis="y", colors=ink, length=0)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
