#!/usr/bin/env python3
"""Random clustering baseline for the omni-pipelines-plan benchmark.

Assigns every cell to one of k clusters uniformly at random, where k is the
true number of clusters plus `d_cluster` (the same rule the real pipelines use).
Because the labels ignore the data, its ARI shows what "no skill" looks like:
it should sit close to 0.

Writes the same four files as the other methods so the metrics stage can
score it unchanged:
  {name}.clusters.tsv         cell_id, louvain, leiden (two independent random draws)
  {name}.pca.tsv              cell_id, PC1..PCn (a plain PCA, so structure metrics can run)
  {name}.timings.json
  {name}.clustering_info.json
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scanpy as sc

parser = argparse.ArgumentParser(description="Random clustering baseline")
parser.add_argument("--output_dir", "-o", required=True)
parser.add_argument("--name", "-n", required=True)
parser.add_argument("--data.h5ad", dest="data_h5ad", required=True)
parser.add_argument("--data.clusters_truth_num", dest="clusters_truth_num_path", required=True)
parser.add_argument("--seed", type=int, default=42, help="random seed, for reproducibility")
parser.add_argument("--d_cluster", type=int, default=0,
                    help="offset from the true number of clusters")
parser.add_argument("--n_comp", type=int, default=30, help="PCs written to pca.tsv")
args, _ = parser.parse_known_args()

adata = sc.read_h5ad(args.data_h5ad)
adata.var_names_make_unique()
with open(args.clusters_truth_num_path) as f:
    k = max(1, int(f.read()) + args.d_cluster)

# random labels: two independent draws so the louvain/leiden columns differ
t0 = time.perf_counter()
rng = np.random.default_rng(args.seed)
labels = pd.DataFrame({
    "cell_id": adata.obs_names,
    "louvain": rng.integers(0, k, adata.n_obs),
    "leiden": rng.integers(0, k, adata.n_obs),
})
t_random = time.perf_counter() - t0

# a basic PCA, only so the metrics stage has coordinates for silhouette etc.
t0 = time.perf_counter()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
n_comp = min(args.n_comp, adata.n_obs - 1, adata.n_vars - 1)
sc.pp.pca(adata, n_comps=n_comp, random_state=args.seed)
t_pca = time.perf_counter() - t0

out = args.output_dir
labels.to_csv(os.path.join(out, f"{args.name}.clusters.tsv"), sep="\t", index=False)

pd.DataFrame(
    adata.obsm["X_pca"],
    index=adata.obs_names,
    columns=[f"PC{i + 1}" for i in range(n_comp)],
).to_csv(os.path.join(out, f"{args.name}.pca.tsv"), sep="\t", index_label="cell_id")

with open(os.path.join(out, f"{args.name}.timings.json"), "w") as f:
    json.dump({"pca": t_pca, "louvain": t_random, "leiden": t_random}, f, indent=2)

with open(os.path.join(out, f"{args.name}.clustering_info.json"), "w") as f:
    json.dump({"resolutions": {"louvain": None, "leiden": None},
               "num_runs": {"louvain": 1, "leiden": 1},
               "n_clusters": k, "seed": args.seed}, f, indent=2)

print(f"Assigned {adata.n_obs} cells to {k} random clusters (seed {args.seed})")
