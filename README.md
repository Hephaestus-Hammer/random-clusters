# omni-random-clusters

A random clustering baseline for the
[omni-pipelines-plan](https://github.com/sta426hs2026/omni-pipelines-plan)
Omnibenchmark benchmark (STA 426, exercise 6).

Every cell is assigned to one of *k* clusters uniformly at random, where *k* is
the true number of clusters plus `d_cluster`. The labels ignore the data, so the
ARI against the ground truth should be close to 0. That gives the "no skill"
reference point the real pipelines are compared with.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `seed` | 42 | Random number generator seed, so results are reproducible |
| `d_cluster` | 0 | Offset from the true number of clusters |
| `n_comp` | 30 | Number of PCs written to `pca.tsv` |

## Outputs

The same four files as the other methods, so the metrics stage scores it unchanged:
`{dataset}.clusters.tsv` (`cell_id`, `louvain`, `leiden`, two independent random
draws), `{dataset}.pca.tsv` (a plain PCA on log-normalised counts, used only for
the structure metrics), `{dataset}.timings.json` and `{dataset}.clustering_info.json`.

It runs in the benchmark's existing `scanpy` environment.
