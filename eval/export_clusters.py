#!/usr/bin/env python3
"""Export the canonical cluster structure behind the pairwise labels.

A cluster is a connected component of the graph whose edges are the positive
(same-entity) pairs, computed with the same union-find used to build the
entity-disjoint split (eval/build_splits.py). Entities that never appear in a
positive pair are singleton clusters. This supports n-ary evaluation
(clustering, transitivity) on top of the pairwise benchmark: 1,002,093
entities in 459,763 clusters, 35,341 of them with more than two entities (up
to 81). The count includes 178 entities that appear only in negative pairs,
which results/cluster_sizes.json (computed during split building) omits.

Output (Parquet, one row per entity in the corpus):
    entity_id      OpenSanctions entity id as it appears in the pairs file
    cluster_id     smallest entity_id in the cluster (stable, deterministic)
    cluster_size   number of entities in the cluster

The file is derived from the gated corpus, so it is written locally and not
committed; the dataset release on Hugging Face carries the same file.

    python eval/export_clusters.py                       # data/clusters/clusters.parquet
    python eval/export_clusters.py --out some/path.parquet
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from build_splits import UnionFind  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="eval/config.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "clusters" / "clusters.parquet")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    corpus = ROOT / cfg["corpus"]
    uf = UnionFind()
    with gzip.open(corpus, "rt", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            lid, rid = p["left"]["id"], p["right"]["id"]
            uf.find(lid)
            uf.find(rid)
            if p["judgement"] == "positive":
                uf.union(lid, rid)

    members = defaultdict(list)
    for node in list(uf.parent):
        members[uf.find(node)].append(node)
    rows_e, rows_c, rows_s = [], [], []
    for nodes in members.values():
        cid, size = min(nodes), len(nodes)
        for n in sorted(nodes):
            rows_e.append(n)
            rows_c.append(cid)
            rows_s.append(size)

    order = sorted(range(len(rows_e)), key=lambda i: (rows_c[i], rows_e[i]))
    table = pa.table({
        "entity_id": [rows_e[i] for i in order],
        "cluster_id": [rows_c[i] for i in order],
        "cluster_size": pa.array([rows_s[i] for i in order], pa.int32()),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")

    sizes = Counter(len(v) for v in members.values())
    multi = {k: v for k, v in members.items() if len(v) > 1}
    print(f"entities: {table.num_rows:,}  clusters (incl. singletons): {len(members):,}")
    print(f"clusters with >=2 entities: {len(multi):,}  with >2: {sum(1 for v in multi.values() if len(v) > 2):,}"
          f"  max size: {max(sizes)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
