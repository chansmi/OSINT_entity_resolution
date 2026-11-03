#!/usr/bin/env python3
"""Build frozen, reproducible evaluation splits for OpenSanctions Pairs.

One pass streams the corpus to build canonical clusters (connected components over
positive pairs) and a compact per-pair index; a second pass writes the selected pairs
to JSONL split files. Every split is label-stratified with a fixed seed, and a
manifest.json records the seed, filters, and sha256 of every input and output so the
whole thing is auditable and re-runnable.

Usage:
    python eval/build_splits.py --config eval/config.yaml

Outputs (under data/splits/ and results/):
    common_analyst_judged.jsonl      primary apples-to-apples split (analyst-judged only)
    common_auto_merge.jsonl          auto-merge positives, combined with the above for overall F1
    entity_disjoint.jsonl            one pair per canonical cluster
    source_disjoint.jsonl            left/right share no source dataset
    temporal.jsonl                   both entities first_seen >= cutoff
    script_eval.jsonl                script-bucket-oversampled for per-script F1
    manifest.json                    seed, filters, counts, sha256 of every file
    results/cluster_sizes.json       cluster-size distribution + >2-entity stats

Each written pair carries an added "pair_key" = "{left_id}__{right_id}__{corpus_line}"
for audit joins; the model runners align predictions by position, so order is preserved.
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ----------------------------- helpers -------------------------------------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def char_script(ch):
    o = ord(ch)
    if (0x41 <= o <= 0x24F) or (0x1E00 <= o <= 0x1EFF):
        return "Latin"
    if (0x400 <= o <= 0x52F):
        return "Cyrillic"
    if (0x600 <= o <= 0x6FF) or (0x750 <= o <= 0x77F) or (0x8A0 <= o <= 0x8FF) or (0xFB50 <= o <= 0xFDFF) or (0xFE70 <= o <= 0xFEFF):
        return "Arabic"
    if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
        return "CJK"
    return None


def entity_scripts(ent):
    names = []
    if ent.get("caption"):
        names.append(ent["caption"])
    names.extend(ent.get("properties", {}).get("name", []) or [])
    scripts = set()
    for s in names:
        for ch in s:
            sc = char_script(ch)
            if sc:
                scripts.add(sc)
    return scripts


def pair_bucket(left, right):
    ls, rs = entity_scripts(left), entity_scripts(right)
    left_nonlatin = ls - {"Latin"}
    right_nonlatin = rs - {"Latin"}
    cross = (left_nonlatin and rs == {"Latin"}) or (right_nonlatin and ls == {"Latin"}) or (
        left_nonlatin and right_nonlatin and left_nonlatin != right_nonlatin)
    if cross:
        return "cross_script"
    both = ls | rs
    if "Arabic" in both:
        return "arabic"
    if "Cyrillic" in both:
        return "cyrillic"
    if "CJK" in both:
        return "cjk"
    return "latin"


# ----------------------------- pass 1 --------------------------------------

def scan_corpus(corpus_path, auto_merge_schemas):
    uf = UnionFind()
    metas = []  # (idx, lid, rid, y, is_am, bucket, cross_source, fs_min)
    am = set(auto_merge_schemas)
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            p = json.loads(line)
            l, r = p["left"], p["right"]
            lid, rid = l["id"], r["id"]
            y = 1 if p["judgement"] == "positive" else 0
            is_am = (l.get("schema") in am) or (r.get("schema") in am)
            if y == 1:
                uf.union(lid, rid)
            bucket = pair_bucket(l, r)
            cross_source = len(set(l.get("datasets", [])) & set(r.get("datasets", []))) == 0
            lfs, rfs = l.get("first_seen"), r.get("first_seen")
            fs_min = min(lfs, rfs) if (lfs and rfs) else None
            metas.append((idx, lid, rid, y, is_am, bucket, cross_source, fs_min))
    return uf, metas


# ----------------------------- selection -----------------------------------

def stratified(rng, pool, n, target_ratio):
    """pool: list of metas. Sample n keeping positive fraction ~ target_ratio."""
    pos = [m for m in pool if m[3] == 1]
    neg = [m for m in pool if m[3] == 0]
    n_pos = min(len(pos), round(n * target_ratio))
    n_neg = min(len(neg), n - n_pos)
    return rng.sample(pos, n_pos) + rng.sample(neg, n_neg)


def pair_hash(a, b):
    """sha256 of the two entity ids, sorted and joined by '|'."""
    a, b = sorted((a, b))
    return hashlib.sha256(f"{a}|{b}".encode("utf-8")).hexdigest()


class _Contamination:
    """Membership test over (left_id, right_id) tuples, backed by id pairs or their hashes."""

    def __init__(self):
        self.pairs, self.hashes = set(), set()

    def __contains__(self, key):
        return key in self.pairs or (bool(self.hashes) and pair_hash(*key) in self.hashes)

    def __len__(self):
        return len(self.pairs) + len(self.hashes)


def load_contamination(paths, root):
    """Pairs from prior eval/dev/tuning samples, to exclude from every split.

    Accepts sample files (full records), id-pair files ({"pairs": [[l, r], ...]}) or
    hashed files ({"pairs_sha256": [...]}, see pair_hash). A missing file is an error,
    since silently skipping it would change the splits.
    """
    contam = _Contamination()
    for p in paths:
        fp = root / p
        if not fp.exists():
            raise FileNotFoundError(f"contamination file missing: {fp}")
        data = json.load(open(fp, encoding="utf-8"))
        if isinstance(data, dict) and "pairs_sha256" in data:
            contam.hashes.update(data["pairs_sha256"])
            continue
        pairs = data["pairs"] if isinstance(data, dict) else data
        for it in pairs:
            ids = it if isinstance(it, list) else (it["left"]["id"], it["right"]["id"])
            contam.pairs.add(tuple(sorted(ids)))
    return contam


def build_selections(metas, uf, cfg, rng, contam):
    keep = [m for m in metas if tuple(sorted((m[1], m[2]))) not in contam]  # drop contamination
    aj = [m for m in keep if not m[4]]               # analyst-judged
    aj_ratio = sum(m[3] for m in aj) / len(aj)
    sel = {}

    # common analyst-judged
    c = cfg["splits"]["common_analyst_judged"]
    sel["common_analyst_judged"] = stratified(rng, aj, c["n"], aj_ratio)

    # auto-merge positives block (combined per model with the above for overall F1)
    am = cfg["splits"]["common_auto_merge"]
    am_pos = [m for m in keep if m[4] and m[3] == 1]
    sel["common_auto_merge"] = rng.sample(am_pos, min(am["n"], len(am_pos)))

    # entity-disjoint: one pair per canonical cluster, then stratify
    ed = cfg["splits"]["entity_disjoint"]
    shuffled = aj[:]
    rng.shuffle(shuffled)
    used, pool = set(), []
    for m in shuffled:
        cl, cr = uf.find(m[1]), uf.find(m[2])
        if cl in used or cr in used:
            continue
        used.add(cl)
        used.add(cr)
        pool.append(m)
    sel["entity_disjoint"] = stratified(rng, pool, ed["n"], aj_ratio)

    # source-disjoint: analyst pairs whose two entities share no source dataset
    sd = cfg["splits"]["source_disjoint"]
    sd_pool = [m for m in aj if m[6]]
    sd_ratio = (sum(x[3] for x in sd_pool) / len(sd_pool)) if sd_pool else aj_ratio
    sel["source_disjoint"] = stratified(rng, sd_pool, min(sd["n"], len(sd_pool)), sd_ratio)

    # temporal hold-out
    t = cfg["splits"]["temporal"]
    cutoff = t["first_seen_min"]
    tpool = [m for m in aj if m[7] and m[7] >= cutoff]
    t_ratio = (sum(m[3] for m in tpool) / len(tpool)) if tpool else aj_ratio
    sel["temporal"] = stratified(rng, tpool, min(t["n"], len(tpool)), t_ratio)

    # script eval: oversample non-Latin buckets
    se = cfg["splits"]["script_eval"]
    cap = se["per_bucket_cap"]
    by_bucket = {}
    for m in aj:
        by_bucket.setdefault(m[5], []).append(m)
    script_sel = []
    for bucket, lst in by_bucket.items():
        k = min(cap, len(lst))
        script_sel.extend(rng.sample(lst, k))
    sel["script_eval"] = script_sel

    return sel, aj_ratio


# ----------------------------- pass 2 (write) ------------------------------

def write_splits(corpus_path, sel, outdir):
    # map corpus line idx -> list of (split_name)
    idx_to_splits = {}
    for name, metas in sel.items():
        for m in metas:
            idx_to_splits.setdefault(m[0], []).append(name)
    buffers = {name: [] for name in sel}
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in idx_to_splits:
                continue
            p = json.loads(line)
            p["pair_key"] = f"{p['left']['id']}__{p['right']['id']}__{idx}"
            for name in idx_to_splits[idx]:
                buffers[name].append(p)
    paths = {}
    for name, rows in buffers.items():
        # preserve the seeded selection order (buffers were filled in corpus order;
        # re-order to the selection order so position joins are deterministic)
        order = {m[0]: i for i, m in enumerate(sel[name])}
        rows.sort(key=lambda p: order[int(p["pair_key"].rsplit("__", 1)[1])])
        path = outdir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as w:
            for p in rows:
                w.write(json.dumps(p, ensure_ascii=False) + "\n")
        paths[name] = path
    return paths


def cluster_stats(uf):
    sizes = Counter()
    roots = Counter()
    for node in list(uf.parent):
        roots[uf.find(node)] += 1
    for _, sz in roots.items():
        sizes[sz] += 1
    total = sum(roots.values())
    multi = sum(c for s, c in sizes.items() if s > 2)
    return {
        "num_entities_in_clusters": total,
        "num_clusters": len(roots),
        "clusters_size_2": sizes.get(2, 0),
        "clusters_gt2": multi,
        "max_cluster_size": max(roots.values()) if roots else 0,
        "size_histogram": dict(sorted(sizes.items())[:50]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval/config.yaml")
    ap.add_argument("--write-manifest", action="store_true",
                    help="overwrite data/splits/manifest.json instead of verifying against it")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(__file__).resolve().parents[1]
    corpus = root / cfg["corpus"]
    outdir = root / "data" / "splits"
    resdir = root / "results"
    rng = random.Random(cfg["seed"])

    print("pass 1: scanning corpus + building clusters ...")
    uf, metas = scan_corpus(corpus, cfg["auto_merge_schemas"])
    print(f"  {len(metas):,} pairs scanned")

    contam = load_contamination(cfg.get("contamination_files", []), root)
    print(f"  contamination set: {len(contam):,} prior pairs excluded")

    print("selecting splits ...")
    sel, aj_ratio = build_selections(metas, uf, cfg, rng, contam)
    for name, m in sel.items():
        npos = sum(x[3] for x in m)
        print(f"  {name}: n={len(m)} pos={npos} neg={len(m) - npos}")

    print("pass 2: writing split files ...")
    paths = write_splits(corpus, sel, outdir)

    cstats = cluster_stats(uf)
    json.dump(cstats, open(resdir / "cluster_sizes.json", "w"), indent=2)
    print(f"  clusters: {cstats['num_clusters']:,}  >2-entity: {cstats['clusters_gt2']:,}  max={cstats['max_cluster_size']}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": cfg["seed"],
        "corpus": str(cfg["corpus"]),
        "corpus_sha256": sha256_file(corpus),
        "analyst_judged_positive_ratio": round(aj_ratio, 4),
        "auto_merge_schemas": cfg["auto_merge_schemas"],
        "contamination_files": cfg.get("contamination_files", []),
        "contamination_pairs_excluded": len(contam),
        "splits": {},
    }
    for name, m in sel.items():
        npos = sum(x[3] for x in m)
        manifest["splits"][name] = {
            "file": f"data/splits/{name}.jsonl",
            "n": len(m),
            "n_pos": npos,
            "n_neg": len(m) - npos,
            "sha256": sha256_file(paths[name]),
            "filters": cfg["splits"].get(name, {"note": "overall mirror, auto-merge included"}),
        }
    mpath = outdir / "manifest.json"
    if mpath.exists() and not args.write_manifest:
        ref = json.load(open(mpath))
        ok = ref.get("corpus_sha256") == manifest["corpus_sha256"]
        for name, info in manifest["splits"].items():
            match = ref["splits"].get(name, {}).get("sha256") == info["sha256"]
            ok &= match
            print(f"  {name:24s} {'matches published checksum' if match else 'DIFFERS from published checksum'}")
        if not ok:
            sys.exit("splits do not match data/splits/manifest.json")
        print("all splits match data/splits/manifest.json")
    else:
        json.dump(manifest, open(mpath, "w"), indent=2)
        print(f"wrote manifest with {len(sel)} splits -> data/splits/manifest.json")


if __name__ == "__main__":
    main()
