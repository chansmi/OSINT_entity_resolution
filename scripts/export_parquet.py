#!/usr/bin/env python3
"""Convert the corpus or an evaluation split to Parquet.

Entity `properties` differ per FollowTheMoney schema, so a fixed Arrow struct
cannot hold them. Each row therefore keeps a few flat, queryable columns plus
the full left/right entity as a JSON string, which round-trips losslessly:

    judgement, left_id, left_schema, left_caption, left_datasets,
    right_id, right_schema, right_caption, right_datasets,
    left_json, right_json, [pair_key], row

Usage:
    python scripts/export_parquet.py data/raw/pairs-20251209.json.gz data/parquet/pairs.parquet
    python scripts/export_parquet.py --splits          # every data/splits/*.jsonl -> data/parquet/splits/

Read back with pandas, polars, DuckDB or `datasets`:
    pd.read_parquet("data/parquet/splits/common_analyst_judged.parquet")
    scripts/load_data.py: load_pairs("...parquet") yields the original nested dicts.
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_pairs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = pa.schema([
    ("row", pa.int64()),
    ("judgement", pa.string()),
    ("left_id", pa.string()),
    ("left_schema", pa.string()),
    ("left_caption", pa.string()),
    ("left_datasets", pa.list_(pa.string())),
    ("right_id", pa.string()),
    ("right_schema", pa.string()),
    ("right_caption", pa.string()),
    ("right_datasets", pa.list_(pa.string())),
    ("left_json", pa.string()),
    ("right_json", pa.string()),
    ("pair_key", pa.string()),
])


def to_row(i, pair):
    left, right = pair["left"], pair["right"]
    return {
        "row": i,
        "judgement": pair.get("judgement"),
        "left_id": left.get("id"),
        "left_schema": left.get("schema"),
        "left_caption": left.get("caption"),
        "left_datasets": left.get("datasets") or [],
        "right_id": right.get("id"),
        "right_schema": right.get("schema"),
        "right_caption": right.get("caption"),
        "right_datasets": right.get("datasets") or [],
        "left_json": json.dumps(left, ensure_ascii=False),
        "right_json": json.dumps(right, ensure_ascii=False),
        "pair_key": pair.get("pair_key"),
    }


def export(src: Path, dst: Path, batch_size: int = 50_000) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with pq.ParquetWriter(dst, SCHEMA, compression="zstd") as w:
        batch = []
        for i, pair in enumerate(load_pairs(src, quiet=True)):
            batch.append(to_row(i, pair))
            if len(batch) >= batch_size:
                w.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                n += len(batch)
                batch = []
        if batch:
            w.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            n += len(batch)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", type=Path)
    ap.add_argument("dst", nargs="?", type=Path)
    ap.add_argument("--splits", action="store_true", help="export every data/splits/*.jsonl")
    args = ap.parse_args()

    if args.splits:
        out = ROOT / "data" / "parquet" / "splits"
        for f in sorted((ROOT / "data" / "splits").glob("*.jsonl")):
            n = export(f, out / f"{f.stem}.parquet")
            print(f"{f.name}: {n:,} rows -> {out / (f.stem + '.parquet')}")
    elif args.src and args.dst:
        n = export(args.src, args.dst)
        print(f"{n:,} rows -> {args.dst}")
    else:
        ap.error("give SRC DST, or --splits")


if __name__ == "__main__":
    main()
