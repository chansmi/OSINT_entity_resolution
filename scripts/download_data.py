#!/usr/bin/env python3
"""Download the OpenSanctions Pairs corpus and verify its checksum.

The corpus is fetched from the Hugging Face Hub (default) or directly from
OpenSanctions, written to data/raw/pairs-20251209.json.gz, and checked against
the sha256 recorded in data/splits/manifest.json, so the frozen evaluation
splits can be rebuilt bit-for-bit.

    python scripts/download_data.py                  # from the Hugging Face Hub
    python scripts/download_data.py --source opensanctions
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "sanctions-er-anon/opensanctions_pairs"
HF_FILE = "pairs.json.gz"
OPENSANCTIONS_URL = "https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz"
DEST = ROOT / "data" / "raw" / "pairs-20251209.json.gz"
MANIFEST = ROOT / "data" / "splits" / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "opensanctions"], default="hf")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    expected = json.loads(MANIFEST.read_text())["corpus_sha256"]
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    if args.dest.exists() and not args.force:
        print(f"{args.dest} exists, verifying checksum")
    elif args.source == "hf":
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
        shutil.copyfile(cached, args.dest)
    else:
        print(f"downloading {OPENSANCTIONS_URL}")
        urllib.request.urlretrieve(OPENSANCTIONS_URL, args.dest)

    got = sha256(args.dest)
    if got != expected:
        sys.exit(f"checksum mismatch for {args.dest}: expected {expected}, got {got}")
    print(f"ok: {args.dest} (sha256 {got[:12]}...)")


if __name__ == "__main__":
    main()
