#!/usr/bin/env python3
"""Download the OpenSanctions Pairs corpus (v1) and verify its checksum.

The dataset on the Hugging Face Hub is gated: log in (`hf auth login`, or set
HF_TOKEN) and accept the terms of use on the dataset page once; approval is
automatic. The same snapshot can also be fetched directly from OpenSanctions.

The corpus is written to data/raw/pairs-20251209.json.gz and checked against the
sha256 in data/splits/manifest.json, so the frozen evaluation splits rebuild
bit for bit (python eval/build_splits.py).

    python scripts/download_data.py                        # Hugging Face Hub, revision v1
    python scripts/download_data.py --source opensanctions # OpenSanctions download
    python scripts/download_data.py --sample               # also the 1,000-pair dev sample (Hub)
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
HF_REVISION = "v1"
HF_FILE = "pairs.json.gz"
HF_PAGE = f"https://huggingface.co/datasets/{HF_REPO}"
OPENSANCTIONS_URL = "https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz"
DEST = ROOT / "data" / "raw" / "pairs-20251209.json.gz"
SAMPLE_DEST = ROOT / "data" / "samples" / "sample_1000.json"
MANIFEST = ROOT / "data" / "splits" / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_download(filename: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        cached = hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset", revision=HF_REVISION)
    except GatedRepoError:
        sys.exit(
            f"The dataset is gated. Log in (`hf auth login` or set HF_TOKEN) and accept the terms at\n"
            f"  {HF_PAGE}\nthen rerun, or use --source opensanctions."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "opensanctions"], default="hf")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--sample", action="store_true", help=f"also fetch sample_1000.json to {SAMPLE_DEST.relative_to(ROOT)}")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    expected = json.loads(MANIFEST.read_text())["corpus_sha256"]
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    if args.dest.exists() and not args.force:
        print(f"{args.dest} exists, verifying checksum")
    elif args.source == "hf":
        hub_download(HF_FILE, args.dest)
    else:
        print(f"downloading {OPENSANCTIONS_URL}")
        urllib.request.urlretrieve(OPENSANCTIONS_URL, args.dest)

    got = sha256(args.dest)
    if got != expected:
        sys.exit(f"checksum mismatch for {args.dest}: expected {expected}, got {got}")
    print(f"ok: {args.dest} (sha256 {got[:12]}...)")

    if args.sample:
        if SAMPLE_DEST.exists() and not args.force:
            print(f"{SAMPLE_DEST} exists")
        else:
            hub_download("sample_1000.json", SAMPLE_DEST)
            print(f"ok: {SAMPLE_DEST}")


if __name__ == "__main__":
    main()
