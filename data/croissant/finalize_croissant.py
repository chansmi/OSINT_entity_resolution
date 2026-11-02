"""Build the camera-ready Croissant file from the one Hugging Face generates.

HF's auto-generated Croissant describes the data structure (distribution and
record sets) but carries no Responsible AI metadata and names the anonymous
review account as creator. This script keeps HF's structure untouched and adds:

  * the RAI fields (rai:*) and provenance (prov:*) from
    croissant_rai_opensanctions_pairs.json, each exactly once (the older
    patch wrote both `dataSocialImpact` and `rai:dataSocialImpact`, which
    resolve to the same IRI and trigger a cardinality warning);
  * the de-anonymised authors and the paper citation.

Usage:
    curl -sL https://huggingface.co/api/datasets/sanctions-er-anon/opensanctions_pairs/croissant -o croissant_hf.json
    uv run python finalize_croissant.py croissant_hf.json croissant.json
    uvx --from mlcroissant mlcroissant validate --jsonld croissant.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RAI_SRC = HERE / "croissant_rai_opensanctions_pairs.json"

AUTHORS = [
    ("Chandler Smith", "University of Oxford"),
    ("Magnus Sesodia", "University of Oxford"),
    ("Friedrich Lindenberg", "OpenSanctions"),
    ("Christian Schroeder de Witt", "University of Oxford"),
]

CITE_AS = """@inproceedings{smith2026opensanctionspairs,
  title     = {OpenSanctions Pairs: A Large-Scale Dataset for Pairwise Entity Matching},
  author    = {Smith, Chandler and Sesodia, Magnus and Lindenberg, Friedrich and Schroeder de Witt, Christian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Evaluations and Datasets Track},
  year      = {2026}
}"""

DATA_COLLECTION = (
    "Records are sourced verbatim from the OpenSanctions Pairs corpus "
    "(https://www.opensanctions.org/docs/opensource/pairs/), snapshot dated "
    "2025-12-09. OpenSanctions aggregates public sanctions lists, PEP "
    "registers, and watchlists from national and international authorities "
    "via its cross-referencing pipeline. The pairs are emitted by that "
    "pipeline as candidate matches with curator-assigned positive/negative "
    "judgements. This dataset is the unmodified snapshot, archived as a "
    "single gzipped JSON Lines file."
)

RELEASE_PLAN = (
    "The snapshot is immutable: pairs.json.gz will not be modified, so "
    "results remain comparable. Documentation and metadata fixes are made "
    "through the Hugging Face repository history. The continuously updated "
    "corpus is maintained upstream by OpenSanctions."
)


def main(src: Path, dst: Path) -> None:
    c = json.loads(src.read_text())
    rai = json.loads(RAI_SRC.read_text())

    if not c.get("recordSet"):
        sys.exit(
            f"{src} has an empty recordSet: HF has not finished (or failed) "
            "the Parquet conversion. Check the dataset viewer and retry."
        )

    ctx = c.setdefault("@context", {})
    ctx.setdefault("rai", "http://mlcommons.org/croissant/RAI/")
    ctx.setdefault("prov", "http://www.w3.org/ns/prov#")

    for k, v in rai.items():
        if k.startswith(("rai:", "prov:")):
            c[k] = v
    c["rai:dataCollection"] = DATA_COLLECTION
    c["rai:dataReleaseMaintenancePlan"] = RELEASE_PLAN

    c["creator"] = [
        {
            "@type": "sc:Person",
            "name": name,
            "affiliation": {"@type": "sc:Organization", "name": org},
        }
        for name, org in AUTHORS
    ]
    c["citeAs"] = CITE_AS
    c["datePublished"] = "2025-12-09"
    c["version"] = "1.0.0"
    c["sameAs"] = "https://github.com/chansmi/OSINT_entity_resolution"

    dst.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
