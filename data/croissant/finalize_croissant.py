"""Build the camera-ready Croissant file from the one Hugging Face generates.

HF's auto-generated Croissant describes the data structure (distribution and
record sets) but carries no Responsible AI metadata and names the anonymous
review account as creator. This script keeps HF's structure untouched and adds:

  * the RAI fields (rai:*) and provenance (prov:*) from
    rai_fields_source.json (an input, not a deliverable), each exactly once (the older
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
RAI_SRC = HERE / "rai_fields_source.json"

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
    "The current release is v1 (Hugging Face tag v1), a fixed snapshot of the "
    "OpenSanctions pairs file dated 2025-12-09. A released version is never "
    "rewritten, so cited results stay reproducible; documentation fixes go "
    "through the repository history. No further releases are currently planned. "
    "Because v1 is a snapshot, entities delisted after 2025-12-09 still appear "
    "in it; any future version will reflect upstream delistings and removals, "
    "and the terms of use require moving to it and deleting superseded copies. "
    "Access is gated on Hugging Face: users accept the licence and terms of use "
    "(non-commercial research use; no linking with external datasets to "
    "re-identify individuals or to extend use beyond identification; use of the "
    "current version) and are approved automatically. Correction and deletion "
    "requests go to the body that maintains the relevant list "
    "(https://www.opensanctions.org/docs/delisting/)."
)

DESCRIPTION = (
    "OpenSanctions Pairs: 755,540 labelled pairs of entity records (persons, "
    "organisations and other FollowTheMoney schemas) from public sanctions, "
    "watchlist and politically-exposed-person sources aggregated by "
    "OpenSanctions. Each pair is labelled positive (same real-world entity) or "
    "negative by OpenSanctions analysts, for benchmarking pairwise entity "
    "matching. Snapshot of 2025-12-09, released as v1 under CC BY-NC 4.0 with "
    "gated access."
)

PSI_LEGAL = (
    " Legal basis: the records are compiled by OpenSanctions Datenbanken GmbH "
    "(Berlin), which processes them under the EU GDPR on the basis of "
    "legitimate interest (Art. 6(1)(f); https://www.opensanctions.org/docs/privacy/). "
    "The dataset contains no records about the general public. Identifying "
    "fields are kept because matching identities is the task itself; access is "
    "gated with terms of use instead."
)

MITIGATIONS = (
    "Mitigations: access is gated on Hugging Face and requires accepting the "
    "CC BY-NC 4.0 licence and terms of use that prohibit linking the data with "
    "external datasets to re-identify individuals or to extend its use beyond "
    "identification, and that require using the current version. A synthetic "
    "preview with fabricated values lets users inspect the format without "
    "access. The dataset includes only fields the upstream sources publish. "
    "Users should consult live OpenSanctions data, not this snapshot, for any "
    "operational decision, and should not use model predictions as a substitute "
    "for an authoritative compliance workflow. Correction and deletion requests "
    "go to the listing body (https://www.opensanctions.org/docs/delisting/)."
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
    c["description"] = DESCRIPTION
    psi = c.get("rai:personalSensitiveInformation", "")
    if PSI_LEGAL.strip() not in psi:
        c["rai:personalSensitiveInformation"] = psi.rstrip() + PSI_LEGAL
    impact = c.get("rai:dataSocialImpact", "")
    if "Mitigations:" in impact:
        c["rai:dataSocialImpact"] = impact[: impact.index("Mitigations:")] + MITIGATIONS

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
