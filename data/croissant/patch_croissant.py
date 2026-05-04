"""Patch the HF auto-generated Croissant with NeurIPS-required RAI fields."""

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "croissant_hf.json"
DST = HERE / "croissant.json"

c = json.loads(SRC.read_text())

ctx = c.setdefault("@context", {})
for unused in ("equivalentProperty", "samplingRate", "examples"):
    ctx.pop(unused, None)
ctx.setdefault("rai", "http://mlcommons.org/croissant/RAI/")
for k in [
    "dataCollection",
    "dataBiases",
    "personalSensitiveInformation",
    "dataUseCases",
    "dataSocialImpact",
    "dataReleaseMaintenancePlan",
    "dataLimitation",
    "dataAnnotationProtocol",
    "dataAnnotationPlatform",
    "dataAnnotationAnalysis",
    "annotationsPerItem",
    "annotatorDemographics",
    "machineAnnotationTools",
]:
    ctx.setdefault(k, f"rai:{k}")

c["datePublished"] = "2025-12-09"
c["version"] = "1.0.0"

c["citeAs"] = (
    "OpenSanctions. Pairs: Cross-Referencing Training Data. "
    "https://www.opensanctions.org/docs/opensource/pairs/. "
    "Snapshot dated 2025-12-09."
)

c["dataCollection"] = (
    "Records are sourced verbatim from the OpenSanctions Pairs corpus "
    "(https://www.opensanctions.org/docs/opensource/pairs/), snapshot dated "
    "2025-12-09. OpenSanctions aggregates public sanctions lists, PEP "
    "registers, and watchlists from national and international authorities "
    "via its cross-referencing pipeline. The pairs are emitted by that "
    "pipeline as candidate matches with curator-assigned positive/negative "
    "judgements. This dataset is the unmodified snapshot, archived as a "
    "single gzipped JSON Lines file."
)

c["dataBiases"] = (
    "Geographic and political bias: coverage is skewed toward jurisdictions "
    "that publish sanctions, PEP, and watchlist data in machine-readable "
    "form (US OFAC, EU, UK, UN); jurisdictions with less open data "
    "infrastructure are under-represented. Script and language bias: names "
    "are recorded in source-list scripts and inconsistently romanised; "
    "Latin-script names and transliteration variants of Cyrillic, Arabic, "
    "and CJK names dominate. Class imbalance: 76.9% positive vs 23.1% "
    "negative pairs reflects the upstream candidate-generation strategy and "
    "is not a representative base rate for screening. Population bias: the "
    "corpus over-represents persons and organisations of interest to "
    "compliance authorities and is not representative of general populations."
)

c["personalSensitiveInformation"] = (
    "Contains personally identifiable information about real individuals: "
    "names, aliases, dates of birth, nationalities, addresses, and "
    "identification numbers. All records are drawn from publicly published "
    "sanctions lists, PEP registers, and official watchlists maintained by "
    "governmental and intergovernmental bodies; no private or non-public "
    "information is included. Because the data concerns individuals "
    "associated with sanctions or political exposure, downstream use must "
    "respect upstream source licences and avoid contexts that imply guilt "
    "or wrongdoing for any specific individual."
)

c["dataUseCases"] = (
    "Intended uses: benchmarking entity resolution and record linkage "
    "systems; evaluating LLMs and fine-tuned models on noisy multilingual "
    "records; methods research on prompt optimisation, fine-tuning, and "
    "retrieval for entity matching; education and reproducibility for "
    "compliance and OSINT research. Out-of-scope uses: real-world sanctions "
    "enforcement, watchlist deployment, or any decision affecting "
    "individuals without human review and adherence to applicable "
    "regulations."
)

c["dataSocialImpact"] = (
    "Improved entity resolution on sanctions data benefits compliance, "
    "anti-money-laundering, and transparency research. Risks include "
    "misidentification (false positives can affect innocent individuals "
    "sharing names with sanctioned persons) and over-reliance on automated "
    "matching in high-stakes contexts. The benchmark is intended for "
    "research and evaluation, not deployment; practitioners should pair any "
    "system trained or evaluated here with human review before action."
)

c["dataLimitation"] = (
    "Synthetic data disclosure: the dataset contains no synthetic, "
    "generated, or LLM-produced records; all entity records and pair "
    "labels are sourced verbatim from OpenSanctions. Limitations: labels "
    "reflect upstream curator decisions and may contain noise; "
    "transliteration is inconsistent across sources; the distribution is "
    "not representative of general-purpose record linkage."
)

DST.write_text(json.dumps(c, indent=2, ensure_ascii=False))
print(f"Wrote {DST} ({DST.stat().st_size} bytes)")
