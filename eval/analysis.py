#!/usr/bin/env python3
"""Secondary analyses: per-script F1, cost frontier, error taxonomy.

Subcommands (each writes a CSV under results/ and prints a summary):

  per-script  --pred FILE --split data/splits/script_eval.jsonl
      F1 per script bucket (latin / cyrillic / arabic / cjk / cross_script), joined by
      position. Shows where LLMs actually fail (answers the bias / "is it solved" points).

  cost        --pred name=FILE [name=FILE ...] --config eval/config.yaml
      $ per 1,000 pairs from tokens_used x provider pricing, for the cost-accuracy frontier.

  taxonomy    --pred FILE [--label NAME]
      Categorize each error (FP / FN) by the model's own reasoning text into
      transliteration / date-or-id / field-conflict / missing-field / name-collision / other.

  clusters    --stats results/cluster_sizes.json
      Summarize the canonical-cluster size distribution and the >2-entity share.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from build_splits import pair_bucket  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load_preds(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _y(v):
    return 1 if str(v).lower().startswith("pos") else 0


def _f1(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return (2 * p * r / (p + r)) if (p + r) else 0.0, p, r


def per_script(args):
    split = [json.loads(l) for l in open(args.split, encoding="utf-8") if l.strip()]
    preds = _load_preds(args.pred)
    if len(preds) != len(split):
        print(f"WARN: {len(preds)} preds vs {len(split)} split rows; joining by min length")
    buckets = {}
    for i, (sp, pr) in enumerate(zip(split, preds)):
        b = pair_bucket(sp["left"], sp["right"])
        pred, truth = _y(pr["classification"]), _y(pr["ground_truth"])
        d = buckets.setdefault(b, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
        d["n"] += 1
        if pred and truth:
            d["tp"] += 1
        elif pred and not truth:
            d["fp"] += 1
        elif not pred and truth:
            d["fn"] += 1
        else:
            d["tn"] += 1
    label = args.label or Path(args.pred).stem
    out = RESULTS / (f"per_script_{label}.csv" if args.label else "per_script.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n", "f1", "precision", "recall", "tp", "fp", "fn", "tn"])
        for b, d in sorted(buckets.items()):
            f1, p, r = _f1(d["tp"], d["fp"], d["fn"])
            w.writerow([b, d["n"], round(f1 * 100, 2), round(p * 100, 2), round(r * 100, 2),
                        d["tp"], d["fp"], d["fn"], d["tn"]])
            print(f"  {b:14s} n={d['n']:4d}  F1={f1*100:5.2f}  P={p*100:5.2f}  R={r*100:5.2f}")
    print(f"-> {out}")


def cost(args):
    import yaml
    cfg = yaml.safe_load(open(args.config))
    price = {m["name"]: (m.get("price_in"), m.get("price_out")) for m in cfg["models"]}
    out = RESULTS / "cost.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "prompt_tok", "completion_tok", "usd_per_1k_pairs"])
        for spec in args.pred:
            name, path = spec.split("=", 1)
            rows = _load_preds(path)
            pin, pout = price.get(name, (None, None))
            ptok = sum((r.get("tokens_used") or {}).get("prompt", 0) for r in rows)
            ctok = sum((r.get("tokens_used") or {}).get("completion", 0) for r in rows)
            if pin is None:
                usd1k = ""
            else:
                usd = ptok / 1e6 * pin + ctok / 1e6 * pout
                usd1k = round(usd / max(1, len(rows)) * 1000, 4)
            w.writerow([name, len(rows), ptok, ctok, usd1k])
            print(f"  {name:22s} n={len(rows):5d} prompt={ptok:8d} completion={ctok:7d} ${usd1k}/1k")
    print(f"-> {out}")


TAXO = {
    "transliteration": ["translit", "script", "cyrillic", "arabic", "chinese", "romaniz",
                         "spelling", "latin", "diacritic"],
    "date_or_id": ["date", "birth", "id number", "idnumber", "identifier", "passport",
                   "off by", "digit", "tax", "imo"],
    "field_conflict": ["conflict", "contradict", "father", "nationality", "address",
                       "country", "gender", "different"],
    "missing_field": ["missing", "absent", "no data", "not provided", "lack of", "empty"],
    "name_collision": ["same name", "common name", "identical name", "name match", "shared name"],
}


def _categorize(text):
    t = (text or "").lower()
    for cat, kws in TAXO.items():
        if any(k in t for k in kws):
            return cat
    return "other"


def taxonomy(args):
    rows = _load_preds(args.pred)
    fp_cats, fn_cats = {}, {}
    for r in rows:
        pred, truth = _y(r["classification"]), _y(r["ground_truth"])
        if pred == truth:
            continue
        cat = _categorize(r.get("reasoning"))
        (fp_cats if pred and not truth else fn_cats)[cat] = \
            (fp_cats if pred and not truth else fn_cats).get(cat, 0) + 1
    label = args.label or Path(args.pred).stem
    out = RESULTS / f"taxonomy_{label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error_type", "category", "count"])
        for c, n in sorted(fp_cats.items(), key=lambda x: -x[1]):
            w.writerow(["false_positive", c, n])
        for c, n in sorted(fn_cats.items(), key=lambda x: -x[1]):
            w.writerow(["false_negative", c, n])
    print(f"  {label}: FP={sum(fp_cats.values())} {dict(fp_cats)}")
    print(f"  {label}: FN={sum(fn_cats.values())} {dict(fn_cats)}")
    print(f"-> {out}")


def clusters(args):
    s = json.load(open(args.stats))
    n = s["num_clusters"]
    print(f"  clusters={n:,}  size-2={s['clusters_size_2']:,}  "
          f">2-entity={s['clusters_gt2']:,} ({s['clusters_gt2']/max(1,n)*100:.1f}%)  "
          f"max={s['max_cluster_size']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("per-script"); a.add_argument("--pred", required=True); a.add_argument("--split", required=True); a.add_argument("--label"); a.set_defaults(fn=per_script)
    a = sub.add_parser("cost"); a.add_argument("--pred", action="append", required=True); a.add_argument("--config", default="eval/config.yaml"); a.set_defaults(fn=cost)
    a = sub.add_parser("taxonomy"); a.add_argument("--pred", required=True); a.add_argument("--label"); a.set_defaults(fn=taxonomy)
    a = sub.add_parser("clusters"); a.add_argument("--stats", default="results/cluster_sizes.json"); a.set_defaults(fn=clusters)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
