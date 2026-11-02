#!/usr/bin/env python3
"""Render markdown result tables from whatever predictions exist.

Globs results/predictions/<model>__<split>.jsonl, computes metrics + 95% bootstrap CIs +
McNemar (vs the best-F1 model, on the primary split), and writes markdown into
results/tables/. Safe to run at any point; it uses whatever is present.

    python eval/make_tables.py [--boot 2000]
"""
import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

from significance import bootstrap_ci, load_preds, mcnemar, metrics

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
TABLES = ROOT / "results" / "tables"
PRIMARY = "common_analyst_judged"
OVERALL = "common_auto_merge"   # combined with PRIMARY per model to get overall-with-auto-merge F1


def discover():
    data = defaultdict(dict)  # split -> {model: preds}
    for f in glob.glob(str(PRED / "*__*.jsonl")):
        base = Path(f).name[:-6]
        m = re.match(r"(.+?)__(.+)$", base)
        if not m:
            continue
        model, split = m.group(1), m.group(2)
        data[split][model] = load_preds(f)
    return data


def _mrow(preds, boot):
    idxs = sorted(preds)
    pred = [preds[i][0] for i in idxs]
    truth = [preds[i][1] for i in idxs]
    mt = metrics(pred, truth)
    lo, hi = bootstrap_ci(pred, truth, "f1", boot)
    mt["f1_ci"] = (lo, hi)
    return mt


def headline(data, boot):
    if PRIMARY not in data:
        return
    prim = data[PRIMARY]
    rows = {m: _mrow(p, boot) for m, p in prim.items()}
    order = sorted(rows, key=lambda m: rows[m]["f1"], reverse=True)
    best = order[0]
    overall = data.get(OVERALL, {})
    lines = [
        f"### Apples-to-apples: all models on the identical analyst-judged split (N={rows[best]['n']})",
        "",
        "F1 with 95% bootstrap CI. `sig?` = McNemar vs the best model (accuracy, paired); "
        "n.s. means the gap is within sampling noise. `overall F1` = same model on the "
        "auto-merge-included mirror, showing the inflation.",
        "",
        "| Model | F1 (95% CI) | Acc | P | R | vs best (McNemar p) | overall F1 (w/ auto-merge) |",
        "|---|---|---|---|---|---|---|",
    ]
    bd = prim[best]
    for m in order:
        r = rows[m]
        lo, hi = r["f1_ci"]
        if m == best:
            sig = "— (best)"
        elif len(prim[m]) != len(bd):
            sig = f"n/a (N={r['n']})"   # different-N subset, not paired-comparable
        else:
            shared = sorted(set(bd) & set(prim[m]))
            ca = [bd[i][0] == bd[i][1] for i in shared]
            cb = [prim[m][i][0] == prim[m][i][1] for i in shared]
            _, _, p = mcnemar(ca, cb)
            sig = f"p={p:.3g} {'(sig)' if p < 0.05 else '(n.s.)'}"
        ov = ""
        if m in overall:  # overall F1 = analyst-judged preds + auto-merge block, same model
            av = list(prim[m].values()) + list(overall[m].values())
            ov = f"{metrics([x[0] for x in av], [x[1] for x in av])['f1']*100:.2f}"
        lines.append(f"| {m} | {r['f1']*100:.2f} [{lo*100:.2f}, {hi*100:.2f}] | "
                     f"{r['acc']*100:.2f} | {r['prec']*100:.2f} | {r['rec']*100:.2f} | {sig} | {ov} |")
    (TABLES / "headline_apples_to_apples.md").write_text("\n".join(lines) + "\n")
    print("wrote headline_apples_to_apples.md")


def robustness(data, boot):
    """Per-model F1 across splits (analyst-judged vs entity-disjoint vs temporal)."""
    splits = [s for s in ["common_analyst_judged", "entity_disjoint", "source_disjoint",
                           "temporal"] if s in data]
    models = sorted({m for s in splits for m in data[s]})
    lines = ["### Robustness: F1 across evaluation splits", "",
             "| Model | " + " | ".join(splits) + " |",
             "|---|" + "---|" * len(splits)]
    for m in models:
        cells = []
        for s in splits:
            if m in data[s]:
                r = _mrow(data[s][m], boot)
                lo, hi = r["f1_ci"]
                cells.append(f"{r['f1']*100:.2f} [{lo*100:.1f},{hi*100:.1f}]")
            else:
                cells.append("—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (TABLES / "robustness_across_splits.md").write_text("\n".join(lines) + "\n")
    print("wrote robustness_across_splits.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    data = discover()
    if not data:
        print("no predictions found yet under results/predictions/")
        return
    print("splits with predictions:", {s: len(v) for s, v in data.items()})
    headline(data, args.boot)
    robustness(data, args.boot)


if __name__ == "__main__":
    main()
