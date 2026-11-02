#!/usr/bin/env python3
"""
Programmatic builder for the paper's result artifacts.

Replaces the hand-assembled tables. Reads results/sample_1000/manifest.json and, for
every row:
  - if a per-sample predictions file is present, recomputes accuracy /
    precision / recall / F1 and the confusion matrix from scratch, and computes
    a bootstrap 95% CI on F1 (paired-capable);
  - otherwise falls back to the reported values, deriving integer confusion
    counts from the Table 4 fractions and the row's test_n so it can still emit
    an analytic (Wilson) 95% CI on accuracy.

Emits, into results/sample_1000/generated/:
  - table3.tex           main results table (with a 95% CI column)
  - table4.tex           per-method confusion matrix
  - figure2_data.csv     F1 vs release date (plot data)
  - reconciliation.md    recomputed-vs-reported diff + provenance / TODO report
  - significance.md      CIs for every model and paired McNemar where possible

Only numpy + stdlib are required (no scipy). Run:
  python scripts/build_tables.py
  python scripts/build_tables.py --manifest results/sample_1000/manifest.json --out results/sample_1000/generated
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Z95 = 1.959963984540054  # standard normal 0.975 quantile

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Per-sample loading and metric computation
# --------------------------------------------------------------------------- #

def load_predictions(path: Path) -> Tuple[List[int], List[int]]:
    """Load a per-sample JSONL of {classification, ground_truth, ...} lines.

    Returns (y_true, y_pred) as 0/1 lists, positive == 1. Rows whose
    classification is missing/error are dropped (and reported by the caller via
    the returned lengths).
    """
    y_true, y_pred = [], []
    dropped = 0
    for r in iter_json_objects(path):
        gt = r.get("ground_truth")
        pr = r.get("classification")
        if gt is None or pr is None:
            dropped += 1
            continue
        pr = str(pr).lower().strip()
        gt = str(gt).lower().strip()
        if pr not in ("positive", "negative"):
            dropped += 1  # error / abstain rows do not enter the metric
            continue
        y_true.append(1 if gt == "positive" else 0)
        y_pred.append(1 if pr == "positive" else 0)
    load_predictions.last_dropped = dropped
    load_predictions.last_corrupt = iter_json_objects.last_corrupt
    return y_true, y_pred


load_predictions.last_dropped = 0
load_predictions.last_corrupt = 0


def iter_json_objects(path: Path):
    """Yield JSON objects from a file that is JSONL, contains objects with
    embedded literal newlines, or has corrupt segments from interleaved
    concurrent writes (observed in some parallel API runs). Uses a streaming
    decoder and skips unparseable regions, tracking the drop count on the
    function attribute `last_corrupt` for the caller to report."""
    text = (path if isinstance(path, Path) else Path(path)).read_text()
    dec = json.JSONDecoder()
    i, n, corrupt = 0, len(text), 0
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            # jump to the next line that starts a fresh object
            nxt = text.find("\n{", i)
            if nxt == -1:
                corrupt += 1
                break
            corrupt += 1
            i = nxt + 1
            continue
        yield obj
        i = end
    iter_json_objects.last_corrupt = corrupt


iter_json_objects.last_corrupt = 0


def counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": 100 * acc, "prec": 100 * prec, "recall": 100 * rec,
            "f1": 100 * f1, "n": n}


# --------------------------------------------------------------------------- #
# Confidence intervals (pure numpy / stdlib)
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a proportion, as percentages."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (center - half), 100 * (center + half))


def bootstrap_f1_ci(y_true: List[int], y_pred: List[int],
                    n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI on F1 (as percentages)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        t = yt[idx[b]]
        p = yp[idx[b]]
        tp = np.sum((t == 1) & (p == 1))
        fp = np.sum((t == 0) & (p == 1))
        fn = np.sum((t == 1) & (p == 0))
        denom = 2 * tp + fp + fn
        f1s[b] = (2 * tp / denom) if denom else 0.0
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return (100 * lo, 100 * hi)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact (binomial) McNemar two-sided p-value for discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial with p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- #
# Row processing
# --------------------------------------------------------------------------- #

def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fully-resolved row: recomputed metrics + CI + provenance."""
    out = dict(row)
    pf = row.get("predictions_file")
    reported = row.get("reported", {})

    if pf and (REPO / pf).exists():
        y_true, y_pred = load_predictions(REPO / pf)
        c = counts(y_true, y_pred)
        m = metrics_from_counts(c)
        f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred)
        out.update({
            "source": "per-sample (recomputed)",
            "computed": m, "counts": c,
            "ci_kind": "bootstrap F1",
            "ci": (f1_lo, f1_hi),
            "dropped": load_predictions.last_dropped,
            "corrupt": load_predictions.last_corrupt,
            "_y_true": y_true, "_y_pred": y_pred,
        })
    else:
        # aggregate fallback: reported metrics + Wilson CI on accuracy from n
        n = row.get("test_n")
        m = {"acc": reported.get("acc"), "prec": reported.get("prec"),
             "recall": reported.get("recall"), "f1": reported.get("f1"), "n": n}
        ci = None
        if n and reported.get("acc") is not None:
            k = round(reported["acc"] / 100 * n)
            ci = wilson_ci(k, n)
        out.update({
            "source": "reported (aggregate)",
            "computed": m, "counts": None,
            "ci_kind": "Wilson acc" if ci else "none",
            "ci": ci,
        })
    return out


def fmt(x: Optional[float], d: int = 2) -> str:
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "--"


def fmt_ci(row: Dict[str, Any]) -> str:
    ci = row.get("ci")
    if not ci:
        return "--"
    lo, hi = ci
    return f"[{lo:.1f}, {hi:.1f}]"


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #

def emit_table3(rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"% Auto-generated by scripts/build_tables.py. Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Main results. Metrics recomputed from stored per-sample "
        r"predictions where available (\emph{per-sample}); otherwise the "
        r"submitted values are shown (\emph{reported}). The 95\% CI is a "
        r"bootstrap interval on F1 for per-sample rows and a Wilson interval on "
        r"accuracy otherwise.}",
        r"\label{tab:main}",
        r"\begin{tabular}{llrrrrl}", r"\toprule",
        r"Model & Config & Acc & F1 & Prec & Recall & 95\% CI \\",
        r"\midrule",
    ]
    for r in rows:
        m = r["computed"]
        star = r"$^\dagger$" if r["source"].startswith("per-sample") else ""
        lines.append(
            f"{r['model']}{star} & {r['config']} & {fmt(m['acc'])} & "
            f"{fmt(m['f1'])} & {fmt(m['prec'])} & {fmt(m['recall'])} & "
            f"{fmt_ci(r)} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\\[2pt]\footnotesize $^\dagger$ recomputed from stored per-sample "
        r"predictions; CI is a 10k-sample bootstrap on F1.",
        r"\end{table}",
    ]
    return "\n".join(lines)


def emit_table4(confusion: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    # prefer recomputed fractions where we have per-sample data
    recomputed = {}
    for r in rows:
        if r.get("counts") and r["config"].startswith("0-shot") and "opt" not in r["config"]:
            c = r["counts"]
            n = sum(c.values())
            recomputed[r["model"]] = {k: c[k] / n for k in ("tp", "tn", "fp", "fn")}
    lines = [
        r"% Auto-generated by scripts/build_tables.py. Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-method confusion matrix on the test split (0-shot, "
        r"unoptimized). Fractions recomputed from per-sample predictions where "
        r"available, else transcribed from run logs.}",
        r"\label{tab:confusion}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & TP\% & TN\% & FP\% & FN\% \\", r"\midrule",
    ]
    for c in confusion:
        src = recomputed.get(c["method"], c)
        mark = r"$^\dagger$" if c["method"] in recomputed else ""
        lines.append(
            f"{c['method']}{mark} & {src['tp']:.3f} & {src['tn']:.3f} & "
            f"{src['fp']:.3f} & {src['fn']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def emit_figure2(rows: List[Dict[str, Any]], dates: Dict[str, Any]) -> str:
    out = ["model,config,f1,release_date"]
    seen = set()
    for r in rows:
        if not r["config"].startswith("0-shot") or "opt" in r["config"]:
            continue
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        d = dates.get(r["model"], "TODO")
        out.append(f"{r['model']},{r['config']},{fmt(r['computed']['f1'])},{d}")
    return "\n".join(out)


def emit_reconciliation(rows: List[Dict[str, Any]]) -> str:
    out = ["# Reconciliation: recomputed vs reported", "",
           "Rows marked per-sample were recomputed from stored predictions. A "
           "delta flags where the recomputed value diverges from the submitted "
           "PDF (often because the stored file covers a different n than the "
           "reported cell).", "",
           "| Model | Config | Source | n | F1 (comp) | F1 (rep) | dF1 | Notes |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, rep = r["computed"], r.get("reported", {})
        d = (m["f1"] - rep.get("f1", m["f1"])) if m.get("f1") is not None and rep.get("f1") is not None else None
        note = r.get("n_note", "")
        if r.get("dropped") or r.get("corrupt"):
            note = (note + f" dropped={r.get('dropped',0)} corrupt_records={r.get('corrupt',0)}").strip()
        if "TODO" in str(r.get("model_snapshot", "")):
            note = (note + " snapshot TODO(coauthor)").strip()
        out.append(
            f"| {r['model']} | {r['config']} | {r['source']} | {m.get('n','--')} | "
            f"{fmt(m.get('f1'))} | {fmt(rep.get('f1'))} | "
            f"{fmt(d) if d is not None else '--'} | {note} |"
        )
    return "\n".join(out)


def emit_significance(rows: List[Dict[str, Any]]) -> str:
    out = ["# Statistical significance", "",
           "## Per-model 95% confidence intervals", "",
           "| Model | Config | n | metric | CI kind | 95% CI |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("ci"):
            continue
        metric = "F1" if r["ci_kind"].startswith("bootstrap") else "accuracy"
        out.append(f"| {r['model']} | {r['config']} | {r['computed'].get('n','--')} | "
                   f"{metric} | {r['ci_kind']} | {fmt_ci(r)} |")

    # paired McNemar between any two rows that both carry per-sample predictions
    ps = [r for r in rows if "_y_true" in r]
    out += ["", "## Paired McNemar tests (per-sample rows only)", ""]
    if len(ps) < 2:
        out.append("Fewer than two per-sample rows available; no paired test possible.")
        return "\n".join(out)
    out += ["| Model A | Model B | n (aligned) | b | c | exact p |",
            "|---|---|---|---|---|---|"]
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, bb = ps[i], ps[j]
            n = min(len(a["_y_pred"]), len(bb["_y_pred"]))
            # position-aligned only when the two runs used the same ordered sample
            pa, pb = a["_y_pred"][:n], bb["_y_pred"][:n]
            b_cnt = sum(1 for x, y in zip(pa, pb) if x == 1 and y == 0)
            c_cnt = sum(1 for x, y in zip(pa, pb) if x == 0 and y == 1)
            p = mcnemar_exact_p(b_cnt, c_cnt)
            out.append(f"| {a['model']} {a['config']} | {bb['model']} {bb['config']} "
                       f"| {n} | {b_cnt} | {c_cnt} | {p:.4g} |")
    out += ["", "> Note: paired tests assume both runs scored the same ordered "
            "sample. Predictions carry no join key, so alignment is by position; "
            "verify the runs share sample ordering before trusting these p-values."]
    return "\n".join(out)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Build paper tables/figures from the results manifest.")
    ap.add_argument("--manifest", default="results/sample_1000/manifest.json")
    ap.add_argument("--out", default="results/sample_1000/generated")
    args = ap.parse_args()

    manifest = json.loads((REPO / args.manifest).read_text())
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [process_row(r) for r in manifest["table3"]]

    (out_dir / "table3.tex").write_text(emit_table3(rows))
    (out_dir / "table4.tex").write_text(emit_table4(manifest["table4_confusion_test"], rows))
    (out_dir / "figure2_data.csv").write_text(emit_figure2(rows, manifest.get("figure2_release_dates", {})))
    (out_dir / "reconciliation.md").write_text(emit_reconciliation(rows))
    (out_dir / "significance.md").write_text(emit_significance(rows))

    n_ps = sum(1 for r in rows if r["source"].startswith("per-sample"))
    n_todo = sum(1 for r in rows if "TODO" in str(r.get("model_snapshot", "")))
    print(f"Wrote 5 artifacts to {out_dir}")
    print(f"  {len(rows)} Table 3 rows: {n_ps} recomputed from per-sample, "
          f"{len(rows) - n_ps} from reported values")
    print(f"  {n_todo} rows still need a model snapshot from the coauthor")
    print(f"  See {out_dir/'reconciliation.md'} and {out_dir/'significance.md'}")


if __name__ == "__main__":
    main()
