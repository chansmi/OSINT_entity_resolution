#!/usr/bin/env python3
"""Significance tests and confidence intervals for OpenSanctions Pairs.

Pure numpy/stdlib (no scipy) so results are reproducible in a clean-room environment.

Provides:
  - metrics(pred, truth): accuracy, precision, recall, F1 (positive class).
  - bootstrap_ci(...): percentile bootstrap CI for any metric (default F1), seeded.
  - mcnemar(correct_a, correct_b): exact two-sided binomial McNemar test on the
    discordant pairs (paired, same examples), returns (b, c, p_value).
  - wilson(k, n): Wilson score interval for a proportion.

CLI aggregates a set of per-sample prediction JSONL files (each with fields
classification, ground_truth, pair_index) into results/metrics.csv (metrics + 95% CIs)
and results/significance.csv (McNemar of every model vs the best-F1 model, over the
pairs they share by pair_index).

    python eval/significance.py \
        --pred gpt-5-nano=results/sample_1000/llm_responses_binary_gpt-5-nano.jsonl \
        --pred gpt-5.2-pro=results/sample_1000/llm_responses_binary_gpt-5_2-pro.jsonl \
        --out results
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

POS, NEG = 1, 0


def load_preds(path):
    """Return dict: pair_index -> (pred, truth) as 0/1."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pred = POS if str(r.get("classification", "")).lower().startswith("pos") else NEG
            truth = POS if str(r.get("ground_truth", "")).lower().startswith("pos") else NEG
            idx = r.get("pair_index", i)
            out[idx] = (pred, truth)
    return out


def metrics(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    tp = int(np.sum((pred == POS) & (truth == POS)))
    tn = int(np.sum((pred == NEG) & (truth == NEG)))
    fp = int(np.sum((pred == POS) & (truth == NEG)))
    fn = int(np.sum((pred == NEG) & (truth == POS)))
    acc = (tp + tn) / max(1, len(pred))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"n": len(pred), "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _f1_from_idx(pred, truth, idx):
    p, t = pred[idx], truth[idx]
    tp = np.sum((p == POS) & (t == POS))
    fp = np.sum((p == POS) & (t == NEG))
    fn = np.sum((p == NEG) & (t == POS))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def bootstrap_ci(pred, truth, metric="f1", B=10000, seed=42, alpha=0.05):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    n = len(pred)
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        if metric == "f1":
            stats[b] = _f1_from_idx(pred, truth, idx)
        elif metric == "acc":
            stats[b] = np.mean(pred[idx] == truth[idx])
        else:
            raise ValueError(metric)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def mcnemar(correct_a, correct_b):
    """Exact two-sided binomial McNemar test on discordant pairs (paired samples)."""
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n <= 40:
        # exact two-sided binomial p at theta=0.5 (safe from float overflow for small n)
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        return b, c, min(1.0, 2 * tail)
    # McNemar normal approximation with continuity correction (two-sided) for large n
    z = max(0.0, abs(b - c) - 1) / math.sqrt(n)
    return b, c, min(1.0, math.erfc(z / math.sqrt(2)))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], help="name=path")
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        models[name] = load_preds(path)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, d in models.items():
        idxs = sorted(d)
        pred = [d[i][0] for i in idxs]
        truth = [d[i][1] for i in idxs]
        m = metrics(pred, truth)
        f1lo, f1hi = bootstrap_ci(pred, truth, "f1", args.boot, args.seed)
        alo, ahi = bootstrap_ci(pred, truth, "acc", args.boot, args.seed)
        m.update({"model": name, "f1_lo": f1lo, "f1_hi": f1hi, "acc_lo": alo, "acc_hi": ahi})
        rows.append(m)
        print(f"{name:22s} n={m['n']:5d}  F1={m['f1']*100:5.2f} [{f1lo*100:5.2f},{f1hi*100:5.2f}]"
              f"  Acc={m['acc']*100:5.2f} [{alo*100:5.2f},{ahi*100:5.2f}]  P={m['prec']*100:5.2f} R={m['rec']*100:5.2f}")

    with open(outdir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n", "acc", "acc_lo", "acc_hi",
                                          "f1", "f1_lo", "f1_hi", "prec", "rec",
                                          "tp", "tn", "fp", "fn"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # McNemar vs best-F1 model, over shared pair_index
    if len(models) > 1:
        best = max(rows, key=lambda r: r["f1"])["model"]
        bd = models[best]
        sig = []
        for name, d in models.items():
            if name == best:
                continue
            shared = sorted(set(bd) & set(d))
            if not shared:
                continue
            ca = [bd[i][0] == bd[i][1] for i in shared]
            cb = [d[i][0] == d[i][1] for i in shared]
            b, c, p = mcnemar(ca, cb)
            sig.append({"best": best, "vs": name, "n_shared": len(shared),
                        "b_best_right": b, "c_vs_right": c, "mcnemar_p": p,
                        "significant_0.05": p < 0.05})
            print(f"McNemar {best} vs {name}: n={len(shared)} b={b} c={c} p={p:.4g}"
                  f" {'(sig)' if p < 0.05 else '(n.s.)'}")
        if sig:
            with open(outdir / "significance.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(sig[0].keys()))
                w.writeheader()
                w.writerows(sig)


if __name__ == "__main__":
    main()
