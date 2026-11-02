#!/usr/bin/env python3
"""Run the RegressionV1 rule baseline on a frozen split, position-aligned.

Unlike scripts/baselines/nomenklatura_v1.py this does NOT shuffle or re-split: it
scores every pair in the input file in order and writes one prediction per line, so
pair_index matches the split-file position (the same join the LLM runners use). Uses
the production threshold (default 0.15, the value reported in the paper).

    python eval/run_nomenklatura.py \
        --input data/splits/common_analyst_judged.jsonl \
        --output results/predictions/nomenklatura-regv1__common_analyst_judged.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts" / "vendor" / "nomenklatura"))

from nomenklatura.matching import RegressionV1          # noqa: E402
from nomenklatura.matching.types import ScoringConfig    # noqa: E402
from followthemoney.proxy import EntityProxy             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    config = ScoringConfig.defaults()
    tp = tn = fp = fn = 0
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            truth = p["judgement"]
            try:
                left = EntityProxy.from_dict(p["left"])
                right = EntityProxy.from_dict(p["right"])
                score = RegressionV1.compare(left, right, config).score
                pred = "positive" if score >= args.threshold else "negative"
            except Exception as e:  # keep alignment: record but flag
                score, pred = None, "negative"
                print(f"  warn: pair {i} failed ({type(e).__name__}); scored negative")
            rows.append({"classification": pred, "ground_truth": truth,
                         "pair_index": i, "score": score,
                         "pair_key": p.get("pair_key")})
            if pred == "positive" and truth == "positive":
                tp += 1
            elif pred == "negative" and truth == "negative":
                tn += 1
            elif pred == "positive":
                fp += 1
            else:
                fn += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"{Path(args.input).name}: n={n} thr={args.threshold} "
          f"Acc={acc*100:.2f} F1={f1*100:.2f} P={prec*100:.2f} R={rec*100:.2f} "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
