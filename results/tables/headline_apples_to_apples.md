### Apples-to-apples: all models on the identical analyst-judged split (N=3000)

F1 with 95% bootstrap CI. `sig?` = McNemar vs the best model (accuracy, paired); n.s. means the gap is within sampling noise. `overall F1` = same model on the auto-merge-included mirror, showing the inflation.

| Model | F1 (95% CI) | Acc | P | R | vs best (McNemar p) | overall F1 (w/ auto-merge) |
|---|---|---|---|---|---|---|
| gpt-5.2-pro | 98.05 [97.58, 98.47] | 97.43 | 97.08 | 99.03 | — (best) | 98.70 |
| claude-sonnet-4.5 | 97.83 [97.36, 98.29] | 97.20 | 98.59 | 97.08 | p=0.505 (n.s.) | 98.57 |
| claude-opus-4.5 | 97.75 [97.26, 98.22] | 97.10 | 98.64 | 96.87 | p=0.332 (n.s.) | 98.52 |
| gpt-4o | 97.51 [96.97, 97.98] | 96.73 | 96.72 | 98.31 | p=0.0301 (sig) | 98.35 |
| deepseek-r1-qwen-14b | 94.27 [93.50, 95.00] | 92.23 | 90.59 | 98.26 | p=4.77e-29 (sig) | 96.16 |
| gpt-5-nano | 89.57 [88.60, 90.48] | 85.20 | 82.65 | 97.74 | p=1.91e-78 (sig) | 92.90 |
| nomenklatura-regv1 | 85.80 [84.76, 86.89] | 78.50 | 75.16 | 99.95 | p=5.24e-117 (sig) | 90.14 |
| llama-3.1-8b | 76.53 [75.01, 78.08] | 72.20 | 84.79 | 69.74 | p=1.12e-152 (sig) | 84.80 |
