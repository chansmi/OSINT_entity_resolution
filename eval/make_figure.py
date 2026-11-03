"""Figure 2: F1 on the frozen analyst-judged split, with 95% bootstrap CIs.

Reads results/metrics.csv and writes results/figures/figure_2_fixedsplit.pdf.
Needs matplotlib (`uv sync --extra notebooks`).

Palette: categorical slots 1-2 of the reference palette (blue #2a78d6,
orange #eb6834), validated all-pairs light mode (CVD dE 24.7, normal 33.6).
The rule baseline uses a deliberate neutral -- it is the incumbent reference
rather than a third peer -- with a square marker as secondary encoding.
"""
import csv, matplotlib
from pathlib import Path
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 (proceedings font check)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

CLOSED, OPEN, RULE = "#2a78d6", "#eb6834", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
BAND = "#eda100"

LBL = {"gpt-5.2-pro": "GPT-5.2 Pro", "claude-sonnet-4.5": "Claude Sonnet 4.5",
       "claude-opus-4.5": "Claude Opus 4.5", "gpt-4o": "GPT-4o",
       "deepseek-r1-qwen-14b": "DeepSeek-R1-Qwen-14B", "gpt-5-nano": "GPT-5 Nano",
       "nomenklatura-regv1": "RegressionV1 (rules)", "llama-3.1-8b": "Llama-3.1-8B"}
OPEN_W = {"deepseek-r1-qwen-14b", "llama-3.1-8b"}
TIED = {"gpt-5.2-pro", "claude-sonnet-4.5", "claude-opus-4.5"}

rows = list(csv.DictReader(open(Path(__file__).resolve().parents[1] / "results" / "metrics.csv")))
d = sorted(((r["model"], float(r["f1"]) * 100,
             float(r["f1_lo"]) * 100, float(r["f1_hi"]) * 100) for r in rows),
           key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(7.0, 3.5))
n = len(d)

# statistically-indistinguishable band, drawn first so marks sit above it
lo_b = min(x[2] for x in d if x[0] in TIED)
hi_b = max(x[3] for x in d if x[0] in TIED)
ax.axvspan(lo_b, hi_b, color=BAND, alpha=0.16, lw=0, zorder=0)
ax.text(102, n - 0.34, "top 3 statistically tied", ha="right", va="bottom",
        fontsize=7.5, color=INK2)

for i, (m, f, lo, hi) in enumerate(d):
    if m == "nomenklatura-regv1":
        c, mk, ms = RULE, "s", 6
    elif m in OPEN_W:
        c, mk, ms = OPEN, "^", 7
    else:
        c, mk, ms = CLOSED, "o", 6.5
    ax.plot([72.5, lo], [i, i], color=GRID, lw=0.6, ls=":", zorder=1)  # leader
    ax.errorbar(f, i, xerr=[[f - lo], [hi - f]], fmt=mk, color=c, ecolor=c,
                elinewidth=1.6, capsize=0, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.text(hi + 0.7, i, f"{f:.1f}", va="center", ha="left",
            fontsize=8.5, color=INK)

ax.set_yticks(range(n))
ax.set_yticklabels([LBL[m] for m, _, _, _ in d], fontsize=9, color=INK)
ax.set_xlabel("F1 (%) on the frozen analyst-judged split ($N=3{,}000$), 95% bootstrap CI",
              fontsize=8.5, color=INK2, labelpad=7)
ax.set_xlim(72.5, 102); ax.set_ylim(-0.7, n + 0.05)
ax.set_xticks([75, 80, 85, 90, 95, 100])
ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=3)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=GRID, lw=0.6, ls="-", alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)

handles = [plt.Line2D([], [], color=CLOSED, marker="o", ls="", markersize=6.5,
                      markeredgecolor="white", markeredgewidth=0.8, label="Closed-weight LLM"),
           plt.Line2D([], [], color=OPEN, marker="^", ls="", markersize=7,
                      markeredgecolor="white", markeredgewidth=0.8, label="Open-weight LLM"),
           plt.Line2D([], [], color=RULE, marker="s", ls="", markersize=6,
                      markeredgecolor="white", markeredgewidth=0.8, label="Production rule baseline")]
leg = ax.legend(handles=handles, fontsize=8, loc="upper left",
                bbox_to_anchor=(0.005, 0.99), frameon=False,
                handletextpad=0.5, labelspacing=0.45)
for t in leg.get_texts():
    t.set_color(INK2)

fig.tight_layout()
OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "figure_2_fixedsplit.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print("ok")
