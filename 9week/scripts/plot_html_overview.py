"""Produce the opening comparison from original development and final run tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

root = Path(__file__).resolve().parents[1]
for family in ["Malgun Gothic", "Noto Sans CJK KR", "AppleGothic"]:
    if family in font_manager.get_font_names():
        plt.rcParams["font.family"] = family
        break
test = pd.read_csv(root / "final/test_runs.csv").groupby("model").macro_f1.mean()
fig, ax = plt.subplots(figsize=(8.4, 3.1), layout="constrained")
for y, model in enumerate(["RF", "A0", "A2"]):
    runs = pd.read_csv(root / f"runs/{model}_runs.csv")
    dev = runs.groupby("fold").macro_f1.mean().mean()
    held = test[model]
    ax.plot([held, dev], [y, y], color="#92a6b4", linewidth=2.4, zorder=2)
    ax.scatter(dev, y, s=77, color="#194b78", marker="o", zorder=3,
               label="개발 검증" if y == 0 else None)
    ax.scatter(held, y, s=84, color="#a13c24", marker="D", zorder=3,
               label="보류 테스트" if y == 0 else None)
    for val, offset in [(dev, 13), (held, -16)]:
        ax.annotate(f"{val:.4f}", (val, y), xytext=(0, offset),
                    textcoords="offset points", ha="center", color="#202e38", fontsize=10)
ax.set(yticks=[0, 1, 2], yticklabels=["RF", "A0", "A2"], ylim=(-.53, 2.55),
       xlim=(.735, .925), xticks=[.75, .80, .85, .90], xlabel="Macro F1")
ax.spines[:].set_visible(False)
ax.tick_params(length=0, labelsize=11)
ax.grid(axis="x", color="#d9e1e7")
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=2, frameon=False)
fig.savefig(root / "reports/figures/html_development_holdout.png", dpi=170)
