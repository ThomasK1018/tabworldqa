#!/usr/bin/env python3
"""
plot_attention_mse.py
---------------------
Standalone script to reproduce plot (c): Attention MSE on Tabular Actions.

Dataset:
    metrics_results.csv  (same directory as this script)

Output:
    attention_mse.pdf    (same directory as this script)

Usage:
    python plot_attention_mse.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_results.csv")


def draw_mse_panel(ax):
    results_df = pd.read_csv(METRICS_CSV)

    mod_suffixes = ["text_markdown", "text_json", "text_latex", "image"]
    mod_labels   = {
        "text_markdown": "Markdown",
        "text_json":     "JSON",
        "text_latex":    "LaTeX",
        "image":         "Image",
    }
    bot_styles = {
        "text_markdown": {"color": "#2E86C1", "hatch": "//", "edgecolor": "#1a5276"},
        "text_json":     {"color": "#5B9BD5", "hatch": "//", "edgecolor": "#1a5276"},
        "text_latex":    {"color": "#1A5276", "hatch": "//", "edgecolor": "#0d2233"},
        "image":         {"color": "#E67E22", "hatch": "",   "edgecolor": "#9a4500"},
    }
    cols   = [f"bmse_{s}" for s in mod_suffixes]
    labels = [mod_labels[s] for s in mod_suffixes]

    # Explode multi-action rows so each (action, modality) pair gets its own record
    action_rows = []
    for _, r in results_df.iterrows():
        for act in [a.strip() for a in str(r["required_actions"]).split(",")]:
            for col, lbl in zip(cols, labels):
                action_rows.append({"action": act, "modality": lbl, "val": r[col]})
    adf = pd.DataFrame(action_rows)

    actions_raw     = ["lookup_row", "comparison_across_rows", "max_over_column",
                       "sum_over_column", "ratio_within_column",
                       "difference_within_column", "average_over_column"]
    actions_display = ["Row Lookup", "Row Compare", "Col Max",
                       "Col Sum", "Col Ratio", "Col Diff", "Col Avg"]

    x     = np.arange(len(actions_raw))
    width = 0.17
    n_mod = len(mod_suffixes)

    for i, (suf, lbl) in enumerate(zip(mod_suffixes, labels)):
        sub   = adf[adf["modality"] == lbl]
        means = [sub[sub["action"] == a]["val"].dropna().mean() for a in actions_raw]
        st    = bot_styles[suf]
        ax.bar(x + i * width, means, width, label=lbl,
               color=st["color"], hatch=st["hatch"], edgecolor=st["edgecolor"],
               linewidth=0.6, alpha=0.88)

    x_left  = x[0] - 0.5
    x_right = x[-1] + n_mod * width + 0.5
    divider = (x[1] + n_mod * width + x[2]) / 2

    ax.set_xlim(x_left, x_right)
    ax.axvspan(x_left,  divider, color="#FADBD8", alpha=0.45, zorder=0)
    ax.axvspan(divider, x_right, color="#D6EAF8", alpha=0.45, zorder=0)
    ax.axvline(divider, color="#1B4F72", linewidth=1.2, linestyle=":", alpha=0.7)

    y_top = ax.get_ylim()[1]
    ax.text(x_left  + 0.1, y_top * 0.97, "Row Ops",    fontsize=12, color="#943126", va="top")
    ax.text(divider + 0.1, y_top * 0.97, "Column Ops", fontsize=12, color="#1B4F72", va="top")

    ax.set_xticks(x + width * (n_mod - 1) / 2)
    ax.set_xticklabels(actions_display, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Mean Balanced MSE", fontsize=12)
    ax.set_title("(c) Attention MSE on Tabular Actions",
                 fontsize=20, fontweight="bold", pad=8)
    ax.legend(fontsize=12, ncol=4, loc="lower center")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.18)
    draw_mse_panel(ax)

    out = os.path.join(RESULTS_DIR, "attention_mse.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight",
                metadata={"Creator": "Matplotlib"})
    print(f"[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
