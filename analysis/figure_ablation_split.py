#!/usr/bin/env python3
"""
figure_ablation_split.py

Saves each panel of figure_combined_main_v4 as a standalone full-width PDF:
  figures/figure_main_ablation_a.pdf  — Modality Preference heatmap
  figures/figure_main_ablation_b.pdf  — Expected and Realized Reward bar chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR    = os.path.dirname(SCRIPTS_DIR)
FIGURES_DIR = os.path.join(REPO_DIR, "COLM-Tabular-World-Model", "figures")
PERC_DIR    = os.path.join(REPO_DIR, "perception_importance", "text_case_study")
STEP_CSV    = os.path.join(PERC_DIR, "modality_output_v2", "step_level.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# (a)  HEATMAP — Modality Preference on Tabular Actions
# ═══════════════════════════════════════════════════════════════════════════════
MODALITY_COLORS = {
    "image":      "#1f77b4",
    "text":       "#2ca02c",
    "multimodal": "#d62728",
}
MODALITY_LABELS_C1 = {
    "image":      "Image",
    "text":       "Text (Markdown)",
    "multimodal": "Multimodal",
}


def draw_heatmap_panel(ax, step_df):
    step_df = step_df.copy()
    step_df["tool_short"] = step_df["tool_name"].str.replace(r"^f_", "", regex=True)

    MODAL_ORDER = ["image", "text", "multimodal"]
    all_m  = step_df["modality"].fillna("text").unique().tolist()
    modals = [m for m in MODAL_ORDER if m in all_m]
    modals += [m for m in all_m if m not in MODAL_ORDER and m != "unknown"]

    step_known = step_df[step_df["tool_name"] != "unknown"].copy()
    step_known["modality"] = step_known["modality"].fillna("text")

    tool_counts = step_known["tool_short"].value_counts()
    tools = tool_counts.index.tolist()

    pivot_all = (
        step_known.groupby(["tool_short", "modality"])
        .size().unstack(fill_value=0)
        .reindex(index=tools, fill_value=0)
    )
    row_totals = pivot_all.sum(axis=1)
    pivot      = pivot_all.reindex(columns=modals, fill_value=0)
    pivot_frac = pivot.div(row_totals.replace(0, 1), axis=0)

    cmap = plt.cm.Blues
    im = ax.imshow(pivot_frac.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(modals)))
    ax.set_xticklabels([MODALITY_LABELS_C1.get(m, m) for m in modals],
                       fontsize=12, fontweight="bold")
    ax.xaxis.set_ticks_position("bottom")

    ylabels = [f"{t.replace('_', ' ')}\n(n={int(row_totals[t])})" for t in tools]
    ax.set_yticks(range(len(tools)), labels=ylabels, fontsize=12)
    for lbl in ax.get_yticklabels():
        if "compute" in lbl.get_text():
            lbl.set_fontweight("bold")
    ax.set_title("Modality Preference on Tabular Actions",
                 fontweight="bold", fontsize=18, pad=8)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.ax.tick_params(labelsize=12)

    for ri in range(len(tools)):
        for ci in range(len(modals)):
            v   = pivot_frac.values[ri, ci]
            cnt = int(pivot.values[ri, ci])
            txt_color = "white" if v > 0.55 else "#222"
            if cnt > 0:
                ax.text(ci, ri - 0.12, f"{v:.0%}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=txt_color)
                ax.text(ci, ri + 0.22, f"n={cnt}",
                        ha="center", va="center", fontsize=9,
                        color=txt_color, alpha=0.85)
            else:
                ax.text(ci, ri - 0.12, "0%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="#222")
                ax.text(ci, ri + 0.22, "n=0",
                        ha="center", va="center", fontsize=9,
                        color="#222", alpha=0.85)

    ax.set_xticks(np.arange(-0.5, len(modals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tools),  1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


# ═══════════════════════════════════════════════════════════════════════════════
# (b)  REWARD PANEL — Expected and Realized Reward in Compressed Actions
# ═══════════════════════════════════════════════════════════════════════════════
_TOOL_RISK = {
    "f_select_row": "filter", "f_select_col": "filter",
    "f_select_column": "filter", "f_filter_val": "filter",
    "f_sort": "transform", "f_sort_by": "transform",
    "f_aggregate": "transform", "f_group_by": "transform",
    "f_join": "transform", "f_compute_column": "transform",
    "f_final_answer": "answer",
}

BUCKET_COLORS = {
    "single-filter":    "#4dac26",
    "merged-filter":    "#f1a340",
    "merged-transform": "#d7191c",
    "merged+answer":    "#998ec3",
    "answer-only":      "#92c5de",
}

_label_map = {
    "single-filter":    "single\nfilter",
    "merged-filter":    "merged\nfilter",
    "merged-transform": "merged\ntransform",
    "merged+answer":    "merged\n+ answer",
    "answer-only":      "answer\nonly",
}


def _complexity_bucket(n_actions, tool_names):
    has_transform = any(_TOOL_RISK.get(t, "other") == "transform" for t in tool_names)
    has_answer    = any("final_answer" in t for t in tool_names)
    if has_answer and n_actions == 1:
        return "answer-only"
    if has_answer and n_actions > 1:
        return "merged+answer"
    if has_transform and n_actions > 1:
        return "merged-transform"
    if n_actions > 1:
        return "merged-filter"
    return "single-filter"


def rebuild_buckets(step_df):
    step_level = step_df.drop_duplicates(subset=["question", "step_num"]).copy()
    tool_per_step = (
        step_df.groupby(["question", "step_num"])["tool_name"]
        .apply(list).reset_index(name="all_tools")
    )
    step_level = step_level.merge(tool_per_step, on=["question", "step_num"], how="left")
    step_level["bucket"] = step_level.apply(
        lambda r: _complexity_bucket(
            r["n_actions"],
            r["all_tools"] if isinstance(r["all_tools"], list) else []
        ), axis=1
    )
    return step_level


def draw_reward_panel(ax, step_df):
    rng = np.random.default_rng(42)
    step_level = rebuild_buckets(step_df)

    bucket_order    = ["single-filter", "merged-filter", "merged-transform",
                       "merged+answer", "answer-only"]
    present_buckets = [b for b in bucket_order if b in step_level["bucket"].unique()]
    hlabels = [_label_map.get(b, b) for b in present_buckets]
    n_b = len(present_buckets)

    REWARD_COLOR = "#4575b4"
    PRED_COLOR   = "#d73027"
    BAR_H = 0.35
    OFFSET = 0.20

    merged_idx = [i for i, b in enumerate(present_buckets) if b.startswith("merged")]
    other_idx  = [i for i, b in enumerate(present_buckets) if not b.startswith("merged")]

    for i in merged_idx:
        ax.axhspan(i - 0.48, i + 0.48, color="#f1a340", alpha=0.08, zorder=0)
    for i in other_idx:
        ax.axhspan(i - 0.48, i + 0.48, color="#4dac26", alpha=0.07, zorder=0)

    r_means, r_errs, r_vals = [], [], []
    p_means, p_errs, p_vals = [], [], []
    ns = []
    for b in present_buckets:
        rs = step_level[step_level["bucket"] == b]["reward"].dropna()
        ps = step_level[step_level["bucket"] == b]["pred_score"].dropna()
        r_means.append(rs.mean() if len(rs) else np.nan)
        r_errs.append(rs.sem()   if len(rs) > 1 else 0)
        r_vals.append(rs.values)
        p_means.append(ps.mean() if len(ps) else np.nan)
        p_errs.append(ps.sem()   if len(ps) > 1 else 0)
        p_vals.append(ps.values)
        ns.append(len(rs))

    y   = np.arange(n_b)
    y_r = y - OFFSET
    y_p = y + OFFSET

    ax.barh(y_r, r_means, xerr=r_errs, height=BAR_H,
            color=REWARD_COLOR, alpha=0.82, capsize=3,
            edgecolor="white", linewidth=0.5, zorder=2)
    ax.barh(y_p, p_means, xerr=p_errs, height=BAR_H,
            color=PRED_COLOR, alpha=0.82, capsize=3, hatch="///",
            edgecolor="white", linewidth=0.5, zorder=2)

    for i, (rv, pv) in enumerate(zip(r_vals, p_vals)):
        jr = rng.uniform(-BAR_H / 2, BAR_H / 2, size=len(rv))
        jp = rng.uniform(-BAR_H / 2, BAR_H / 2, size=len(pv))
        ax.scatter(rv, y_r[i] + jr, color=REWARD_COLOR, alpha=0.30, s=7, zorder=3, linewidths=0)
        ax.scatter(pv, y_p[i] + jp, color=PRED_COLOR,   alpha=0.30, s=7, zorder=3, linewidths=0)

    for i, (rm, re, pm, pe, n) in enumerate(zip(r_means, r_errs, p_means, p_errs, ns)):
        if not np.isnan(rm):
            ax.text(min(rm + re + 0.02, 1.10), y_r[i],
                    f"{rm:.2f}±{re:.2f}", va="center", fontsize=15, color=REWARD_COLOR)
        if not np.isnan(pm):
            ax.text(min(pm + pe + 0.02, 1.10), y_p[i],
                    f"{pm:.2f}±{pe:.2f}", va="center", fontsize=15, color=PRED_COLOR)

    for i, n in enumerate(ns):
        ax.text(1.14, y[i], f"n={n}", va="center", ha="left", fontsize=18, color="#444")

    overall_r = step_level["reward"].mean()
    overall_p = step_level["pred_score"].mean()
    ax.axvline(overall_r, color=REWARD_COLOR, linewidth=1.1, linestyle="--", alpha=0.7,
               label=f"overall reward = {overall_r:.2f}")
    ax.axvline(overall_p, color=PRED_COLOR,   linewidth=1.1, linestyle=":",  alpha=0.7,
               label=f"overall pred. = {overall_p:.2f}")

    metric_handles = [
        mpatches.Patch(color=REWARD_COLOR, label="Expected", alpha=0.82),
        mpatches.Patch(color=PRED_COLOR,   label="Realized", alpha=0.82, hatch="///"),
    ]
    ax.legend(handles=metric_handles, fontsize=12, loc="lower right",
              framealpha=0.92, edgecolor="#ccc", bbox_to_anchor=(1, -0.015), ncol=2)

    ax.set_xlim(0, 1.22)
    ax.set_yticks(y, labels=hlabels, fontsize=18)
    ax.set_xlabel("Mean Score (± SD)", fontsize=18)
    ax.set_title("Expected and Realized Reward in Compressed Actions",
                 fontsize=18, fontweight="bold", pad=6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    step_df = pd.read_csv(STEP_CSV)
    step_df.columns = [c.strip() for c in step_df.columns]
    print(f"[load] {len(step_df)} step-tool records from {STEP_CSV}")

    # ── Panel (a): heatmap ────────────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(7.5, 5.5))
    draw_heatmap_panel(ax_a, step_df)
    out_a = os.path.join(FIGURES_DIR, "figure_main_ablation_a.pdf")
    fig_a.savefig(out_a, format="pdf", bbox_inches="tight",
                  metadata={"Creator": "Matplotlib"})
    plt.close(fig_a)
    print(f"[saved] {out_a}")

    # ── Panel (b): reward bar chart ───────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(7.5, 5.0))
    draw_reward_panel(ax_b, step_df)
    out_b = os.path.join(FIGURES_DIR, "figure_main_ablation_b.pdf")
    fig_b.savefig(out_b, format="pdf", bbox_inches="tight",
                  metadata={"Creator": "Matplotlib"})
    plt.close(fig_b)
    print(f"[saved] {out_b}")


if __name__ == "__main__":
    main()
