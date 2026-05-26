#!/usr/bin/env python3
"""
Attention analysis pipeline — WTQ table nt-2317, 10 questions.

For each question:
  • Text attention  (markdown / json / latex): generate_with_attention →
    last-layer sparse Q·K → _map_text_attention_to_cells
  • Visual attention (image):  10 stochastic runs, layer-10 Q·K from key-noun
    query tokens → averaged patch grid → cell scores via bbox mapping

Outputs per question (results/q{i}/):
  text_panels.{png,pdf}       — 3-panel cell-attention heatmap
  visual_attn_solo.{png,pdf}  — single-panel averaged visual attention overlay
  table_render.jpg            — rendered full-table image used by the model

Aggregate outputs (results/):
  mse_results.csv             — MSE per question × modality
  bmse_heatmap.{png,pdf}, ap_heatmap.{png,pdf}
  bmse_by_action.{png,pdf}, ap_by_action.{png,pdf}
  bmse_overall.{png,pdf}, ap_overall.{png,pdf}
"""

import os, sys, math, json, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom as nd_zoom
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from sklearn.metrics import average_precision_score

# ── CLI args ──────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--question-ids",      default="",  help="Comma-separated IDs to run, e.g. 0,1,2 (default: all)")
_p.add_argument("--max-display-rows",  type=int, default=0, help="Truncate table to N rows for rendering/text (0=full table)")
_p.add_argument("--n-runs",            type=int, default=10, help="Stochastic visual-attention runs (default 10)")
_p.add_argument("--run-tag",           default="",  help="Output subdir tag, e.g. 'illustrative' or 'full'")
args = _p.parse_args()

# ── Path setup ───────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
TABWORLDQA   = os.path.dirname(HERE)                # tabworldqa/
VAGEN_ROOT   = os.path.dirname(TABWORLDQA)           # VAGEN/
sys.path.insert(0, TABWORLDQA)
sys.path.insert(0, VAGEN_ROOT)
sys.path.insert(0, os.path.join(VAGEN_ROOT, "scripts"))

from visual_table_reasoning_visualize_hf import (
    render_dataframe_to_image,
    build_image_message,
    build_serialized_table_message,
    serialize_dataframe_to_text,
    _map_text_attention_to_cells,
    _map_image_attention_to_cells,
)
from vagen.inference.model_interface.hf_qwen_vl import (
    HFQwenVLModelInterface,
    HFQwenVLModelConfig,
)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-VL-8B-Instruct-FP8"
DEVICE       = "cuda:0"
DATA_DIR     = os.path.join(TABWORLDQA, "data", "wikitq_nt2317")
_tag         = f"_{args.run_tag}" if args.run_tag else ""
OUT_DIR      = os.path.join(TABWORLDQA, "results" + _tag)
QUESTION_CSV = os.path.join(DATA_DIR, "wikitq_nt2317_question.csv")
TABLE_XLSX   = os.path.join(DATA_DIR, "wikitq_nt2317_tables.xlsx")
TARGET_LAYER = 10        # best visual-attention layer from sweep
N_RUNS       = args.n_runs
DISPLAY_ROWS = args.max_display_rows if args.max_display_rows > 0 else None  # None = full table
QUESTION_IDS = [int(x) for x in args.question_ids.split(",") if x.strip()] \
               if args.question_ids else list(range(10))
TEMPERATURE  = 0.8
MAX_GEN_TOKENS = 256

TEXT_MODES  = ["text_markdown", "text_json", "text_latex"]
TEXT_CMAPS  = {"text_markdown": "YlGn",  "text_json": "YlOrRd", "text_latex": "BuPu"}
TEXT_LABELS = {"text_markdown": "Markdown", "text_json": "JSON",  "text_latex": "LaTeX"}
TEXT_FMTS   = {"text_markdown": "markdown", "text_json": "json",  "text_latex": "latex"}

# Key noun/filter word(s) per question for the visual attention query strategy.
# These are the tokens whose Q vectors are averaged to compute attention to image patches.
# For questions with no specific row-filter noun, we fall back to all question tokens.
KEY_NOUNS: dict[int, list[str]] = {
    0: ["algeria"],
    1: ["algeria"],
    2: ["brunei"],
    3: ["malaysia"],
    4: ["qatar"],
    5: ["capacity"],
    6: ["country"],
    7: ["egypt"],
    8: ["malaysian", "egyptian"],
    9: ["capacity"],
    # Q10-Q19 (new questions)
    10: ["algeria"],
    11: ["algeria"],
    12: ["angola", "brunei"],
    13: ["indonesia", "yemen"],
    14: [],                          # no specific filter — fallback to all question tokens
    15: ["algeria", "indonesia"],
    16: [],                          # no specific filter
    17: ["brunei", "nigeria"],
    18: ["brunei", "nigeria"],
    19: [],                          # comparison across all countries
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
questions_df = pd.read_csv(QUESTION_CSV, encoding="latin-1")
table_df     = pd.read_excel(TABLE_XLSX, sheet_name="original_table")
n_rows, n_cols = len(table_df), len(table_df.columns)
col_names = list(table_df.columns)
print(f"Table: {n_rows} rows × {n_cols} cols  |  columns: {col_names}")

# GT matrices: (n_rows, n_cols) binary float arrays.
# A sheet is unusable if it contains non-numeric content (e.g. the original table
# was accidentally pasted in instead of the annotation matrix).
gt_matrices: dict[int, np.ndarray] = {}
bad_gt_sheets: list[int] = []
for qi in range(20):
    gt = pd.read_excel(TABLE_XLSX, sheet_name=f"q{qi}")
    try:
        arr = gt.values.astype(float)
        assert arr.shape == (n_rows, n_cols), (
            f"shape mismatch: expected ({n_rows},{n_cols}), got {arr.shape}"
        )
        gt_matrices[qi] = arr
    except (ValueError, AssertionError) as e:
        print(f"WARNING: GT sheet q{qi} is not a valid binary matrix ({e}). "
              f"MSE will be skipped for Q{qi}.")
        bad_gt_sheets.append(qi)
        gt_matrices[qi] = None   # placeholder; MSE will be NaN for this question

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model …")
cfg = HFQwenVLModelConfig(
    model_name=MODEL_NAME, device=DEVICE, torch_dtype="bfloat16",
    temperature=TEMPERATURE, max_tokens=MAX_GEN_TOKENS,
)
iface        = HFQwenVLModelInterface(cfg)
model        = iface.model
proc         = iface.processor
target_attn  = model.model.language_model.layers[TARGET_LAYER].self_attn
print("Model ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    """Return (balanced_mse, average_precision) for a pred/gt cell-score pair.

    balanced_mse = average of pos_MSE (pred[gt==1] vs 1) and neg_MSE (pred[gt==0] vs 0).
                   Gives equal weight to positive and negative classes regardless of size.
    average_precision = area under precision-recall curve (sklearn convention).
                        Random baseline ≈ prevalence (n_pos / n_total).
    """
    pred_f = pred.ravel().astype(float)
    gt_f   = gt.ravel().astype(float)
    pos    = gt_f == 1
    neg    = ~pos
    pos_mse = float(np.mean((pred_f[pos] - 1.0) ** 2)) if pos.any() else float("nan")
    neg_mse = float(np.mean((pred_f[neg] - 0.0) ** 2)) if neg.any() else float("nan")
    bmse    = (pos_mse + neg_mse) / 2 if not (math.isnan(pos_mse) or math.isnan(neg_mse)) \
              else float("nan")
    ap      = float(average_precision_score(gt_f.astype(int), pred_f)) \
              if (pos.any() and not pos.all()) else float("nan")
    return bmse, ap


def find_keyword_positions(input_ids: list[int], keyword: str) -> list[int]:
    """Return the token positions (in input_ids) whose concatenated decoded text
    contains `keyword` (case-insensitive, window up to 8 consecutive tokens)."""
    decoded = [proc.tokenizer.decode([t]) for t in input_ids]
    kw = keyword.lower()
    for window in range(1, 9):
        for i in range(len(decoded) - window + 1):
            if kw in "".join(decoded[i : i + window]).lower().strip():
                return list(range(i, i + window))
    return []


def capture_layer_attn(
    inputs: dict,
    q_positions: list[int],
    img_pos_t: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> np.ndarray | None:
    """One no_grad forward pass capturing Q/K from TARGET_LAYER.
    Returns normalised (grid_h, grid_w) attention grid, or None on failure."""
    captured: dict = {}
    _attn_key = getattr(
        getattr(model.model.language_model.layers[TARGET_LAYER].self_attn, "config", None),
        "_attn_implementation", "sdpa",
    )
    _orig_fn = ALL_ATTENTION_FUNCTIONS[_attn_key]

    def _capturing_fn(module, query, key, value, *args, **kwargs):
        if module is target_attn:
            captured["Q"] = query.detach().cpu().float()
            captured["K"] = key.detach().cpu().float()
        return _orig_fn(module, query, key, value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS[_attn_key] = _capturing_fn
    try:
        torch.cuda.empty_cache()
        with torch.no_grad():
            model(**inputs, return_dict=True)
    finally:
        del ALL_ATTENTION_FUNCTIONS[_attn_key]
        torch.cuda.empty_cache()

    if "Q" not in captured:
        return None

    Q_all = captured["Q"][0]   # (Hq, seq, d)
    K_all = captured["K"][0]   # (Hkv, seq, d)
    Hq, Hkv, d_k = Q_all.shape[0], K_all.shape[0], Q_all.shape[2]

    valid = [p for p in q_positions if p < Q_all.shape[1]]
    if not valid:
        return None

    Q     = Q_all[:, valid, :].mean(dim=1)     # (Hq, d)
    K_img = K_all[:, img_pos_t, :]             # (Hkv, n_img, d)
    if Hkv < Hq:
        K_img = K_img.repeat_interleave(Hq // Hkv, dim=0)

    scores = torch.einsum("hd,hid->hi", Q, K_img) / (d_k ** 0.5)
    attn   = torch.softmax(scores, dim=-1).mean(0).numpy()
    grid   = attn.reshape(grid_h, grid_w)
    mx = grid.max()
    if mx > 0:
        grid = grid / mx
    return grid


def grid_to_cell_scores(
    avg_grid: np.ndarray,
    cell_bboxes: dict,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Map averaged patch-attention grid → normalised (n_rows, n_cols) cell-score array."""
    grid_h, grid_w = avg_grid.shape
    scores = np.zeros((n_rows, n_cols), dtype=float)
    ph_px = img_h / grid_h
    pw_px = img_w / grid_w
    for r in range(n_rows):
        for c in range(n_cols):
            bbox = cell_bboxes.get((r, c))
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            ph0 = max(0, int(y0 / ph_px))
            ph1 = min(grid_h, int(math.ceil(y1 / ph_px)))
            pw0 = max(0, int(x0 / pw_px))
            pw1 = min(grid_w, int(math.ceil(x1 / pw_px)))
            if ph1 > ph0 and pw1 > pw0:
                scores[r, c] = float(avg_grid[ph0:ph1, pw0:pw1].mean())
    mx = scores.max()
    if mx > 0:
        scores /= mx
    return scores


def plot_text_panels(q_id, question, text_scores, display_df, out_dir):
    """3-panel cell-attention heatmap: Markdown | JSON | LaTeX."""
    d_rows = len(display_df)
    d_cols = len(display_df.columns)
    d_col_names = list(display_df.columns)
    fig_h = max(6, d_rows * 0.30 + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, fig_h))
    row_keys = [str(i) for i in range(d_rows)]

    for ax, mode in zip(axes, TEXT_MODES):
        matrix = text_scores[mode]
        im = ax.imshow(matrix, cmap=TEXT_CMAPS[mode], vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(d_cols))
        ax.set_xticklabels(d_col_names, rotation=30, ha="right", fontsize=6)
        ax.set_yticks(range(d_rows))
        ax.set_yticklabels(row_keys, fontsize=5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for r in range(d_rows):
            for c in range(d_cols):
                score = matrix[r, c]
                try:
                    raw = display_df.iloc[r, c]
                    val = "" if (raw is None or (isinstance(raw, float) and math.isnan(raw))) \
                          else str(raw)[:8]
                except Exception:
                    val = ""
                color = "white" if score > 0.6 else "black"
                ax.text(c, r, f"{val}\n{score:.2f}",
                        ha="center", va="center", fontsize=3.5, color=color)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(TEXT_LABELS[mode], fontsize=10, pad=5)

    q_short = question[:100] + ("…" if len(question) > 100 else "")
    fig.suptitle(f"Q{q_id}: \"{q_short}\"", fontsize=9, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "text_panels.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "text_panels.pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_visual_solo(q_id, question, avg_grid, img_arr, img_h, img_w, n_valid, out_dir):
    """Single-panel averaged visual attention overlay (Reds, no row markers)."""
    grid_h, grid_w = avg_grid.shape
    zh = img_h / grid_h
    zw = img_w / grid_w
    vmin = float(np.percentile(avg_grid, 20))
    vmax = float(np.percentile(avg_grid, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    up   = nd_zoom(avg_grid, (zh, zw), order=1)
    up   = np.clip(up, 0, 1)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap("Reds")(norm(up))
    alpha = 0.6
    base  = img_arr.astype(float) / 255.0
    comp  = base * (1 - alpha * rgba[:, :, 3:]) + rgba[:, :, :3] * alpha * rgba[:, :, 3:]
    comp  = np.clip(comp, 0, 1)

    aspect = img_h / img_w
    fig_w, fig_h_fig = 6.0, max(5.0, 6.0 * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h_fig))
    ax.imshow(comp, aspect="auto")
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    ax.axis("off")
    q_short = question[:70] + ("…" if len(question) > 70 else "")
    ax.set_title(
        f"Q{q_id} — Visual attention  (n={n_valid} runs)\n"
        f"Layer {TARGET_LAYER}  ·  key-noun query\n\"{q_short}\"",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "visual_attn_solo.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "visual_attn_solo.pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)


# ── Per-question loop ─────────────────────────────────────────────────────────
all_results = []

for _, qrow in questions_df.iterrows():
    q_id     = int(qrow["question_id"])
    if q_id not in QUESTION_IDS:
        continue
    question = str(qrow["question"])
    req_acts = str(qrow["required_actions"])
    key_nouns = KEY_NOUNS[q_id]
    gt_matrix = gt_matrices[q_id]          # (n_rows, n_cols) binary, or None if sheet is bad

    # Truncated display dataframe (used for rendering, text serialisation, plotting)
    display_df   = table_df.head(DISPLAY_ROWS) if DISPLAY_ROWS else table_df
    d_rows       = len(display_df)
    d_cols       = len(display_df.columns)
    # GT truncated to displayed rows
    gt_used = gt_matrix[:d_rows, :] if gt_matrix is not None else None

    print(f"\n{'='*65}")
    print(f"Q{q_id}: {question}")
    print(f"  display_rows={d_rows}  required_actions: {req_acts}  |  key_nouns: {key_nouns}")

    q_out = os.path.join(OUT_DIR, f"q{q_id}")
    os.makedirs(q_out, exist_ok=True)

    tool_prompt = (
        "You are a table reasoning agent.\n"
        f"Question: {question}\n"
        "Answer the question based on the table."
    )

    # ── 1. Text attention ─────────────────────────────────────────────────────
    text_cell_scores: dict[str, np.ndarray] = {}

    for mode in TEXT_MODES:
        fmt        = TEXT_FMTS[mode]
        serialized = serialize_dataframe_to_text(
            display_df, text_table_format=fmt, max_rows=None, max_cols=None
        )
        message    = build_serialized_table_message(serialized, tool_prompt, text_table_format=fmt)[0]
        result     = iface.generate_with_attention([message])
        attn_data  = result.get("attention_data")

        if attn_data is not None:
            scores_df = _map_text_attention_to_cells(display_df, attn_data, proc)
            text_cell_scores[mode] = scores_df.values.astype(float)
        else:
            print(f"  WARNING: no attention data for {mode}")
            text_cell_scores[mode] = np.zeros((d_rows, d_cols))

        print(f"  {mode}: max={text_cell_scores[mode].max():.4f}")

    # ── 2. Visual attention ───────────────────────────────────────────────────
    img_path    = os.path.join(q_out, "table_render.jpg")
    cell_bboxes = render_dataframe_to_image(
        display_df, img_path,
        max_rows=d_rows, max_cols=d_cols,
        return_cell_bboxes=True,
    )
    img      = Image.open(img_path)
    img_w, img_h = img.size
    img_arr  = np.array(img)

    # Build inputs for the attention capture forward pass
    base_messages = build_image_message(img_path, tool_prompt)
    _chat_fmt     = iface._to_qwen_chat_format(base_messages)
    _text_base    = proc.apply_chat_template(_chat_fmt, tokenize=False, add_generation_prompt=False)
    base_inputs_raw = proc(text=[_text_base], images=[img], return_tensors="pt", padding=True)
    base_inputs   = {k: v.to(DEVICE) if hasattr(v, "to") else v
                     for k, v in base_inputs_raw.items()}
    input_ids_base = base_inputs["input_ids"][0].cpu().tolist()

    # Derive patch grid dimensions
    thw = base_inputs["image_grid_thw"][0]
    T, H, W  = int(thw[0]), int(thw[1]), int(thw[2])
    ranges   = iface._find_image_token_ranges(input_ids_base)
    img_start, img_end = ranges[0]
    n_img    = img_end - img_start
    merge    = max(1, int(round(math.sqrt(T * H * W / n_img))))
    grid_h, grid_w = H // merge, W // merge
    img_pos_t = torch.tensor(list(range(img_start, img_end)), dtype=torch.long)

    # Find key-noun token positions in the base input
    q_pos_set: list[int] = []
    for noun in key_nouns:
        q_pos_set.extend(find_keyword_positions(input_ids_base, noun))
    q_pos_set = sorted(set(q_pos_set))

    if not q_pos_set:
        # Fallback: all question tokens (after image end)
        q_pos_set = list(range(img_end, len(input_ids_base)))
        print(f"  key nouns not found — using all question tokens ({len(q_pos_set)} tokens)")
    else:
        decoded_kw = [proc.tokenizer.decode([input_ids_base[p]]) for p in q_pos_set]
        print(f"  key noun positions: {q_pos_set} → {decoded_kw}")

    # 10-run stochastic loop
    visual_grids: list[np.ndarray] = []
    for run_idx in range(N_RUNS):
        gen_out   = iface.generate([base_messages])[0]
        resp_text = gen_out.get("text", "").strip()

        # Append model response and re-encode for the forward pass
        full_messages = base_messages + [{"role": "assistant", "content": resp_text}]
        _chat_full    = iface._to_qwen_chat_format(full_messages)
        _text_full    = proc.apply_chat_template(_chat_full, tokenize=False, add_generation_prompt=False)
        full_inputs_raw = proc(text=[_text_full], images=[img], return_tensors="pt", padding=True)
        full_inputs   = {k: v.to(DEVICE) if hasattr(v, "to") else v
                         for k, v in full_inputs_raw.items()}

        g = capture_layer_attn(full_inputs, q_pos_set, img_pos_t, grid_h, grid_w)
        if g is not None:
            visual_grids.append(g)
        print(f"    run {run_idx+1}/{N_RUNS}: {'ok' if g is not None else 'skip'}"
              f"  resp={resp_text[:50]!r}")

    n_valid = len(visual_grids)
    avg_grid = np.mean(visual_grids, axis=0) if visual_grids else np.zeros((grid_h, grid_w))
    visual_cell_scores = grid_to_cell_scores(avg_grid, cell_bboxes, img_w, img_h)
    # Trim to displayed rows (grid_to_cell_scores uses n_rows global; clamp here)
    visual_cell_scores = visual_cell_scores[:d_rows, :]
    print(f"  visual: {n_valid} valid runs  max_cell={visual_cell_scores.max():.4f}")

    # Save averaged grids for potential re-use
    np.save(os.path.join(q_out, "avg_grid.npy"), avg_grid)

    # ── 3. Visualise ─────────────────────────────────────────────────────────
    plot_text_panels(q_id, question, text_cell_scores, display_df, q_out)
    plot_visual_solo(q_id, question, avg_grid, img_arr, img_h, img_w, n_valid, q_out)
    print(f"  plots saved → {q_out}/")

    # ── 4. Fair image MSE — single generate_with_attention call ──────────────
    # Uses Q[-1] from the last decoder layer, identical query strategy to text
    # modes. This is separate from the 10-run key-noun visualization above.
    image_mse_result  = iface.generate_with_attention(base_messages)
    image_attn_data   = image_mse_result.get("attention_data")
    if image_attn_data is not None:
        image_cell_scores_fair = _map_image_attention_to_cells(
            display_df, image_attn_data, cell_bboxes, (img_w, img_h)
        ).values.astype(float)
    else:
        print("  WARNING: no attention data for image (fair MSE)")
        image_cell_scores_fair = np.zeros((d_rows, d_cols))
    print(f"  image (fair): max={image_cell_scores_fair.max():.4f}")

    # ── 5. Metrics (balanced MSE + Average Precision) ────────────────────────
    metrics_row: dict = {
        "question_id": q_id,
        "question": question,
        "required_actions": req_acts,
        "display_rows": d_rows,
    }
    if gt_used is not None:
        for mode in TEXT_MODES:
            bmse, ap = compute_metrics(text_cell_scores[mode], gt_used)
            metrics_row[f"bmse_{mode}"] = bmse
            metrics_row[f"ap_{mode}"]   = ap
        bmse, ap = compute_metrics(image_cell_scores_fair, gt_used)
        metrics_row["bmse_image"]     = bmse
        metrics_row["ap_image"]       = ap
        bmse, ap = compute_metrics(visual_cell_scores, gt_used)
        metrics_row["bmse_image_viz"] = bmse
        metrics_row["ap_image_viz"]   = ap
    else:
        for mode in TEXT_MODES:
            metrics_row[f"bmse_{mode}"] = float("nan")
            metrics_row[f"ap_{mode}"]   = float("nan")
        for k in ["bmse_image", "ap_image", "bmse_image_viz", "ap_image_viz"]:
            metrics_row[k] = float("nan")
        print(f"  Metrics skipped (GT sheet q{q_id} is invalid)")
    all_results.append(metrics_row)

    bmse_str = "  |  ".join(
        f"{k.replace('bmse_','').replace('ap_','')}  bmse={metrics_row.get('bmse_'+k.replace('bmse_','').replace('ap_',''), float('nan')):.4f}"
        f"  ap={metrics_row.get('ap_'+k.replace('bmse_','').replace('ap_',''), float('nan')):.4f}"
        for k in [f"bmse_{m}" for m in TEXT_MODES] + ["bmse_image", "bmse_image_viz"]
    )
    print(f"  Metrics:\n    " + "\n    ".join(
        f"{k.replace('bmse_','')}: bmse={metrics_row[k]:.4f}  ap={metrics_row['ap_'+k[5:]]:.4f}"
        for k in [f"bmse_{m}" for m in TEXT_MODES] + ["bmse_image", "bmse_image_viz"]
    ))

# ── Save results CSV ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR, "metrics_results.csv"), index=False)
print(f"\nSaved results → {OUT_DIR}/metrics_results.csv")
metric_cols = [c for c in results_df.columns if c.startswith("bmse_") or c.startswith("ap_")]
print(results_df[["question_id"] + metric_cols].to_string(index=False))

# ── Shared plot config ────────────────────────────────────────────────────────
mod_suffixes = [f"{m}" for m in TEXT_MODES] + ["image", "image_viz"]
mod_labels   = {f"{m}": l for m, l in TEXT_LABELS.items()}
mod_labels["image"]     = "Image (Q[-1])"
mod_labels["image_viz"] = "Image (key-noun)"
mod_colors   = {"text_markdown": "#2ca02c", "text_json": "#ff7f0e", "text_latex": "#9467bd",
                "image": "#1f77b4", "image_viz": "#aec7e8"}

def _save(fig, stem):
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)

def _heatmap(data, row_labels, col_labels, title, cbar_label, cmap, higher_better=False):
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.4), max(4, len(row_labels) * 0.55 + 1.5)))
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(data) if not np.all(np.isnan(data)) else 0,
                   vmax=np.nanmax(data) if not np.all(np.isnan(data)) else 1)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
    vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            lbl = "N/A" if math.isnan(v) else f"{v:.3f}"
            dark = (v > vmax * 0.65) if not math.isnan(v) else False
            color = "white" if (higher_better != dark) else "black"
            ax.text(j, i, lbl, ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.03)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig

def _bar_by_action(metric_prefix, ylabel, title, higher_better=False):
    cols = [f"{metric_prefix}{s}" for s in mod_suffixes]
    labels = [mod_labels[s] for s in mod_suffixes]
    action_rows = []
    for _, r in results_df.iterrows():
        for act in [a.strip() for a in r["required_actions"].split(",")]:
            for col, lbl in zip(cols, labels):
                action_rows.append({"action": act, "modality": lbl, "val": r[col]})
    adf = pd.DataFrame(action_rows)
    actions = sorted(adf["action"].unique())
    x = np.arange(len(actions)); width = 0.17
    fig, ax = plt.subplots(figsize=(max(9, len(actions) * 1.8), 5))
    for i, (suf, lbl) in enumerate(zip(mod_suffixes, labels)):
        sub = adf[adf["modality"] == lbl]
        means = [sub[sub["action"] == a]["val"].dropna().mean() for a in actions]
        bars = ax.bar(x + i * width, means, width, label=lbl,
                      color=mod_colors.get(suf), alpha=0.85)
        for bar, v in zip(bars, means):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x + width * (len(mod_suffixes) - 1) / 2)
    ax.set_xticklabels(actions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def _bar_overall(metric_prefix, ylabel, title):
    cols = [f"{metric_prefix}{s}" for s in mod_suffixes]
    labels = [mod_labels[s] for s in mod_suffixes]
    means = [results_df[c].dropna().mean() for c in cols]
    colors = [mod_colors[s] for s in mod_suffixes]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, color=colors, alpha=0.85)
    for bar, v in zip(bars, means):
        if not math.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=15); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

# ── 1. Heatmaps ───────────────────────────────────────────────────────────────
row_labels = [f"Q{r['question_id']}: {r['question'][:35]}…" for _, r in results_df.iterrows()]
col_labels = [mod_labels[s] for s in mod_suffixes]

bmse_data = results_df[[f"bmse_{s}" for s in mod_suffixes]].values
fig = _heatmap(bmse_data, row_labels, col_labels,
               "Balanced MSE  (questions × modalities) — lower = better",
               "Balanced MSE", "YlOrRd", higher_better=False)
_save(fig, "bmse_heatmap")

ap_data = results_df[[f"ap_{s}" for s in mod_suffixes]].values
fig = _heatmap(ap_data, row_labels, col_labels,
               "Average Precision  (questions × modalities) — higher = better",
               "AP", "YlGn", higher_better=True)
_save(fig, "ap_heatmap")

# ── 2. By required-action ─────────────────────────────────────────────────────
fig = _bar_by_action("bmse_", "Mean Balanced MSE",
                     "Balanced MSE by required action  (lower = better)")
_save(fig, "bmse_by_action")

fig = _bar_by_action("ap_", "Mean Average Precision",
                     "Average Precision by required action  (higher = better)",
                     higher_better=True)
_save(fig, "ap_by_action")

# ── 3. Overall ────────────────────────────────────────────────────────────────
fig = _bar_overall("bmse_", "Mean Balanced MSE",
                   "Overall balanced MSE by modality  (lower = better)")
_save(fig, "bmse_overall")

fig = _bar_overall("ap_", "Mean Average Precision",
                   "Overall Average Precision by modality  (higher = better)")
_save(fig, "ap_overall")

print(f"\nAll done.  Results → {OUT_DIR}/")
