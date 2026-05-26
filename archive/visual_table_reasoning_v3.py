#!/usr/bin/env python3
import argparse
import ast
import contextlib
import csv
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from vagen.inference.model_interface.factory_model import ModelFactory


def _ensure_tool_imports():
    try:
        from enhanced_modular_tools_new import ENHANCED_MODULAR_TOOLS  # type: ignore
        from enhanced_modular_tools_def import function_definition  # type: ignore
        return ENHANCED_MODULAR_TOOLS, function_definition
    except Exception:
        tools_path = os.getenv("VAGEN_TOOLS_PATH")
        if not tools_path:
            tools_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "multitable_reasoning",
                "proof_of_concept",
            )
        if tools_path and tools_path not in sys.path:
            sys.path.append(tools_path)
        from enhanced_modular_tools_new import ENHANCED_MODULAR_TOOLS  # type: ignore
        from enhanced_modular_tools_def import function_definition  # type: ignore
        return ENHANCED_MODULAR_TOOLS, function_definition


ENHANCED_MODULAR_TOOLS, FUNCTION_DEFINITION = _ensure_tool_imports()


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def remove_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    total_like = df.apply(
        lambda row: any(
            isinstance(cell, str) and cell.strip().lower() in {"total", "totals", "subtotal"}
            for cell in row
        ),
        axis=1,
    )
    return df.loc[~total_like].reset_index(drop=True)


def load_wikitablequestions_dataset(
    dataset_path: str,
    n_samples: int = 1,
    random_seed: int = 42,
    use_sequential: bool = False,
    start: int = 0,
    end: int | None = None,
):
    try:
        from datasets import load_from_disk
    except Exception as e:
        raise ImportError("datasets library is required. Install with: pip install datasets") from e

    dataset = load_from_disk(dataset_path)
    if "train" in dataset:
        split_data = dataset["train"]
    else:
        split_name = list(dataset.keys())[0]
        split_data = dataset[split_name]

    total_examples = len(split_data)
    random.seed(random_seed)
    if use_sequential:
        if end is None:
            end = start + max(0, int(n_samples))
        end = min(end, total_examples)
        indices = list(range(start, end))
    else:
        k = total_examples if n_samples < 0 else min(int(n_samples), total_examples)
        indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    for idx in indices:
        example = split_data[idx]
        table_data = example.get("table", {})
        if "header" in table_data and "rows" in table_data:
            df = pd.DataFrame(table_data["rows"], columns=table_data["header"])
            df.columns = [
                f"{col}_{i}" if df.columns.tolist().count(col) > 1 and i > 0 else col
                for i, col in enumerate(df.columns)
            ]
            df = df.dropna(axis=1, how="all")
            df.columns = [c.replace(" ", "") if isinstance(c, str) and " " in c else c for c in df.columns]
            df.rename(columns=lambda c: c.replace("–", "-") if isinstance(c, str) else c, inplace=True)
            df.replace("–", "-", regex=True, inplace=True)
            df = df.rename(columns=lambda c: c.replace("\u00A0", " ") if isinstance(c, str) else c).replace(
                "\u00A0", " ", regex=True
            )
            df = df.rename(columns=lambda c: c.replace("\"", " ") if isinstance(c, str) else c).replace(
                "\"", " ", regex=True
            )
            df = df.rename(columns=lambda c: c.replace("'", "`") if isinstance(c, str) else c).replace(
                "'", "\\'", regex=True
            )
            df = df.rename(columns=lambda c: c.replace("\n", "").replace("\\n", "") if isinstance(c, str) else c).replace(
                {r"\n ": ",", r"\\n df": ","}, regex=True
            )
            df = df.applymap(normalize_text)
            df = remove_total_rows(df)

            df_og = df.copy()
            yield {
                "question": normalize_text(example.get("question", "")),
                "table": df,
                "df_og": df_og,
                "answer": ", ".join(example.get("answers", [])) if example.get("answers") else "",
                "table_id": example.get("id", f"sample_{idx}"),
                "original_index": idx,
            }


def load_mmqa_dataset(
    dataset_path: str,
    n_samples: int = 1,
    random_seed: int = 1018,
    start: int = 0,
    end: int | None = None,
) -> List[Dict[str, Any]]:
    base_df = pd.read_csv(f"{dataset_path}/MMQA_dataset.csv")
    total_examples = len(base_df)
    random.seed(random_seed)
    k = total_examples if n_samples < 0 else min(int(n_samples), total_examples)
    indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    samples = []
    for idx in indices:
        example = base_df.iloc[idx]
        from datasets import load_from_disk
        df = load_from_disk(f"{dataset_path}/MMQA_datasets/{example['table_id']}").to_pandas()
        df.columns = [
            f"{col}_{i}" if df.columns.tolist().count(col) > 1 and i > 0 else col
            for i, col in enumerate(df.columns)
        ]
        df = df.dropna(axis=1, how="all")
        df.columns = [c.replace(" ", "") if isinstance(c, str) and " " in c else c for c in df.columns]
        df.rename(columns=lambda c: c.replace("–", "-") if isinstance(c, str) else c, inplace=True)
        df.replace("–", "-", regex=True, inplace=True)
        df = df.rename(columns=lambda c: c.replace("\u00A0", " ") if isinstance(c, str) else c).replace(
            "\u00A0", " ", regex=True
        )
        df = df.rename(columns=lambda c: c.replace("\"", " ") if isinstance(c, str) else c).replace(
            "\"", " ", regex=True
        )
        df = df.rename(columns=lambda c: c.replace("'", "`") if isinstance(c, str) else c).replace(
            "'", "\\'", regex=True
        )
        df = df.rename(columns=lambda c: c.replace("\n", "").replace("\\n", "") if isinstance(c, str) else c).replace(
            {r"\n ": ",", r"\\n df": ","}, regex=True
        )
        df = df.applymap(normalize_text)
        df = remove_total_rows(df)
        df_og = df.copy()
        samples.append(
            {
                "question": normalize_text(example["question"]),
                "table": df,
                "df_og": df_og,
                "answer": example.get("correct_answer", ""),
                "table_id": example["table_id"],
                "original_index": idx,
            }
        )
    return samples


def load_mmtu_dataset(
    dataset_path: str,
    n_samples: int = 1,
    start: int = 0,
    end: int | None = None,
    random_seed: int = 1018,
) -> List[Dict[str, Any]]:
    base_df = pd.read_excel(f"{dataset_path}/mmtu_record.xlsx")
    total_examples = len(base_df)
    random.seed(random_seed)
    k = total_examples if n_samples < 0 else min(int(n_samples), total_examples)
    indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    samples = []
    for idx in indices:
        example = base_df.iloc[idx]
        from datasets import load_from_disk
        df = load_from_disk(f"{dataset_path}/MMTU_datasets/{example['id']}").to_pandas()
        df.columns = [
            f"{col}_{i}" if df.columns.tolist().count(col) > 1 and i > 0 else col
            for i, col in enumerate(df.columns)
        ]
        df = df.dropna(axis=1, how="all")
        df.columns = [c.replace(" ", "") if isinstance(c, str) and " " in c else c for c in df.columns]
        df.rename(columns=lambda c: c.replace("–", "-") if isinstance(c, str) else c, inplace=True)
        df.replace("–", "-", regex=True, inplace=True)
        df = df.rename(columns=lambda c: c.replace("\u00A0", " ") if isinstance(c, str) else c).replace(
            "\u00A0", " ", regex=True
        )
        df = df.rename(columns=lambda c: c.replace("\"", " ") if isinstance(c, str) else c).replace(
            "\"", " ", regex=True
        )
        df = df.rename(columns=lambda c: c.replace("'", "`") if isinstance(c, str) else c).replace(
            "'", "\\'", regex=True
        )
        df = df.rename(columns=lambda c: c.replace("\n", "").replace("\\n", "") if isinstance(c, str) else c).replace(
            {r"\n ": ",", r"\\n df": ","}, regex=True
        )
        df = df.applymap(normalize_text)
        df = remove_total_rows(df)
        df_og = df.copy()
        samples.append(
            {
                "question": normalize_text(example["question"]),
                "table": df,
                "df_og": df_og,
                "answer": example.get("ground_truth", ""),
                "table_id": example["id"],
                "original_index": idx,
            }
        )
    return samples


def render_dataframe_to_image(df: pd.DataFrame, output_path: str, max_rows: int = 30, max_cols: int = 5):
    if df is None or df.empty:
        raise ValueError("Empty dataframe, nothing to render.")

    display_df = df.copy()
    if len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
    if display_df.shape[1] > max_cols:
        display_df = display_df.iloc[:, :max_cols]

    rows, cols = display_df.shape
    fig_w = min(20, max(6, cols * 1.2))
    fig_h = min(12, max(3, rows * 0.4))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_image_message(image_path: str, prompt: str) -> List[Dict[str, Any]]:
    img = Image.open(image_path)
    return [{
        "role": "user",
        "content": f"<image>\n{prompt}",
        "multi_modal_data": {"<image>": [img]},
    }]


def build_multimodal_table_message(
    image_path: str,
    column_names: List[Any],
    prompt: str,
) -> List[Dict[str, Any]]:
    img = Image.open(image_path)
    column_text = ", ".join(normalize_text(col) for col in column_names)
    content = (
        "<image>\n"
        f"{prompt}\n\n"
        f"Column names: [{column_text}]"
    )
    return [{
        "role": "user",
        "content": content,
        "multi_modal_data": {"<image>": [img]},
    }]


def serialize_dataframe_to_text(
    df: pd.DataFrame,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> str:
    if df is None or df.empty:
        raise ValueError("Empty dataframe, nothing to serialize.")

    display_df = df.copy()
    if max_rows is not None and len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
    if max_cols is not None and display_df.shape[1] > max_cols:
        display_df = display_df.iloc[:, :max_cols]

    serialized = display_df.to_markdown(index=True)
    notes = []
    if max_rows is not None and len(df) > len(display_df):
        notes.append(f"showing first {len(display_df)} of {len(df)} rows")
    if max_cols is not None and df.shape[1] > display_df.shape[1]:
        notes.append(f"showing first {display_df.shape[1]} of {df.shape[1]} columns")
    if notes:
        serialized = f"{serialized}\n\n[Truncated view: {', '.join(notes)}]"
    return serialized


def serialize_dataframe_to_json(
    df: pd.DataFrame,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> str:
    if df is None or df.empty:
        raise ValueError("Empty dataframe, nothing to serialize.")
    display_df = df.copy()
    if max_rows is not None and len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
    if max_cols is not None and display_df.shape[1] > max_cols:
        display_df = display_df.iloc[:, :max_cols]
    serialized = display_df.to_json(orient="records", force_ascii=False, indent=2)
    notes = []
    if max_rows is not None and len(df) > len(display_df):
        notes.append(f"showing first {len(display_df)} of {len(df)} rows")
    if max_cols is not None and df.shape[1] > display_df.shape[1]:
        notes.append(f"showing first {display_df.shape[1]} of {df.shape[1]} columns")
    if notes:
        serialized = f"{serialized}\n\n[Truncated view: {', '.join(notes)}]"
    return serialized


def serialize_dataframe_to_latex(
    df: pd.DataFrame,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> str:
    if df is None or df.empty:
        raise ValueError("Empty dataframe, nothing to serialize.")
    display_df = df.copy()
    if max_rows is not None and len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
    if max_cols is not None and display_df.shape[1] > max_cols:
        display_df = display_df.iloc[:, :max_cols]
    try:
        serialized = display_df.to_latex(index=True)
    except Exception:
        serialized = display_df.to_string(index=True)
    notes = []
    if max_rows is not None and len(df) > len(display_df):
        notes.append(f"showing first {len(display_df)} of {len(df)} rows")
    if max_cols is not None and df.shape[1] > display_df.shape[1]:
        notes.append(f"showing first {display_df.shape[1]} of {df.shape[1]} columns")
    if notes:
        serialized = f"{serialized}\n\n[Truncated view: {', '.join(notes)}]"
    return serialized


def build_serialized_table_message(serialized_table: str, prompt: str) -> List[Dict[str, Any]]:
    return [{
        "role": "user",
        "content": f"{prompt}\n\nTable:\n```markdown\n{serialized_table}\n```",
    }]


def _strip_code_fences(text: str) -> str:
    fence = re.search(r"```(?:json|csv|txt)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _extract_json_blob(text: str) -> Optional[str]:
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def _safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _parse_table_from_text(text: str) -> Optional[pd.DataFrame]:
    blob = _extract_json_blob(text)
    if blob:
        parsed = _safe_json_loads(blob)
        if isinstance(parsed, dict) and "header" in parsed and "rows" in parsed:
            try:
                df = pd.DataFrame(parsed["rows"], columns=parsed["header"])
                return df
            except Exception:
                pass

    cleaned = _strip_code_fences(text)
    if cleaned.count(",") >= 2 and "\n" in cleaned:
        try:
            return pd.read_csv(io.StringIO(cleaned))
        except Exception:
            return None
    return None


def extract_table_from_image(model, image_path: str, question: str) -> pd.DataFrame:
    prompt = (
        "You are given a table image and a question about it.\n"
        "Extract the table into strict JSON with keys: header (list of column names) "
        "and rows (list of row lists). Output JSON only.\n\n"
        f"Question: {question}"
    )
    messages = build_image_message(image_path, prompt)
    result = model.generate([messages])[0]
    #print(result)
    text = result.get("text", "")
    df = _parse_table_from_text(text)
    if df is None or df.empty:
        print(result)
        raise ValueError("Failed to parse table from model output.")
    df = df.dropna(axis=1, how="all")
    df.columns = [normalize_text(c).replace(" ", "") for c in df.columns]
    df = df.applymap(normalize_text)
    df = remove_total_rows(df)
    return df


def extract_table_from_image_with_usage(
    model,
    image_path: str,
    question: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    prompt = (
        "You are given a table image and a question about it.\n"
        "Extract the table into strict JSON with keys: header (list of column names) "
        "and rows (list of row lists). Output JSON only.\n\n"
        f"Question: {question}"
    )
    messages = build_image_message(image_path, prompt)
    result = model.generate([messages])[0]
    text = result.get("text", "")
    df = _parse_table_from_text(text)
    if df is None or df.empty:
        print(result)
        raise ValueError("Failed to parse table from model output.")
    df = df.dropna(axis=1, how="all")
    df.columns = [normalize_text(c).replace(" ", "") for c in df.columns]
    df = df.applymap(normalize_text)
    df = remove_total_rows(df)
    return df, _merge_usage(result.get("usage"))


def _tool_catalog() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tool_map: Dict[str, Any] = {}
    tool_specs: Dict[str, Any] = {}
    for tool in ENHANCED_MODULAR_TOOLS:
        schema = tool.args_schema.model_json_schema()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        params = {}
        for name, info in props.items():
            params[name] = {
                "type": info.get("type", "string"),
                "required": name in required,
            }
        tool_map[tool.name] = tool
        tool_specs[tool.name] = {
            "description": tool.description,
            "params": params,
        }
    return tool_map, tool_specs


def _format_tool_specs(tool_specs: Dict[str, Any]) -> str:
    lines = []
    for name, spec in tool_specs.items():
        params = spec.get("params") or {}
        if params:
            params_str = ", ".join(
                f"{p}{'*' if params[p].get('required') else ''}" for p in params.keys()
            )
        else:
            params_str = "none"
        lines.append(f"- {name}: {spec.get('description', '')} | params: {params_str}")
    return "\n".join(lines)


def _capture_exec(code: str, env: Dict[str, Any]) -> Tuple[str, Optional[Exception]]:
    buffer = io.StringIO()
    err: Optional[Exception] = None
    with contextlib.redirect_stdout(buffer):
        try:
            exec(code, env)
        except Exception as exc:
            err = exc
    return buffer.getvalue(), err


def _empty_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(*usages: Optional[Dict[str, Any]]) -> Dict[str, int]:
    merged = _empty_usage()
    for usage in usages:
        if not usage:
            continue
        for key in merged:
            try:
                merged[key] += int(usage.get(key, 0) or 0)
            except Exception:
                continue
    return merged


def _uses_step_image(table_input_mode: str, image_each_step: bool) -> bool:
    return table_input_mode in {"image", "multimodal"} or image_each_step


def _parse_table_input_modes(table_input_mode: str, table_input_modes: Optional[str]) -> List[str]:
    valid_modes = {"text", "image", "multimodal", "text_latex", "text_json"}
    if table_input_modes:
        modes = [m.strip() for m in table_input_modes.split(",") if m.strip()]
    else:
        modes = [table_input_mode]
    invalid = [m for m in modes if m not in valid_modes]
    if invalid:
        raise ValueError(f"Invalid table input mode(s): {invalid}. Valid modes are: {sorted(valid_modes)}")
    deduped = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def _mode_fieldnames(mode: str, compact_csv: bool) -> List[str]:
    fields = [
        f"answer_{mode}",
        f"prompt_tokens_{mode}",
        f"completion_tokens_{mode}",
        f"total_tokens_{mode}",
        f"all_usage_{mode}",
        f"error_{mode}",
        f"num_turns_{mode}",
        f"latency_seconds_{mode}",
    ]
    if not compact_csv:
        fields.extend(
            [
                f"reasoning_trace_{mode}",
                f"all_answers_{mode}",
                f"all_reasoning_traces_{mode}",
                f"chosen_index_{mode}",
                f"selection_reason_{mode}",
            ]
        )
    return fields


def _run_mode_reasoning(
    model,
    df: pd.DataFrame,
    question: str,
    args: argparse.Namespace,
    table_input_mode: str,
    render_dir: Optional[str],
    base_usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    error_msg = ""
    reasoning_trace: List[Dict[str, Any]] = []
    all_answers: List[str] = []
    all_usages: List[Dict[str, int]] = []
    all_traces: List[List[Dict[str, Any]]] = []
    chosen_index = None
    selection_reason = ""
    usage = _empty_usage()
    answer = ""

    base_usage = _merge_usage(base_usage)
    attempts = max(1, int(args.attempts_per_question))
    try:
        for _ in range(attempts):
            try:
                answer, reasoning_trace, run_usage = run_reasoning_loop(
                    model=model,
                    df=df.copy(),
                    question=question,
                    max_steps=args.max_steps,
                    json_retries=args.json_retries,
                    max_tokens=args.max_tokens,
                    image_each_step=args.image_each_step,
                    table_input_mode=table_input_mode,
                    render_dir=render_dir,
                    debug=args.debug,
                )
                run_usage = _merge_usage(base_usage, run_usage)
            except Exception as attempt_err:
                answer = ""
                run_usage = _empty_usage()
                reasoning_trace = [{
                    "step": 0,
                    "reasoning": "",
                    "tool_name": "",
                    "tool_args": {},
                    "tool_output": "",
                    "error": str(attempt_err),
                }]
            all_answers.append(answer)
            all_usages.append(run_usage)
            all_traces.append(reasoning_trace)

        if len(all_answers) == 1:
            answer = all_answers[0]
            usage = all_usages[0]
            reasoning_trace = all_traces[0]
        elif args.rerank_answers:
            chosen_index, selection_reason = _select_best_answer(
                model=model,
                question=question,
                candidates=all_answers,
                traces=all_traces,
                json_retries=args.json_retries,
            )
            answer = all_answers[chosen_index]
            usage = all_usages[chosen_index]
            reasoning_trace = all_traces[chosen_index]
        else:
            answer = all_answers[0]
            usage = all_usages[0]
            reasoning_trace = all_traces[0]
    except Exception as e:
        answer = ""
        error_msg = str(e)
        print(f"Error: {e}")

    return {
        "answer": answer,
        "usage": usage,
        "error": error_msg,
        "reasoning_trace": reasoning_trace,
        "all_answers": all_answers,
        "all_usage": all_usages,
        "all_reasoning_traces": all_traces,
        "chosen_index": chosen_index,
        "selection_reason": selection_reason,
    }


def _select_best_answer(
    model,
    question: str,
    candidates: List[str],
    traces: List[List[Dict[str, Any]]],
    json_retries: int = 2,
) -> Tuple[int, str]:
    """Use the model to pick the best answer among candidates."""
    condensed_traces = []
    for idx, trace in enumerate(traces):
        steps = []
        for entry in trace[:6]:
            steps.append(
                {
                    "tool": entry.get("tool_name"),
                    "args": entry.get("tool_args"),
                    "error": entry.get("error"),
                }
            )
        condensed_traces.append({"index": idx, "steps": steps})

    prompt = (
        "Choose the best answer for the question based on the candidates and traces.\n"
        "Return JSON only: {\"chosen_index\": int, \"reason\": str}\n\n"
        f"Question: {question}\n"
        f"Candidates: {json.dumps(candidates, ensure_ascii=True)}\n"
        f"Traces (condensed): {json.dumps(condensed_traces, ensure_ascii=True)}"
    )
    messages = [{"role": "user", "content": prompt}]

    payload = None
    text = ""
    usage_totals = _empty_usage()
    for _ in range(max(0, int(json_retries)) + 1):
        resp = model.generate([messages])[0]
        usage_totals = _merge_usage(usage_totals, resp.get("usage"))
        text = resp.get("text", "")
        payload_text = _extract_json_blob(text) or _strip_code_fences(text)
        payload = _safe_json_loads(payload_text)
        if isinstance(payload, dict):
            break
    if not isinstance(payload, dict):
        return 0, f"Selection parse failed; defaulting to index 0. Raw: {text[:200]}"

    chosen_index = payload.get("chosen_index")
    reason = payload.get("reason", "")
    try:
        chosen_index = int(chosen_index)
    except Exception:
        chosen_index = 0
    if chosen_index < 0 or chosen_index >= len(candidates):
        chosen_index = 0
    return chosen_index, str(reason)


def run_reasoning_loop(
    model,
    df: pd.DataFrame,
    question: str,
    max_steps: int,
    json_retries: int = 2,
    max_tokens: int | None = None,
    image_each_step: bool = False,
    table_input_mode: str = "image",
    render_dir: str | None = None,
    debug: bool = False,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    tool_map, tool_specs = _tool_catalog()
    tool_specs_text = _format_tool_specs(tool_specs)

    env: Dict[str, Any] = {"pd": pd}
    exec(FUNCTION_DEFINITION(), env)
    env["df"] = df
    env["df_og"] = df.copy()

    messages: List[Dict[str, Any]] = []
    last_tool_output = ""
    last_error = ""
    usage_totals = _empty_usage()

    use_step_image = _uses_step_image(table_input_mode, image_each_step)

    temp_ctx = None
    if use_step_image and render_dir is None:
        temp_ctx = tempfile.TemporaryDirectory()
        render_dir = temp_ctx.name

    reasoning_trace: List[Dict[str, Any]] = []
    action_ledger: List[Tuple[int, str, str]] = []  # (step_num, tool_name, canonical_args)
    try:
        for step in range(1, max_steps + 1):
            df_preview = env["df"].head(15).to_dict(orient="records")

            # --- Step budget note ---
            steps_left = max_steps - step + 1
            if steps_left == 1:
                step_budget_note = f"[URGENT] This is your last step (step {step}/{max_steps}). You MUST call f_final_answer now with your best answer."
            elif steps_left <= 3:
                step_budget_note = f"[NOTE] {steps_left} steps remaining including this one (step {step}/{max_steps}). Start converging toward f_final_answer."
            else:
                step_budget_note = f"Step {step} of {max_steps}."

            # --- Cross-step action ledger ---
            if action_ledger:
                key_counts: Dict[Tuple[str, str], int] = {}
                for _, n, a in action_ledger:
                    key_counts[(n, a)] = key_counts.get((n, a), 0) + 1
                ledger_lines = []
                for step_n, n, a in action_ledger:
                    display_a = a[:80] + "..." if len(a) > 80 else a
                    repeat = key_counts[(n, a)]
                    tag = f" <- REPEATED {repeat}x, avoid calling again" if repeat > 1 else ""
                    ledger_lines.append(f"  step{step_n}: {n}({display_a}){tag}")
                repeated_keys = [(n, a) for (n, a), cnt in key_counts.items() if cnt >= 2]
                if repeated_keys:
                    drift_warning = (
                        "\n[DRIFT WARNING] These tool+args combos have been called 2+ times: "
                        + ", ".join(f"{n}({a[:40]})" for n, a in repeated_keys)
                        + ". You MUST try a different approach or call f_final_answer."
                    )
                else:
                    drift_warning = ""
                ledger_text = "Prior actions this episode:\n" + "\n".join(ledger_lines) + drift_warning
            else:
                ledger_text = "Prior actions this episode: none"

            tool_prompt = (
                "You are a table reasoning agent. Use tools to transform the table and answer the question.\n"
                "You may batch multiple simple actions together in one step to save time.\n"
                "Respond with JSON only using the schema:\n"
                "{"
                "\"reasoning\": str, "
                "\"actions\": [{\"tool_name\": str, \"tool_args\": object}]"
                "}\n"
                "\"actions\" is a list of one or more tools to execute in sequence this step.\n"
                "Group low-complexity operations together (e.g. filter + sort, select + rename).\n"
                "Use a single action when the result of one step determines the next (e.g. after a filter you need to check row count).\n"
                "To answer, include {\"tool_name\": \"f_final_answer\", \"tool_args\": {\"answer\": \"...\"}} as the last action.\n"
                "To revert the table, use f_retrieve_original_df.\n\n"
                f"{step_budget_note}\n"
                f"Question: {question}\n"
                f"Table shape: {env['df'].shape}\n"
                f"Columns: {env['df'].columns.tolist()}\n"
                f"Preview rows (JSON): {json.dumps(df_preview, ensure_ascii=True)}\n"
                f"Last step output: {last_tool_output[:400] if last_tool_output else 'N/A'}\n"
                f"Last error: {last_error if last_error else 'N/A'}\n"
                f"{ledger_text}\n\n"
                f"Tools:\n{tool_specs_text}"
            )
            if table_input_mode in {"image", "multimodal"}:
                if not render_dir:
                    raise ValueError("render_dir is required when step image input is enabled.")
                os.makedirs(render_dir, exist_ok=True)
                image_path = os.path.join(render_dir, f"step_{step}.jpg")
                render_dataframe_to_image(env["df"], image_path)
                if table_input_mode == "multimodal":
                    messages.append(build_multimodal_table_message(image_path, env["df"].columns.tolist(), tool_prompt)[0])
                else:
                    messages.append(build_image_message(image_path, tool_prompt)[0])
            elif table_input_mode == "text":
                serialized_table = serialize_dataframe_to_text(env["df"])
                messages.append(build_serialized_table_message(serialized_table, tool_prompt)[0])
            elif table_input_mode == "text_json":
                serialized_table = serialize_dataframe_to_json(env["df"])
                messages.append({"role": "user", "content": f"{tool_prompt}\n\nTable:\n```json\n{serialized_table}\n```"})
            elif table_input_mode == "text_latex":
                serialized_table = serialize_dataframe_to_latex(env["df"])
                messages.append({"role": "user", "content": f"{tool_prompt}\n\nTable:\n```latex\n{serialized_table}\n```"})
            elif image_each_step:
                if not render_dir:
                    raise ValueError("render_dir is required when image_each_step is enabled.")
                os.makedirs(render_dir, exist_ok=True)
                image_path = os.path.join(render_dir, f"step_{step}.jpg")
                render_dataframe_to_image(env["df"], image_path)
                messages.append(build_image_message(image_path, tool_prompt)[0])
            else:
                messages.append({"role": "user", "content": tool_prompt})

            max_parse_retries = max(0, int(json_retries))
            payload = None
            text = ""
            for attempt in range(max_parse_retries + 1):
                response = model.generate([messages])[0]
                usage_totals = _merge_usage(usage_totals, response.get("usage"))
                text = response.get("text", "")
                usage = response.get("usage") or {}
                finish_reason = response.get("finish_reason")
                completion_tokens = usage.get("completion_tokens", 0)
                was_truncated = finish_reason == "length" or bool(
                    max_tokens and completion_tokens and completion_tokens >= max_tokens
                )
                if debug and was_truncated:
                    print(
                        json.dumps(
                            {
                                "event": "completion_truncated",
                                "finish_reason": finish_reason,
                                "usage": usage,
                                "text_preview": (text or "")[:200],
                            },
                            ensure_ascii=True,
                        )
                    )
                messages.append({"role": "assistant", "content": text})
                payload_text = _extract_json_blob(text) or _strip_code_fences(text)
                payload = _safe_json_loads(payload_text)
                if isinstance(payload, dict):
                    break
                if attempt < max_parse_retries:
                    if was_truncated:
                        # Response was cut off — ask for minimal JSON with no thinking chain.
                        corrective_prompt = (
                            "Your previous response was cut off before the JSON was complete.\n"
                            "Do NOT include any thinking or preamble text.\n"
                            "Output JSON ONLY, as concise as possible:\n"
                            "{"
                            "\"reasoning\": \"<one sentence>\", "
                            "\"actions\": [{\"tool_name\": str, \"tool_args\": object}]"
                            "}\n"
                            "Pick the single most useful next action, or call f_final_answer with your best answer now."
                        )
                    else:
                        # Response was not truncated but failed to parse as JSON.
                        corrective_prompt = (
                            "Your previous response was not valid JSON.\n"
                            "Respond with JSON ONLY using this schema:\n"
                            "{"
                            "\"reasoning\": str, "
                            "\"actions\": [{\"tool_name\": str, \"tool_args\": object}]"
                            "}\n"
                            "\"actions\" must be a non-empty list. "
                            "To answer use tool_name f_final_answer with tool_args {\"answer\": \"...\"}.\n"
                            "Do not include any extra text."
                        )
                    messages.append({"role": "user", "content": corrective_prompt})

            if not isinstance(payload, dict):
                last_error = f"Could not parse tool JSON at step {step}: {text[:200]}"
                if debug:
                    print(last_error)
                reasoning_trace.append(
                    {
                        "step": step,
                        "reasoning": "",
                        "tool_name": "",
                        "tool_args": {},
                        "tool_output": "",
                        "error": last_error,
                    }
                )
                continue

            # Support both new "actions" list and legacy single "tool_name"/"tool_args"
            raw_actions = payload.get("actions")
            if not isinstance(raw_actions, list) or len(raw_actions) == 0:
                # Backward compat: wrap single action
                single_name = str(payload.get("tool_name") or "").strip()
                single_args = payload.get("tool_args") or {}
                if single_name:
                    raw_actions = [{"tool_name": single_name, "tool_args": single_args}]
                else:
                    last_error = f"No actions found in response at step {step}."
                    if debug:
                        print(last_error)
                    reasoning_trace.append({
                        "step": step,
                        "reasoning": payload.get("reasoning", ""),
                        "actions": [],
                        "tool_output": "",
                        "error": last_error,
                    })
                    continue

            print(payload)

            # --- Resolve all actions to (tool_name, tool_args, code) ---
            resolved: List[Tuple[str, Dict, str]] = []  # (name, args, code)
            resolution_error = ""
            final_answer: Optional[str] = None

            for act in raw_actions:
                a_name = str(act.get("tool_name") or "").strip()
                a_args = act.get("tool_args") or {}
                if not a_name:
                    resolution_error = f"Empty tool_name in action list at step {step}."
                    break
                if a_name == "f_final_answer":
                    final_answer = str(a_args.get("answer", ""))
                    break  # nothing to execute after final answer
                tool = tool_map.get(a_name)
                if tool is None:
                    resolution_error = f"Unknown tool: {a_name}"
                    break
                if not isinstance(a_args, dict):
                    resolution_error = f"tool_args must be an object for {a_name}, got {type(a_args)}"
                    break
                try:
                    code = tool._run(**a_args)
                except Exception as exc:
                    resolution_error = f"Tool args error for {a_name}: {exc}"
                    break
                resolved.append((a_name, a_args, code))

            if resolution_error:
                last_error = resolution_error
                last_tool_output = ""
                reasoning_trace.append({
                    "step": step,
                    "reasoning": payload.get("reasoning", ""),
                    "actions": [{"tool_name": a.get("tool_name"), "tool_args": a.get("tool_args")} for a in raw_actions],
                    "tool_output": "",
                    "error": last_error,
                })
                continue

            # --- Execute resolved (non-final) actions ---
            step_output = ""
            step_error = ""
            if resolved:
                combined_code = "\n".join(code for _, _, code in resolved)
                output, err = _capture_exec(combined_code, env)
                if err:
                    # Batch failed — re-run individually to find the culprit and
                    # roll back env to the snapshot taken before this batch.
                    env_snapshot = {k: (v.copy() if isinstance(v, pd.DataFrame) else v)
                                    for k, v in env.items() if k in ("df", "df_og")}
                    env.update(env_snapshot)
                    individual_outputs = []
                    for a_name, a_args, code in resolved:
                        out, e = _capture_exec(code, env)
                        individual_outputs.append(out.strip())
                        if e:
                            step_error = f"[{a_name}] {e}"
                            break
                    step_output = "\n".join(individual_outputs)
                else:
                    step_output = output.strip()

            last_tool_output = step_output
            last_error = step_error

            # Record executed actions in the cross-step ledger
            for a_name, a_args, _ in resolved:
                action_ledger.append((step, a_name, json.dumps(a_args, sort_keys=True)))
            if final_answer is not None:
                action_ledger.append((step, "f_final_answer", ""))

            action_summary = [{"tool_name": n, "tool_args": a} for n, a, _ in resolved]
            if final_answer is not None:
                action_summary.append({"tool_name": "f_final_answer", "tool_args": {"answer": final_answer}})

            reasoning_trace.append({
                "step": step,
                "reasoning": payload.get("reasoning", ""),
                "actions": action_summary,
                "tool_output": last_tool_output[:800],
                "error": last_error,
            })

            if debug:
                print(f"[step {step}] actions={[a['tool_name'] for a in action_summary]}")
                print(last_tool_output[:600])
                if last_error:
                    print(f"[step {step}] error={last_error}")

            if final_answer is not None:
                if step_error:
                    print(f"[step {step}] f_final_answer extracted despite step error: {step_error}")
                return final_answer, reasoning_trace, usage_totals

        last_error = f"Max steps ({max_steps}) reached without final answer."
        if debug:
            print(last_error)
        reasoning_trace.append(
            {
                "step": max_steps,
                "reasoning": "",
                "actions": [],
                "tool_output": "",
                "error": last_error,
            }
        )
        return "", reasoning_trace, usage_totals
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual table reasoning with iterative tool use."
    )
    parser.add_argument("--image", help="Path to table image")
    parser.add_argument("--question", help="Question about the table")
    parser.add_argument("--provider", default="openai", choices=["openai", "vllm", "ollama"],
                        help="Model provider backend")
    parser.add_argument("--model-name", default="gpt-4o",
                        help="Model name for the selected provider")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--max-steps", type=int, default=12, help="Max tool steps")
    parser.add_argument("--json-retries", type=int, default=2, help="Retries per step for JSON parsing")
    parser.add_argument("--attempts-per-question", type=int, default=1, help="Number of attempts per question")
    parser.add_argument("--rerank-answers", action="store_true", help="Use model to select best answer")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    parser.add_argument("--debug-empty", action="store_true", help="Print debug info on empty answers")
    parser.add_argument("--image-each-step", action="store_true", help="Render current table and include image each step")
    parser.add_argument(
        "--table-input-mode",
        default="image",
        choices=["text", "image", "multimodal"],
        help="How the reasoning loop should receive the working table at each step: serialized text, image, or image plus column names.",
    )
    parser.add_argument(
        "--table-input-modes",
        help="Comma-separated list of table input modes to run per question in one row, e.g. text,image,multimodal",
    )

    parser.add_argument("--wikitablequestions-path", help="Path to a WikiTableQuestions dataset on disk")
    parser.add_argument("--mmqa-path", help="Path to an MMQA dataset on disk")
    parser.add_argument("--mmtu-path", help="Path to an MMTU dataset on disk")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--start", type=int, default=0, help="Start index for dataset slicing")
    parser.add_argument("--end", type=int, default=1, help="End index for dataset slicing (exclusive)")
    parser.add_argument("--render-output", help="Where to write the rendered table image (or directory for step images)")
    parser.add_argument("--csv-output", help="Write question/answer to a CSV file")
    parser.add_argument(
        "--compact-csv",
        action="store_true",
        help="Omit large trace/selection fields from CSV output.",
    )

    args = parser.parse_args()

    config = {
        "provider": args.provider,
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.seed is not None:
        config["seed"] = args.seed

    selected_modes = _parse_table_input_modes(args.table_input_mode, args.table_input_modes)

    model = ModelFactory.create(config)

    samples = None
    if args.wikitablequestions_path:
        samples = load_wikitablequestions_dataset(
            dataset_path=args.wikitablequestions_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            use_sequential=False,
            start=args.start,
            end=args.end,
        )
    elif args.mmqa_path:
        samples = load_mmqa_dataset(
            dataset_path=args.mmqa_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
        )
    elif args.mmtu_path:
        samples = load_mmtu_dataset(
            dataset_path=args.mmtu_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
        )

    if samples is not None:
        import itertools
        samples, _peek = itertools.tee(samples)
        if next(_peek, None) is None:
            raise ValueError("No samples found in dataset.")

        csv_path = args.csv_output
        multi_mode_row = len(selected_modes) > 1
        if multi_mode_row:
            csv_fieldnames = ["question", "image_path", "table_id"]
            for mode in selected_modes:
                csv_fieldnames.extend(_mode_fieldnames(mode, args.compact_csv))
        else:
            trace_fieldnames = [
                "reasoning_trace",
                "all_answers",
                "all_reasoning_traces",
                "chosen_index",
                "selection_reason",
            ]
            csv_fieldnames = [
                "question",
                "answer",
                "image_path",
                "table_id",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "all_usage",
                "error",
                f"num_turns_{selected_modes[0]}",
                f"latency_seconds_{selected_modes[0]}",
            ]
            if not args.compact_csv:
                csv_fieldnames[7:7] = trace_fieldnames
        if csv_path:
            is_new = not os.path.exists(csv_path)
            csv_file = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=csv_fieldnames,
            )
            if is_new:
                writer.writeheader()
        else:
            csv_file = None
            writer = None

        try:
            for sample in samples:
                question = sample["question"]
                table_id = sample.get("table_id", "")
                if not question:
                    print("Skipping sample with empty question.")
                    continue
                image_path = ""
                if any(_uses_step_image(mode, args.image_each_step) for mode in selected_modes) or args.csv_output or args.render_output:
                    output_path = args.render_output or tempfile.mkstemp(suffix=".jpg")[1]
                    render_dataframe_to_image(sample["table"], output_path)
                    image_path = output_path
                row = {"question": question, "image_path": image_path, "table_id": table_id}
                mode_results: Dict[str, Dict[str, Any]] = {}
                for mode in selected_modes:
                    render_dir = None
                    if _uses_step_image(mode, args.image_each_step):
                        if args.render_output:
                            if args.render_output.lower().endswith((".png", ".jpg", ".jpeg")):
                                render_dir = os.path.dirname(args.render_output) or "."
                            else:
                                render_dir = args.render_output
                        else:
                            render_dir = os.path.dirname(image_path)
                    _t0 = time.time()
                    mode_result = _run_mode_reasoning(
                        model=model,
                        df=sample["table"],
                        question=question,
                        args=args,
                        table_input_mode=mode,
                        render_dir=render_dir,
                        base_usage=_empty_usage(),
                    )
                    mode_result["_latency"] = round(time.time() - _t0, 3)
                    mode_results[mode] = mode_result
                    if args.debug_empty and not (mode_result["answer"] or "").strip():
                        print(
                            json.dumps(
                                {
                                    "table_id": table_id,
                                    "question": question,
                                    "mode": mode,
                                    "error": mode_result["error"],
                                },
                                ensure_ascii=True,
                            )
                        )
                    print(f"[{mode}] {mode_result['answer']}")
                if writer is not None:

                    if multi_mode_row:
                        for mode in selected_modes:
                            mode_result = mode_results[mode]
                            row.update(
                                {
                                    f"answer_{mode}": mode_result["answer"],
                                    f"prompt_tokens_{mode}": mode_result["usage"].get("prompt_tokens", 0),
                                    f"completion_tokens_{mode}": mode_result["usage"].get("completion_tokens", 0),
                                    f"total_tokens_{mode}": mode_result["usage"].get("total_tokens", 0),
                                    f"all_usage_{mode}": json.dumps(mode_result["all_usage"], ensure_ascii=False),
                                    f"error_{mode}": mode_result["error"],
                                    f"num_turns_{mode}": len(mode_result["reasoning_trace"]),
                                    f"latency_seconds_{mode}": mode_result["_latency"],
                                }
                            )
                            if not args.compact_csv:
                                row.update(
                                    {
                                        f"reasoning_trace_{mode}": json.dumps(mode_result["reasoning_trace"], ensure_ascii=False),
                                        f"all_answers_{mode}": json.dumps(mode_result["all_answers"], ensure_ascii=False),
                                        f"all_reasoning_traces_{mode}": json.dumps(mode_result["all_reasoning_traces"], ensure_ascii=False),
                                        f"chosen_index_{mode}": mode_result["chosen_index"],
                                        f"selection_reason_{mode}": mode_result["selection_reason"],
                                    }
                                )
                    else:
                        mode_result = mode_results[selected_modes[0]]
                        row.update(
                            {
                                "answer": mode_result["answer"],
                                "prompt_tokens": mode_result["usage"].get("prompt_tokens", 0),
                                "completion_tokens": mode_result["usage"].get("completion_tokens", 0),
                                "total_tokens": mode_result["usage"].get("total_tokens", 0),
                                "all_usage": json.dumps(mode_result["all_usage"], ensure_ascii=False),
                                "error": mode_result["error"],
                                f"num_turns_{selected_modes[0]}": len(mode_result["reasoning_trace"]),
                                f"latency_seconds_{selected_modes[0]}": mode_result["_latency"],
                            }
                        )
                        if not args.compact_csv:
                            row.update(
                                {
                                    "reasoning_trace": json.dumps(mode_result["reasoning_trace"], ensure_ascii=False),
                                    "all_answers": json.dumps(mode_result["all_answers"], ensure_ascii=False),
                                    "all_reasoning_traces": json.dumps(mode_result["all_reasoning_traces"], ensure_ascii=False),
                                    "chosen_index": mode_result["chosen_index"],
                                    "selection_reason": mode_result["selection_reason"],
                                }
                            )
                    writer.writerow(row)
                    csv_file.flush()
        finally:
            if csv_file is not None:
                csv_file.close()
        return

    if not args.image or not args.question:
        raise ValueError("Provide --image and --question, or a dataset path.")

    df, base_usage = extract_table_from_image_with_usage(model, args.image, args.question)
    direct_results: Dict[str, Dict[str, Any]] = {}
    for mode in selected_modes:
        render_dir = None
        if _uses_step_image(mode, args.image_each_step):
            if args.render_output:
                if args.render_output.lower().endswith((".png", ".jpg", ".jpeg")):
                    render_dir = os.path.dirname(args.render_output) or "."
                else:
                    render_dir = args.render_output
            else:
                render_dir = os.path.dirname(args.image)
        direct_results[mode] = _run_mode_reasoning(
            model=model,
            df=df,
            question=args.question,
            args=args,
            table_input_mode=mode,
            render_dir=render_dir,
            base_usage=base_usage,
        )

    if len(selected_modes) == 1:
        result = direct_results[selected_modes[0]]
        print(result["answer"])
        print(json.dumps(result["usage"], ensure_ascii=True))
    else:
        payload = {}
        for mode in selected_modes:
            result = direct_results[mode]
            payload[mode] = {
                "answer": result["answer"],
                "usage": result["usage"],
                "error": result["error"],
            }
        print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
