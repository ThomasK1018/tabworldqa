from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import re


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _cols_exist(df: pd.DataFrame, cols: List[str]) -> bool:
    if not cols:
        return False
    return all(c in df.columns for c in cols)


def _df_unchanged(prev_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
    try:
        return prev_df.equals(new_df)
    except Exception:
        return False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _question_overlap_reward(question: str, cols: List[str]) -> float:
    if not question or not cols:
        return 0.0
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0.0
    col_tokens: List[str] = []
    for c in cols:
        col_tokens.extend(_tokenize(c))
    if not col_tokens:
        return 0.0
    overlap = len(set(col_tokens) & q_tokens)
    return float(overlap) / float(len(q_tokens))


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() >= 0.8


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def compute_tool_reward(
    tool_name: str,
    tool_args: Dict[str, Any],
    prev_df: pd.DataFrame,
    new_df: pd.DataFrame,
    last_error: str,
    tool_output: str,
    df_og: Optional[pd.DataFrame] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    components: Dict[str, float] = {}
    if last_error:
        return {"scalar": 0.0, "components": {"error": 1.0}}

    prev_rows, prev_cols = prev_df.shape
    new_rows, new_cols = new_df.shape

    if tool_name == "f_select_column":
        cols = _as_list(tool_args.get("column_names"))
        components["cols_exist"] = 1.0 if _cols_exist(prev_df, cols) else 0.0
        components["cols_subset"] = 1.0 if set(new_df.columns).issubset(set(prev_df.columns)) else 0.0
        components["cols_reduced_or_same"] = 1.0 if new_cols <= prev_cols else 0.0
        components["rows_unchanged"] = 1.0 if new_rows == prev_rows else 0.0
        if question:
            components["question_overlap"] = _question_overlap_reward(question, cols)
        if question and any(k in question.lower() for k in ["avg", "average", "sum", "total", "max", "min", "count"]):
            numeric_cols = [c for c in cols if c in prev_df.columns and _is_numeric_series(prev_df[c])]
            components["numeric_cols_kept"] = 1.0 if numeric_cols else 0.0
    elif tool_name == "f_select_row":
        indices = tool_args.get("indices")
        if indices is None:
            components["indices_valid"] = 1.0
        else:
            try:
                indices_ok = all(0 <= int(i) < prev_rows for i in indices)
            except Exception:
                indices_ok = False
            components["indices_valid"] = 1.0 if indices_ok else 0.0
        components["rows_nonincrease"] = 1.0 if new_rows <= prev_rows else 0.0
        components["cols_unchanged"] = 1.0 if list(new_df.columns) == list(prev_df.columns) else 0.0
        lookup = bool(tool_args.get("lookup", False))
        if lookup:
            components["lookup_no_change"] = 1.0 if _df_unchanged(prev_df, new_df) else 0.0
        components["nonempty"] = 1.0 if new_rows > 0 else 0.0
        if question and any(k in question.lower() for k in ["which", "what", "who", "list"]):
            components["expected_match"] = 1.0 if new_rows > 0 else 0.0
        if isinstance(tool_args.get("condition"), str):
            components["condition_present"] = 1.0
    elif tool_name == "f_sort_by":
        cols = _as_list(tool_args.get("columns"))
        components["cols_exist"] = 1.0 if _cols_exist(prev_df, cols) else 0.0
        components["rows_unchanged"] = 1.0 if new_rows == prev_rows else 0.0
        components["cols_unchanged"] = 1.0 if list(new_df.columns) == list(prev_df.columns) else 0.0
        if cols and cols[0] in new_df.columns:
            order = str(tool_args.get("order", "asc")).lower()
            series = new_df[cols[0]]
            if order == "desc":
                components["order_applied"] = 1.0 if series.is_monotonic_decreasing else 0.0
            else:
                components["order_applied"] = 1.0 if series.is_monotonic_increasing else 0.0
        if question and any(k in question.lower() for k in ["highest", "lowest", "largest", "smallest", "top", "bottom"]):
            components["sorting_relevant"] = 1.0
    elif tool_name == "f_aggregate":
        show_stats_only = bool(tool_args.get("show_stats_only", False))
        columns_ops = tool_args.get("columns_ops")
        column = tool_args.get("column")
        cols: List[str] = []
        if columns_ops:
            if isinstance(columns_ops, dict):
                cols = list(columns_ops.keys())
            elif isinstance(columns_ops, list):
                for item in columns_ops:
                    if isinstance(item, dict):
                        cols.extend(list(item.keys()))
        elif column:
            cols = [str(column)]
        components["cols_exist"] = 1.0 if _cols_exist(prev_df, [c for c in cols if c]) else 0.0
        if show_stats_only:
            components["df_unchanged"] = 1.0 if _df_unchanged(prev_df, new_df) else 0.0
        else:
            components["rows_nonincrease"] = 1.0 if new_rows <= prev_rows else 0.0
            components["nonempty"] = 1.0 if new_rows > 0 and new_cols > 0 else 0.0
            if tool_args.get("group_by"):
                components["groupby_reduced"] = 1.0 if new_rows <= prev_rows else 0.0
        if question and any(k in question.lower() for k in ["sum", "avg", "average", "mean", "count", "max", "min"]):
            components["aggregate_relevant"] = 1.0
    elif tool_name == "f_compute_column":
        new_col = str(tool_args.get("new_col", ""))
        col_A = str(tool_args.get("col_A", ""))
        col_B = tool_args.get("col_B")
        components["col_A_exist"] = 1.0 if col_A in prev_df.columns else 0.0
        if isinstance(col_B, str):
            components["col_B_exist"] = 1.0 if col_B in prev_df.columns else 0.0
        else:
            components["col_B_exist"] = 1.0
        components["new_col_added"] = 1.0 if new_col and new_col in new_df.columns else 0.0
        components["cols_plus_one"] = 1.0 if new_cols == prev_cols + 1 else 0.0
        components["rows_unchanged"] = 1.0 if new_rows == prev_rows else 0.0
        if new_col in new_df.columns:
            components["new_col_nonempty"] = 1.0 if new_df[new_col].notna().any() else 0.0
            if new_df[new_col].isna().mean() < 1.0:
                components["new_col_valid"] = 1.0
    elif tool_name == "f_distinct_count":
        cols = _as_list(tool_args.get("columns") or tool_args.get("column"))
        components["cols_exist"] = 1.0 if _cols_exist(prev_df, cols) else 0.0
        components["df_unchanged"] = 1.0 if _df_unchanged(prev_df, new_df) else 0.0
        if tool_output:
            components["has_output"] = 1.0
    elif tool_name == "f_calculate_average":
        col = str(tool_args.get("column_name", ""))
        group_by = str(tool_args.get("group_by_column", ""))
        filter_col = tool_args.get("filter_column")
        components["col_exist"] = 1.0 if col in prev_df.columns else 0.0
        components["group_by_exist"] = 1.0 if group_by in prev_df.columns else 0.0
        if filter_col:
            components["filter_col_exist"] = 1.0 if str(filter_col) in prev_df.columns else 0.0
        components["nonempty"] = 1.0 if new_rows > 0 and new_cols > 0 else 0.0
        components["rows_nonincrease"] = 1.0 if new_rows <= prev_rows else 0.0
        if question and any(k in question.lower() for k in ["avg", "average", "mean"]):
            components["avg_relevant"] = 1.0
    elif tool_name == "f_get_data_info":
        components["df_unchanged"] = 1.0 if _df_unchanged(prev_df, new_df) else 0.0
        components["has_output"] = 1.0 if tool_output else 0.0
    elif tool_name == "print_table":
        components["df_unchanged"] = 1.0 if _df_unchanged(prev_df, new_df) else 0.0
        components["has_output"] = 1.0 if tool_output else 0.0
    elif tool_name == "f_lookup":
        components["df_unchanged"] = 1.0 if _df_unchanged(prev_df, new_df) else 0.0
        components["has_output"] = 1.0 if tool_output else 0.0
    elif tool_name == "f_process_datetime":
        cols = _as_list(tool_args.get("columns"))
        components["cols_exist"] = 1.0 if _cols_exist(prev_df, cols) else 0.0
        new_cols = tool_args.get("new_columns")
        if new_cols:
            components["cols_increased"] = 1.0 if new_cols is not None and new_df.shape[1] >= prev_cols else 0.0
        else:
            components["cols_unchanged"] = 1.0 if new_df.shape[1] == prev_cols else 0.0
        components["rows_unchanged"] = 1.0 if new_rows == prev_rows else 0.0
        if question and any(k in question.lower() for k in ["year", "month", "day", "date", "time"]):
            components["datetime_relevant"] = 1.0
    elif tool_name == "f_string_operation":
        cols = _as_list(tool_args.get("columns"))
        components["cols_exist"] = 1.0 if _cols_exist(prev_df, cols) else 0.0
        other_cols = _as_list(tool_args.get("other_columns"))
        if other_cols:
            components["other_cols_exist"] = 1.0 if _cols_exist(prev_df, other_cols) else 0.0
        new_cols = tool_args.get("new_columns")
        if new_cols:
            n_new = len(_as_list(new_cols))
            components["cols_increased"] = 1.0 if new_df.shape[1] >= prev_cols + n_new else 0.0
        else:
            components["cols_unchanged"] = 1.0 if new_df.shape[1] == prev_cols else 0.0
        components["rows_unchanged"] = 1.0 if new_rows == prev_rows else 0.0
        if question and any(k in question.lower() for k in ["name", "string", "text", "case", "format"]):
            components["string_relevant"] = 1.0
    elif tool_name == "f_retrieve_original_df":
        if df_og is not None:
            components["restored_df_og"] = 1.0 if _df_unchanged(df_og, new_df) else 0.0
        else:
            components["restored_df_og"] = 0.0
        if df_og is not None:
            components["was_different_before"] = 1.0 if not _df_unchanged(df_og, prev_df) else 0.0
    elif tool_name == "f_rename_column":
        old_col = tool_args.get("old_col") or tool_args.get("old_column") or tool_args.get("column_name")
        new_col = tool_args.get("new_col") or tool_args.get("new_column") or tool_args.get("new_column_name")
        components["old_col_exist"] = 1.0 if old_col in prev_df.columns else 0.0
        components["new_col_added"] = 1.0 if new_col in new_df.columns else 0.0
        components["cols_unchanged"] = 1.0 if new_cols == prev_cols else 0.0
        if old_col and new_col:
            components["renamed"] = 1.0 if old_col not in new_df.columns and new_col in new_df.columns else 0.0
    elif tool_name == "f_final_answer":
        answer = str(tool_args.get("answer", "")).strip()
        components["nonempty_answer"] = 1.0 if answer else 0.0
        if question and answer:
            components["not_copy_question"] = 1.0 if answer.lower() != question.lower() else 0.0
    else:
        components["unknown_tool"] = 1.0

    if not components:
        return {"scalar": 0.0, "components": {}}
    scalar = float(sum(components.values()) / float(len(components)))
    return {"scalar": scalar, "components": components}
