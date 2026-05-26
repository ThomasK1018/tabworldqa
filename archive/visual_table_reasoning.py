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
) -> List[Dict[str, Any]]:
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

    samples = []
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
            samples.append(
                {
                    "question": normalize_text(example.get("question", "")),
                    "table": df,
                    "df_og": df_og,
                    "answer": ", ".join(example.get("answers", [])) if example.get("answers") else "",
                    "table_id": example.get("id", f"sample_{idx}"),
                    "original_index": idx,
                }
            )
    return samples


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


def load_tabmwp_dataset(
    dataset_path: str,
    n_samples: int = 1,
    random_seed: int = 42,
    start: int = 0,
    end: int | None = None,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Load TabMWP dataset from PromptPG-style problems_{split}.json.

    Args:
        dataset_path: Directory containing problems_train.json / _dev.json / _test.json
                      (i.e. the tabmwp/ subdirectory inside PromptPG/data/).
        split: One of "train", "dev", "dev1k", "test", "test1k".
    """
    fpath = os.path.join(dataset_path, f"problems_{split}.json")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"TabMWP file not found: {fpath}")

    with open(fpath, "r") as f:
        data = json.load(f)

    items = list(data.items())
    total_examples = len(items)

    random.seed(random_seed)
    k = total_examples if n_samples < 0 else min(int(n_samples), total_examples)
    indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    samples = []
    for idx in indices:
        pid, example = items[idx]

        table_for_pd = example.get("table_for_pd", {})
        if table_for_pd and isinstance(table_for_pd, dict):
            try:
                df = pd.DataFrame(table_for_pd)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        if not df.empty:
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

        question = normalize_text(example.get("question", ""))
        table_title = example.get("table_title", "")
        choices = example.get("choices")
        unit = example.get("unit", "")

        if table_title:
            question = f"[Table: {table_title}] {question}"
        if choices:
            opts = ", ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            question = f"{question} Choices: {opts}"
        if unit:
            question = f"{question} (Answer unit: {unit})"

        samples.append(
            {
                "question": question,
                "table": df,
                "df_og": df_og,
                "answer": normalize_text(example.get("answer", "")),
                "table_id": str(pid),
                "original_index": idx,
            }
        )
    return samples


def load_fetaqa_dataset(
    dataset_path: str,
    n_samples: int = 1,
    random_seed: int = 42,
    start: int = 0,
    end: int | None = None,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Load FeTaQA dataset from fetaQA-v1_{split}.json(l).

    Args:
        dataset_path: Directory containing fetaQA-v1_train.json / _dev.json / _test.json.
        split: One of "train", "dev", "test".
    """
    for ext in (".jsonl", ".json"):
        fpath = os.path.join(dataset_path, f"fetaQA-v1_{split}{ext}")
        if os.path.exists(fpath):
            break
    else:
        raise FileNotFoundError(
            f"FeTaQA file not found in {dataset_path} (tried fetaQA-v1_{split}.jsonl/.json)"
        )

    data = []
    with open(fpath, "r") as f:
        if fpath.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            data = json.load(f)

    total_examples = len(data)
    random.seed(random_seed)
    k = total_examples if n_samples < 0 else min(int(n_samples), total_examples)
    indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    samples = []
    for idx in indices:
        example = data[idx]
        table_array = example.get("table_array", [])
        if len(table_array) < 2:
            continue
        header = [str(c) for c in table_array[0]]
        rows = [[str(cell) for cell in row] for row in table_array[1:]]
        n_cols = len(header)
        rows = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
        df = pd.DataFrame(rows, columns=header)

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

        feta_id = example.get("feta_id", f"fetaqa_{idx}")
        samples.append(
            {
                "question": normalize_text(example.get("question", "")),
                "table": df,
                "df_og": df_og,
                "answer": normalize_text(example.get("answer", "")),
                "table_id": str(feta_id),
                "original_index": idx,
            }
        )
    return samples


def load_tatqa_dataset(
    dataset_path: str,
    n_samples: int = -1,
    random_seed: int = 42,
    start: int = 0,
    end: int | None = None,
    split: str = "dev",
    answer_from: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load TAT-QA dataset from tatqa_dataset_{split}.json.

    Flattens to one sample per question. Optionally filters by answer_from
    ("table", "text", "table-text").

    Args:
        dataset_path: Root TAT-QA directory (containing dataset_raw/).
        split: One of "train", "dev", "test", "test_gold".
        answer_from: If set, only include questions with this answer_from value.
    """
    fpath = os.path.join(dataset_path, "dataset_raw", f"tatqa_dataset_{split}.json")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"TAT-QA file not found: {fpath}")

    with open(fpath, "r") as f:
        data = json.load(f)

    all_samples: List[Dict[str, Any]] = []
    for doc_idx, doc in enumerate(data):
        table_data = doc.get("table", {})
        table_rows = table_data.get("table", [])
        table_uid = table_data.get("uid", f"doc_{doc_idx}")

        paragraphs = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
        context_text = "\n".join(str(p["text"]) for p in paragraphs if p.get("text"))

        if len(table_rows) >= 2:
            header = [str(c) for c in table_rows[0]]
            rows = [[str(cell) for cell in row] for row in table_rows[1:]]
            n_cols = len(header)
            rows = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
            df = pd.DataFrame(rows, columns=header)
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
        elif len(table_rows) == 1:
            df = pd.DataFrame(columns=[str(c) for c in table_rows[0]])
        else:
            df = pd.DataFrame()

        df_og = df.copy()

        for q in doc.get("questions", []):
            if answer_from is not None and q.get("answer_from") != answer_from:
                continue
            question = normalize_text(q.get("question", ""))
            answers = q.get("answer", [])
            if isinstance(answers, list):
                answer_str = ", ".join(str(a) for a in answers) if answers else ""
            else:
                answer_str = str(answers) if answers is not None else ""
            scale = q.get("scale", "")
            if scale:
                question = f"{question} (Scale: {scale})"
            question_full = f"Context:\n{context_text}\n\nQuestion: {question}" if context_text else question
            all_samples.append(
                {
                    "question": question_full,
                    "table": df.copy(),
                    "df_og": df_og.copy(),
                    "answer": answer_str,
                    "table_id": f"{table_uid}_q{q.get('order', 0)}",
                    "original_index": len(all_samples),
                }
            )

    total_examples = len(all_samples)
    random.seed(random_seed)
    if n_samples < 0:
        indices = list(range(total_examples))
    else:
        k = min(int(n_samples), total_examples)
        indices = random.sample(range(total_examples), k)
    indices = indices[start:end]
    return [all_samples[i] for i in indices]


def _hitab_leaf_labels(node: Dict, ancestors: tuple = ()) -> List[Tuple[str, Optional[int]]]:
    """Recursively collect (label, line_idx) for every leaf in a HiTab KG tree node."""
    val = node.get("value", "")
    if val not in ("<TOP>", "<LEFT>"):
        ancestors = ancestors + (val,)
    children = node.get("children_dict", [])
    if not children:
        label = " | ".join(str(a) for a in ancestors) if ancestors else val
        return [(label, node.get("line_idx"))]
    result = []
    for child in children:
        result.extend(_hitab_leaf_labels(child, ancestors))
    return result


def load_hitab_dataset(
    dataset_path: str,
    n_samples: int = -1,
    random_seed: int = 42,
    start: int = 0,
    end: int | None = None,
    split: str = "dev",
) -> List[Dict[str, Any]]:
    """Load HiTab dataset from {split}_samples.jsonl + processed_input/tables.jsonl.

    Args:
        dataset_path: Root HiTab directory (containing data/).
        split: One of "train", "dev", "test".
    """
    tables_path = os.path.join(dataset_path, "data", "processed_input", "tables.jsonl")
    if not os.path.exists(tables_path):
        raise FileNotFoundError(f"HiTab tables file not found: {tables_path}")

    with open(tables_path) as f:
        table_lookup: Dict[str, Any] = {(d := json.loads(l))["name"]: d for l in f}

    split_path = os.path.join(dataset_path, "data", f"{split}_samples.jsonl")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"HiTab split file not found: {split_path}")

    with open(split_path) as f:
        raw_samples = [json.loads(l) for l in f]

    all_samples: List[Dict[str, Any]] = []
    for idx, sample in enumerate(raw_samples):
        table_id = sample.get("table_id", "")
        table_data = table_lookup.get(table_id)
        if table_data is None:
            continue

        kg = table_data["kg"]
        title = kg.get("title", "")
        data_rows = kg.get("data", [])

        top_label_idx = _hitab_leaf_labels(kg["top_root"])
        left_label_idx = _hitab_leaf_labels(kg["left_root"])
        col_names = [label for label, _ in top_label_idx]

        rows, row_labels = [], []
        for label, line_idx in left_label_idx:
            if line_idx is not None and line_idx < len(data_rows):
                row = [str(cell.get("value", "")) for cell in data_rows[line_idx]]
                rows.append(row)
                row_labels.append(label)

        if not rows:
            continue

        n_cols = len(col_names)
        rows = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
        all_cols = [""] + col_names
        df = pd.DataFrame([[rl] + r for rl, r in zip(row_labels, rows)], columns=all_cols)

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

        question = normalize_text(sample.get("question", ""))
        if title:
            question = f"[Table: {title}] {question}"
        answers = sample.get("answer", [])
        answer_str = ", ".join(str(a) for a in answers) if isinstance(answers, list) else str(answers or "")

        all_samples.append(
            {
                "question": question,
                "table": df,
                "df_og": df_og,
                "answer": answer_str,
                "table_id": table_id,
                "original_index": idx,
            }
        )

    total_examples = len(all_samples)
    random.seed(random_seed)
    if n_samples < 0:
        indices = list(range(total_examples))
    else:
        k = min(int(n_samples), total_examples)
        indices = random.sample(range(total_examples), k)
    indices = indices[start:end]
    return [all_samples[i] for i in indices]


def load_tabfact_dataset(
    dataset_path: str,
    n_samples: int = -1,
    random_seed: int = 42,
    start: int = 0,
    end: int | None = None,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Load Table-Fact-Checking (TabFact) dataset.

    Label 1 = true, 0 = false. Framed as a true/false verification task.

    Args:
        dataset_path: Root TabFact directory (containing processed_datasets/ and data/all_csv/).
        split: One of "train", "dev", "test", "simple_test", "complex_test", "small_test".
    """
    tsv_path = os.path.join(
        dataset_path, "processed_datasets", "tsv_data_horizontal", f"{split}.tsv"
    )
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TabFact TSV not found: {tsv_path}")

    csv_dir = os.path.join(dataset_path, "data", "all_csv")
    raw = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["table_id", "num_cols", "col_name", "row_text", "statement", "label"],
    )

    total_examples = len(raw)
    random.seed(random_seed)
    if n_samples < 0:
        indices = list(range(total_examples))
    else:
        k = min(int(n_samples), total_examples)
        indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    samples: List[Dict[str, Any]] = []
    for idx in indices:
        row = raw.iloc[idx]
        table_id = str(row["table_id"])
        csv_path = os.path.join(csv_dir, table_id)
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, sep="#")
        except Exception:
            continue
        if df.empty:
            continue

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

        answer_str = "true" if int(row["label"]) == 1 else "false"
        question = f"Is the following statement true or false about the table?\nStatement: {normalize_text(row['statement'])}"

        samples.append(
            {
                "question": question,
                "table": df,
                "df_og": df_og,
                "answer": answer_str,
                "table_id": table_id,
                "original_index": idx,
            }
        )
    return samples


def load_infotabs_dataset(
    dataset_path: str,
    n_samples: int = -1,
    random_seed: int = 42,
    start: int = 0,
    end: int | None = None,
    split: str = "dev",
) -> List[Dict[str, Any]]:
    """Load InfoTabs dataset from infotabs_{split}.tsv + tables/json/.

    Label E = entailment, N = neutral, C = contradiction.
    The infobox JSON is converted to a two-column Property/Value DataFrame.

    Args:
        dataset_path: Root infotabs directory (containing data/).
        split: One of "train", "dev", "test_alpha1", "test_alpha2", "test_alpha3".
    """
    tsv_path = os.path.join(dataset_path, "data", "maindata", f"infotabs_{split}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"InfoTabs TSV not found: {tsv_path}")

    json_dir = os.path.join(dataset_path, "data", "tables", "json")
    raw = pd.read_csv(tsv_path, sep="\t")

    total_examples = len(raw)
    random.seed(random_seed)
    if n_samples < 0:
        indices = list(range(total_examples))
    else:
        k = min(int(n_samples), total_examples)
        indices = random.sample(range(total_examples), k)
    indices = indices[start:end]

    label_map = {"E": "entailment", "N": "neutral", "C": "contradiction"}

    samples: List[Dict[str, Any]] = []
    for idx in indices:
        row = raw.iloc[idx]
        table_id = str(row["table_id"])
        json_path = os.path.join(json_dir, f"{table_id}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            infobox = json.load(f)

        title = infobox.pop("title", None)
        title_str = title[0] if isinstance(title, list) and title else str(title) if title else ""

        rows = []
        for prop, vals in infobox.items():
            value_str = ", ".join(str(v) for v in vals) if isinstance(vals, list) else str(vals)
            rows.append([normalize_text(prop), normalize_text(value_str)])

        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["Property", "Value"])
        df = df.applymap(normalize_text)
        df_og = df.copy()

        hypothesis = normalize_text(row["hypothesis"])
        answer_str = label_map.get(str(row["label"]).strip(), str(row["label"]).strip())
        question = (
            f"[Table: {title_str}] " if title_str else ""
        ) + f"Does the table entail, contradict, or is neutral to the following statement?\nStatement: {hypothesis}"

        samples.append(
            {
                "question": question,
                "table": df,
                "df_og": df_og,
                "answer": answer_str,
                "table_id": table_id,
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
    try:
        for step in range(1, max_steps + 1):
            df_preview = env["df"].head(15).to_dict(orient="records")
            tool_prompt = (
                "You are a table reasoning agent. Use tools to transform the table and answer the question.\n"
                "Respond with JSON only using the schema:\n"
                "{"
                "\"reasoning\": str, "
                "\"tool_name\": str, "
                "\"tool_args\": object"
                "}\n"
                "If you can answer, set tool_name to f_final_answer with tool_args {\"answer\": \"...\"}.\n\n"
                "If you need to revert the table, use tool f_retrieve_original_df to restore df from df_og.\n\n"
                f"Question: {question}\n"
                f"Table shape: {env['df'].shape}\n"
                f"Columns: {env['df'].columns.tolist()}\n"
                f"Preview rows (JSON): {json.dumps(df_preview, ensure_ascii=True)}\n"
                f"Last tool output: {last_tool_output[:400] if last_tool_output else 'N/A'}\n"
                f"Last error: {last_error if last_error else 'N/A'}\n\n"
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
                if debug:
                    usage = response.get("usage") or {}
                    finish_reason = response.get("finish_reason")
                    completion_tokens = usage.get("completion_tokens", 0)
                    if finish_reason == "length" or (max_tokens and completion_tokens and completion_tokens >= max_tokens):
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
                    # Corrective retry: explicitly request valid JSON tool output.
                    corrective_prompt = (
                        "Your previous response was not valid JSON.\n"
                        "Respond with JSON ONLY using this schema:\n"
                        "{"
                        "\"reasoning\": str, "
                        "\"tool_name\": str, "
                        "\"tool_args\": object"
                        "}\n"
                        "If you can answer, set tool_name to f_final_answer with tool_args {\"answer\": \"...\"}.\n"
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

            tool_name = str(payload.get("tool_name") or "").strip()
            tool_args = payload.get("tool_args") or {}
            print(payload)
            if not tool_name:
                last_error = f"Missing tool_name at step {step}."
                if debug:
                    print(last_error)
                reasoning_trace.append(
                    {
                        "step": step,
                        "reasoning": payload.get("reasoning", ""),
                        "tool_name": "",
                        "tool_args": tool_args,
                        "tool_output": "",
                        "error": last_error,
                    }
                )
                continue

            if tool_name == "f_final_answer":
                answer = tool_args.get("answer", "")
                reasoning_trace.append(
                    {
                        "step": step,
                        "reasoning": payload.get("reasoning", ""),
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": "",
                        "error": "",
                    }
                )
                return str(answer), reasoning_trace, usage_totals

            tool = tool_map.get(tool_name)
            if tool is None:
                last_error = f"Unknown tool: {tool_name}"
                last_tool_output = ""
                if debug:
                    print(last_error)
                reasoning_trace.append(
                    {
                        "step": step,
                        "reasoning": payload.get("reasoning", ""),
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": "",
                        "error": last_error,
                    }
                )
                continue

            if not isinstance(tool_args, dict):
                last_error = f"tool_args must be an object, got: {type(tool_args)}"
                last_tool_output = ""
                if debug:
                    print(last_error)
                reasoning_trace.append(
                    {
                        "step": step,
                        "reasoning": payload.get("reasoning", ""),
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": "",
                        "error": last_error,
                    }
                )
                continue

            try:
                code = tool._run(**tool_args)
            except Exception as exc:
                last_error = f"Tool args error: {exc}"
                last_tool_output = ""
                if debug:
                    print(last_error)
                reasoning_trace.append(
                    {
                        "step": step,
                        "reasoning": payload.get("reasoning", ""),
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": "",
                        "error": last_error,
                    }
                )
                continue

            output, err = _capture_exec(code, env)
            last_tool_output = output.strip()
            last_error = str(err) if err else ""
            reasoning_trace.append(
                {
                    "step": step,
                    "reasoning": payload.get("reasoning", ""),
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_output": last_tool_output[:800],
                    "error": last_error,
                }
            )

            if debug:
                print(f"[step {step}] tool={tool_name} args={tool_args}")
                print(last_tool_output[:600])
                if last_error:
                    print(f"[step {step}] error={last_error}")

        last_error = f"Max steps ({max_steps}) reached without final answer."
        if debug:
            print(last_error)
        reasoning_trace.append(
            {
                "step": max_steps,
                "reasoning": "",
                "tool_name": "",
                "tool_args": {},
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
        help="Comma-separated list of table input modes to run per question in one row, e.g. text,image,multimodal,text_json,text_latex",
    )

    parser.add_argument("--wikitablequestions-path", help="Path to a WikiTableQuestions dataset on disk")
    parser.add_argument("--mmqa-path", help="Path to an MMQA dataset on disk")
    parser.add_argument("--mmtu-path", help="Path to an MMTU dataset on disk")
    parser.add_argument("--fetaqa-path", help="Path to FeTaQA dataset directory")
    parser.add_argument("--fetaqa-split", default="test", choices=["train", "validation", "test"], help="FeTaQA split")
    parser.add_argument("--tabmwp-path", help="Path to TabMWP dataset directory")
    parser.add_argument("--tabmwp-split", default="test", choices=["train", "dev", "test"], help="TabMWP split")
    parser.add_argument("--tatqa-path", help="Path to TAT-QA dataset directory")
    parser.add_argument("--tatqa-split", default="dev", choices=["train", "dev", "test"], help="TAT-QA split")
    parser.add_argument("--tatqa-answer-from", default=None, choices=["table", "text", "table-text"], help="Filter TAT-QA by answer_from field")
    parser.add_argument("--hitab-path", help="Path to HiTab dataset directory")
    parser.add_argument("--hitab-split", default="dev", choices=["train", "dev", "test"], help="HiTab split")
    parser.add_argument("--tabfact-path", help="Path to Table-Fact-Checking dataset directory")
    parser.add_argument("--tabfact-split", default="test", choices=["train", "val", "test"], help="TabFact split")
    parser.add_argument("--infotabs-path", help="Path to InfoTabs dataset directory")
    parser.add_argument("--infotabs-split", default="dev", choices=["train", "dev", "test", "test_alpha1", "test_alpha2", "test_alpha3"], help="InfoTabs split")
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
    elif args.fetaqa_path:
        samples = load_fetaqa_dataset(
            dataset_path=args.fetaqa_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
            split=args.fetaqa_split,
        )
    elif args.tabmwp_path:
        samples = load_tabmwp_dataset(
            dataset_path=args.tabmwp_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
            split=args.tabmwp_split,
        )
    elif args.tatqa_path:
        samples = load_tatqa_dataset(
            dataset_path=args.tatqa_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
            split=args.tatqa_split,
            answer_from=args.tatqa_answer_from,
        )
    elif args.hitab_path:
        samples = load_hitab_dataset(
            dataset_path=args.hitab_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
            split=args.hitab_split,
        )
    elif args.tabfact_path:
        samples = load_tabfact_dataset(
            dataset_path=args.tabfact_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
            split=args.tabfact_split,
        )
    elif args.infotabs_path:
        samples = load_infotabs_dataset(
            dataset_path=args.infotabs_path,
            n_samples=-1,
            random_seed=args.sample_seed,
            start=args.start,
            end=args.end,
            split=args.infotabs_split,
        )

    if samples is not None:
        if not samples:
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
