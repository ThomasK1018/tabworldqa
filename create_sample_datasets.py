#!/usr/bin/env python3
"""
create_sample_datasets.py
-------------------------
Extract 20 samples from each benchmark dataset and save to:
    tabworldqa/sample_datasets/{dataset_name}/samples.jsonl

Each JSONL record:
    question       str   — question text (with context / choices prepended as needed)
    answer         str   — ground-truth answer
    table_id       str   — dataset-specific table identifier
    original_index int   — position in the source dataset
    columns        list  — ordered column names of the table
    table          list  — list of row dicts (DataFrame.to_dict('records'))

Usage:
    conda run -n tabworldqa_env python create_sample_datasets.py
"""

import json
import os
import random
from typing import Any, Dict, List

import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "sample_datasets")
N        = 20

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def remove_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    total_like = df.apply(
        lambda col: col.astype(str).str.lower().str.strip()
        .isin({"total", "totals", "sum", "grand total"})
    ).any(axis=1)
    return df[~total_like].reset_index(drop=True)


def _norm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard column / cell normalisation used across all loaders."""
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
    df = df.rename(columns=lambda c: c.replace('"', " ") if isinstance(c, str) else c).replace(
        '"', " ", regex=True
    )
    df = df.rename(columns=lambda c: c.replace("'", "`") if isinstance(c, str) else c).replace(
        "'", "\\'", regex=True
    )
    df = df.rename(
        columns=lambda c: c.replace("\n", "").replace("\\n", "") if isinstance(c, str) else c
    ).replace({r"\n ": ",", r"\\n df": ","}, regex=True)
    df = df.applymap(normalize_text)
    df = remove_total_rows(df)
    return df


def save_samples(name: str, samples: List[Dict[str, Any]]) -> None:
    out_dir  = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "samples.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            record = {
                "question":       s["question"],
                "answer":         s["answer"],
                "table_id":       s["table_id"],
                "original_index": s["original_index"],
                "columns":        list(s["table"].columns),
                "table":          s["table"].to_dict("records"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{name}] saved {len(samples)} samples → {out_path}")


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_wikitablequestions(path: str) -> List[Dict]:
    from datasets import load_from_disk
    dataset = load_from_disk(path)
    split = dataset["train"]
    indices = list(range(min(N, len(split))))
    samples = []
    for idx in indices:
        ex = split[idx]
        td = ex.get("table", {})
        if "header" not in td or "rows" not in td:
            continue
        df = _norm_df(pd.DataFrame(td["rows"], columns=td["header"]))
        samples.append({
            "question":       normalize_text(ex.get("question", "")),
            "table":          df,
            "answer":         ", ".join(ex.get("answers", [])) if ex.get("answers") else "",
            "table_id":       ex.get("id", f"sample_{idx}"),
            "original_index": idx,
        })
    return samples


def load_mmqa(path: str) -> List[Dict]:
    from datasets import load_from_disk
    base_df = pd.read_csv(os.path.join(path, "MMQA_dataset.csv"))
    random.seed(1018)
    indices = random.sample(range(len(base_df)), min(N, len(base_df)))
    samples = []
    for idx in indices:
        ex = base_df.iloc[idx]
        df = _norm_df(
            load_from_disk(os.path.join(path, "MMQA_datasets", ex["table_id"])).to_pandas()
        )
        samples.append({
            "question":       normalize_text(ex["question"]),
            "table":          df,
            "answer":         str(ex.get("correct_answer", "")),
            "table_id":       ex["table_id"],
            "original_index": idx,
        })
    return samples


def load_mmtu(path: str) -> List[Dict]:
    from datasets import load_from_disk
    base_df = pd.read_excel(os.path.join(path, "mmtu_record.xlsx"))
    random.seed(1018)
    indices = random.sample(range(len(base_df)), min(N, len(base_df)))
    samples = []
    for idx in indices:
        ex = base_df.iloc[idx]
        df = _norm_df(
            load_from_disk(os.path.join(path, "MMTU_datasets", str(ex["id"]))).to_pandas()
        )
        samples.append({
            "question":       normalize_text(ex["question"]),
            "table":          df,
            "answer":         str(ex.get("ground_truth", "")),
            "table_id":       str(ex["id"]),
            "original_index": idx,
        })
    return samples


def load_hitab(path: str) -> List[Dict]:
    tables_path = os.path.join(path, "data", "processed_input", "tables.jsonl")
    with open(tables_path) as f:
        table_lookup: Dict[str, Any] = {(d := json.loads(l))["name"]: d for l in f}

    split_path = os.path.join(path, "data", "dev_samples.jsonl")
    with open(split_path) as f:
        raw_samples = [json.loads(l) for l in f]

    def _leaf_labels(node, ancestors=()):
        val = node.get("value", "")
        if val not in ("<TOP>", "<LEFT>"):
            ancestors = ancestors + (val,)
        children = node.get("children_dict", [])
        if not children:
            return [(" | ".join(str(a) for a in ancestors) if ancestors else val, node.get("line_idx"))]
        result = []
        for child in children:
            result.extend(_leaf_labels(child, ancestors))
        return result

    samples: List[Dict] = []
    for idx, sample in enumerate(raw_samples):
        if len(samples) >= N:
            break
        table_id  = sample.get("table_id", "")
        table_data = table_lookup.get(table_id)
        if table_data is None:
            continue
        kg        = table_data["kg"]
        title     = kg.get("title", "")
        data_rows = kg.get("data", [])
        top_labels  = _leaf_labels(kg["top_root"])
        left_labels = _leaf_labels(kg["left_root"])
        col_names   = [lbl for lbl, _ in top_labels]
        rows, row_labels = [], []
        for lbl, line_idx in left_labels:
            if line_idx is not None and line_idx < len(data_rows):
                rows.append([str(cell.get("value", "")) for cell in data_rows[line_idx]])
                row_labels.append(lbl)
        if not rows:
            continue
        n_cols = len(col_names)
        rows   = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
        df = _norm_df(pd.DataFrame(
            [[rl] + r for rl, r in zip(row_labels, rows)],
            columns=[""] + col_names,
        ))
        q = normalize_text(sample.get("question", ""))
        if title:
            q = f"[Table: {title}] {q}"
        answers = sample.get("answer", [])
        ans = ", ".join(str(a) for a in answers) if isinstance(answers, list) else str(answers or "")
        samples.append({
            "question": q, "table": df, "answer": ans,
            "table_id": table_id, "original_index": idx,
        })
    return samples


def load_infotabs(path: str) -> List[Dict]:
    tsv_path = os.path.join(path, "data", "maindata", "infotabs_dev.tsv")
    json_dir  = os.path.join(path, "data", "tables", "json")
    raw = pd.read_csv(tsv_path, sep="\t")
    random.seed(42)
    indices   = random.sample(range(len(raw)), min(N, len(raw)))
    label_map = {"E": "entailment", "N": "neutral", "C": "contradiction"}
    samples: List[Dict] = []
    for idx in indices:
        row      = raw.iloc[idx]
        table_id = str(row["table_id"])
        jp = os.path.join(json_dir, f"{table_id}.json")
        if not os.path.exists(jp):
            continue
        with open(jp) as f:
            infobox = json.load(f)
        title = infobox.pop("title", None)
        title_str = (title[0] if isinstance(title, list) and title else str(title)) if title else ""
        rows = [
            [normalize_text(prop), normalize_text(", ".join(str(v) for v in vals) if isinstance(vals, list) else str(vals))]
            for prop, vals in infobox.items()
        ]
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["Property", "Value"]).applymap(normalize_text)
        q  = (f"[Table: {title_str}] " if title_str else "") + (
            "Does the table entail, contradict, or is neutral to the following statement?\n"
            f"Statement: {normalize_text(row['hypothesis'])}"
        )
        samples.append({
            "question": q, "table": df,
            "answer":   label_map.get(str(row["label"]).strip(), str(row["label"])),
            "table_id": table_id, "original_index": idx,
        })
    return samples


def load_tabmwp(path: str) -> List[Dict]:
    fpath = os.path.join(path, "problems_test.json")
    with open(fpath) as f:
        data = json.load(f)
    items = list(data.items())
    random.seed(42)
    indices = random.sample(range(len(items)), min(N, len(items)))
    samples: List[Dict] = []
    for idx in indices:
        pid, ex    = items[idx]
        table_pd   = ex.get("table_for_pd", {})
        df = _norm_df(pd.DataFrame(table_pd)) if table_pd and isinstance(table_pd, dict) else pd.DataFrame()
        q = normalize_text(ex.get("question", ""))
        title   = ex.get("table_title", "")
        choices = ex.get("choices")
        unit    = ex.get("unit", "")
        if title:
            q = f"[Table: {title}] {q}"
        if choices:
            q = f"{q} Choices: {', '.join(f'({chr(65+i)}) {c}' for i, c in enumerate(choices))}"
        if unit:
            q = f"{q} (Answer unit: {unit})"
        samples.append({
            "question": q, "table": df,
            "answer":   normalize_text(ex.get("answer", "")),
            "table_id": str(pid), "original_index": idx,
        })
    return samples


def load_tabfact(path: str) -> List[Dict]:
    tsv_path = os.path.join(path, "processed_datasets", "tsv_data_horizontal", "test.tsv")
    csv_dir  = os.path.join(path, "data", "all_csv")
    raw = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["table_id", "num_cols", "col_name", "row_text", "statement", "label"],
    )
    random.seed(42)
    # Oversample to account for missing CSVs
    pool = random.sample(range(len(raw)), min(N * 4, len(raw)))
    samples: List[Dict] = []
    for idx in pool:
        if len(samples) >= N:
            break
        row      = raw.iloc[idx]
        csv_path = os.path.join(csv_dir, str(row["table_id"]))
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, sep="#")
        except Exception:
            continue
        if df.empty:
            continue
        df  = _norm_df(df)
        ans = "true" if int(row["label"]) == 1 else "false"
        samples.append({
            "question": (
                "Is the following statement true or false about the table?\n"
                f"Statement: {normalize_text(row['statement'])}"
            ),
            "table":          df,
            "answer":         ans,
            "table_id":       str(row["table_id"]),
            "original_index": idx,
        })
    return samples


def load_tatqa(path: str) -> List[Dict]:
    fpath = os.path.join(path, "dataset_raw", "tatqa_dataset_dev.json")
    with open(fpath) as f:
        data = json.load(f)
    all_samples: List[Dict] = []
    for doc_idx, doc in enumerate(data):
        table_data = doc.get("table", {})
        table_rows = table_data.get("table", [])
        table_uid  = table_data.get("uid", f"doc_{doc_idx}")
        paragraphs = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
        context    = "\n".join(str(p["text"]) for p in paragraphs if p.get("text"))
        if len(table_rows) >= 2:
            header = [str(c) for c in table_rows[0]]
            rows   = [[str(cell) for cell in row] for row in table_rows[1:]]
            n_cols = len(header)
            rows   = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
            df = _norm_df(pd.DataFrame(rows, columns=header))
        else:
            df = pd.DataFrame()
        for q in doc.get("questions", []):
            q_text = normalize_text(q.get("question", ""))
            scale  = q.get("scale", "")
            if scale:
                q_text = f"{q_text} (Scale: {scale})"
            if context:
                q_text = f"Context:\n{context}\n\nQuestion: {q_text}"
            answers = q.get("answer", [])
            ans = ", ".join(str(a) for a in answers) if isinstance(answers, list) else str(answers or "")
            all_samples.append({
                "question": q_text, "table": df.copy(), "answer": ans,
                "table_id": f"{table_uid}_q{q.get('order', 0)}",
                "original_index": len(all_samples),
            })
    random.seed(42)
    indices = random.sample(range(len(all_samples)), min(N, len(all_samples)))
    return [all_samples[i] for i in indices]


def load_fetaqa(path: str) -> List[Dict]:
    fpath = os.path.join(path, "fetaQA-v1_test.jsonl")
    data  = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    random.seed(42)
    indices = random.sample(range(len(data)), min(N, len(data)))
    samples: List[Dict] = []
    for idx in indices:
        ex = data[idx]
        ta = ex.get("table_array", [])
        if len(ta) < 2:
            continue
        header = [str(c) for c in ta[0]]
        rows   = [[str(cell) for cell in r] for r in ta[1:]]
        n_cols = len(header)
        rows   = [r[:n_cols] + [""] * max(0, n_cols - len(r)) for r in rows]
        df = _norm_df(pd.DataFrame(rows, columns=header))
        samples.append({
            "question":       normalize_text(ex.get("question", "")),
            "table":          df,
            "answer":         normalize_text(ex.get("answer", "")),
            "table_id":       str(ex.get("feta_id", idx)),
            "original_index": idx,
        })
    return samples


# ── Registry & main ───────────────────────────────────────────────────────────

DATASETS = [
    ("wikitablequestions", load_wikitablequestions, "/path/to/wikitablequestions"),
    ("mmqa",               load_mmqa,               "/path/to/mmqa"),
    ("mmtu",               load_mmtu,               "/path/to/mmtu"),
    ("hitab",              load_hitab,              "/path/to/HiTab"),
    ("infotabs",           load_infotabs,           "/path/to/infotabs"),
    ("tabmwp",             load_tabmwp,             "/path/to/tabmwp"),
    ("tabfact",            load_tabfact,            "/path/to/Table-Fact-Checking"),
    ("tatqa",              load_tatqa,              "/path/to/TAT-QA"),
    ("fetaqa",             load_fetaqa,             "/path/to/fetaQA_raw"),
]

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    for name, loader, path in DATASETS:
        print(f"[{name}] loading from {path} …")
        try:
            samples = loader(path)
            if not samples:
                print(f"[{name}] WARNING: 0 samples returned, skipping")
                continue
            save_samples(name, samples)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
    print("\nDone.")
