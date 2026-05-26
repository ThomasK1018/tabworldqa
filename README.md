# tabworldqa

Standalone inference and attention-analysis toolkit for visual table reasoning across nine benchmark datasets.

---

## Directory Structure

```
tabworldqa/
├── reasoning_engine.py          # Main inference script (four reasoning modes)
├── create_sample_datasets.py    # One-off script to rebuild sample_datasets/
│
├── tools/                       # Table manipulation tools and reward functions
│   ├── tools_base.py            # Tool implementations (base mode)
│   ├── tools_adaptive.py        # Tool implementations with adaptive modality switching
│   ├── tool_defs_base.py        # Tool function definitions (base mode)
│   ├── tool_defs_adaptive.py    # Tool function definitions (adaptive modes)
│   ├── reward_base.py           # Reward utilities
│   ├── reward_metadata.py       # Reward + metadata prediction evaluator
│   └── reward_table_state.py    # Reward + table-state prediction evaluator
│
├── sample_datasets/             # 20 prebuilt samples per dataset (JSONL, ready to run)
│   ├── wikitablequestions/
│   ├── mmqa/
│   ├── mmtu/
│   ├── hitab/
│   ├── infotabs/
│   ├── tabmwp/
│   ├── tabfact/
│   ├── tatqa/
│   └── fetaqa/
│
├── data/                        # Raw data for attention analysis
│   └── wikitq_nt2317/
│
├── results/                     # Generated outputs (CSV results, plots)
│   └── metrics_results.csv
│
├── analysis/                    # Attention and figure analysis scripts
│   ├── run_attention_analysis.py
│   ├── plot_attention_mse.py
│   └── figure_ablation_split.py  # Generates figure_main_ablation_a/b.pdf
│
└── archive/                     # Original versioned scripts (reference only)
```

---

## Dependencies

```bash
pip install openai pandas pillow matplotlib
```

For dataset loaders that use HuggingFace Arrow format (MMQA, MMTU, WikiTableQuestions):

```bash
pip install datasets
```

---

## Reasoning Modes

`reasoning_engine.py` supports four modes selected with `--mode`:

| Mode | Description |
|---|---|
| `base` | Action-compressed reasoning, no reward tracking |
| `v35` | Single-action steps, adaptive modality, metadata prediction reward |
| `full` | Batched-action steps, adaptive modality, metadata prediction reward |
| `full-table` | Same as `full` but predicts table state instead of metadata |

---

## Quick Start — Prebuilt Samples

The fastest way to run inference is to use the prebuilt 20-sample JSONL files in `sample_datasets/`. No external dataset download required.

**Text-only inference (base mode, OpenAI):**

```bash
python reasoning_engine.py \
  --mode base \
  --model-name gpt-4o \
  --samples-path sample_datasets/wikitablequestions/samples.jsonl \
  --start 0 --end 5 \
  --table-input-mode text \
  --max-steps 8 \
  --csv-output results/wtq_base.csv
```

**Vision inference (full mode, local vLLM server):**

```bash
python reasoning_engine.py \
  --mode full \
  --provider ollama \
  --model-name Qwen/Qwen3-VL-8B-Thinking-FP8 \
  --base-url http://localhost:8000/v1 \
  --max-tokens 16384 \
  --samples-path sample_datasets/wikitablequestions/samples.jsonl \
  --start 0 --end 5 \
  --table-input-mode image \
  --max-steps 8 \
  --csv-output results/wtq_full.csv
```

**Swap `--samples-path` to test any other dataset:**

```bash
--samples-path sample_datasets/tabfact/samples.jsonl
--samples-path sample_datasets/tatqa/samples.jsonl
--samples-path sample_datasets/hitab/samples.jsonl
# etc.
```

---

## Full Dataset Inference

Pass the path to the original dataset instead of a prebuilt JSONL file. Each dataset has a dedicated argument:

| Dataset | Argument | Expected path contents |
|---|---|---|
| WikiTableQuestions | `--wikitablequestions-path` | HuggingFace Arrow dataset (`dataset_dict.json` + splits) |
| MMQA | `--mmqa-path` | `MMQA_dataset.csv` + `MMQA_datasets/` |
| MMTU | `--mmtu-path` | `mmtu_record.xlsx` + `MMTU_datasets/` |
| FeTaQA | `--fetaqa-path` | `fetaQA-v1_{train,dev,test}.jsonl` |
| TabMWP | `--tabmwp-path` | `problems_{split}.json` |
| TAT-QA | `--tatqa-path` | root dir containing `dataset_raw/` |
| HiTab | `--hitab-path` | root dir containing `data/` |
| TabFact | `--tabfact-path` | root dir containing `processed_datasets/` and `data/all_csv/` |
| InfoTabs | `--infotabs-path` | root dir containing `data/maindata/` and `data/tables/json/` |

Example:

```bash
python reasoning_engine.py \
  --mode full \
  --model-name gpt-4o \
  --wikitablequestions-path /path/to/wikitablequestions \
  --start 0 --end 20 \
  --table-input-mode text \
  --csv-output results/wtq_full_20.csv
```

---

## Key Arguments

### Model

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `gpt-4o` | Model name or path passed to the API |
| `--base-url` | *(none)* | API server URL, e.g. `http://localhost:8000/v1` for local vLLM |
| `--api-key` | `EMPTY` | API key (`EMPTY` works for local vLLM / Ollama servers) |
| `--max-tokens` | `8192` | Maximum output tokens per call |
| `--temperature` | `0.7` | Sampling temperature |
| `--thinking-budget` | *(none)* | Thinking token budget for Qwen3 thinking models (text-only inputs) |

### Inference

| Argument | Default | Description |
|---|---|---|
| `--mode` | `full` | Reasoning mode: `base`, `v35`, `full`, `full-table` |
| `--table-input-mode` | `image` | How the table is presented each step: `text`, `image`, `multimodal`, `text_json`, `text_latex` |
| `--max-steps` | `12` | Maximum tool-use steps per question |
| `--question-timeout` | `300` | Skip a question after this many seconds |
| `--start` / `--end` | `0` / `1` | Slice of the dataset to run |
| `--csv-output` | *(none)* | Path to write results CSV |
| `--compact-csv` | off | Omit large trace fields from CSV |
| `--prompt-optimizer-rounds` | `0` | Enable FL-style prompt hint updates every N questions |

### Ablation routing controls

These flags override the model's own modality selection and reproduce the ablation conditions reported in Table 2.

| Argument | Description |
|---|---|
| `--random-modality` | Select modality uniformly at random each step |
| `--heuristic-modality` | Rule-based routing: compute/transform actions use text, all others use image |
| `--question-level-routing` | Model picks modality at step 1 and holds it fixed for the entire question |

---

## Starting a Local vLLM Server

```bash
conda run -n tabworldqa_env python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Thinking-FP8 \
  --max-model-len 31632 \
  --port 8000
```

Then pass `--base-url http://localhost:8000/v1 --api-key EMPTY` to `reasoning_engine.py`.

---

## Rebuilding Sample Datasets

If you want to regenerate the prebuilt JSONL files from the full source datasets:

```bash
conda run -n tabworldqa_env python create_sample_datasets.py
```

Edit the `DATASETS` list at the bottom of `create_sample_datasets.py` to adjust source paths or the number of samples (`N = 20`).

---

## Attention Analysis

To replicate the attention MSE results from `results/metrics_results.csv`:

```bash
# Run full attention analysis (requires GPU + HuggingFace model weights)
conda run -n tabworldqa_env python analysis/run_attention_analysis.py \
  --question-ids 0,1,2,3,4,5,6,7,8,9 \
  --n-runs 10

# Plot panel (c) — Attention MSE on Tabular Actions
conda run -n tabworldqa_env python analysis/plot_attention_mse.py
# Output: results/attention_mse.pdf
```

## Ablation Figures

To regenerate the modality-preference heatmap and reward bar chart:

```bash
python analysis/figure_ablation_split.py
# Output: figures/figure_main_ablation_a.pdf
#         figures/figure_main_ablation_b.pdf
```

The script reads step-level data from the path configured at the top of the file (`STEP_CSV`). Update that path to point to your own run outputs.
