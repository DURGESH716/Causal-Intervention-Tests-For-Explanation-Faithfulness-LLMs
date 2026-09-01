"""
02_download_data.py
====================
Downloads GSM8K + full MMLU-STEM (16 subjects), draws ONE fixed sample of
N_SAMPLES_PER_TASK questions per task using GLOBAL_SEED, and caches that
exact sample to data/benchmarks/<task>.jsonl.

Usage:
    python 02_download_data.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
config = importlib.import_module("00_config")

from datasets import load_dataset

from utils.answer_extraction import extract_gsm8k_gold, extract_mmlu_gold


def build_gsm8k_sample():
    print("Loading GSM8K (main, test split)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n_total = len(ds)
    rng = random.Random(config.GLOBAL_SEED)
    idx = list(range(n_total))
    rng.shuffle(idx)
    chosen = sorted(idx[: config.N_SAMPLES_PER_TASK])

    records = []
    for i in chosen:
        x = ds[i]
        records.append({
            "task": "gsm8k",
            "source_index": i,
            "question": x["question"],
            "choices_text": None,
            "gold_answer": extract_gsm8k_gold(x["answer"]),
            "raw_gold_field": x["answer"],
        })
    return records


def build_mmlu_stem_sample():
    print("Loading MMLU-STEM (16 subjects, test split)...")
    pooled = []
    for subject in config.MMLU_STEM_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i, x in enumerate(ds):
            choices_text = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(x["choices"]))
            pooled.append({
                "task": "mmlu_stem",
                "subject": subject,
                "source_index": i,
                "question": x["question"],
                "choices_text": choices_text,
                "gold_answer": extract_mmlu_gold(x["answer"]),
                "raw_gold_field": x["answer"],
            })

    print(f"Pooled {len(pooled)} MMLU-STEM questions across {len(config.MMLU_STEM_SUBJECTS)} subjects.")
    rng = random.Random(config.GLOBAL_SEED)
    rng.shuffle(pooled)
    chosen = pooled[: config.N_SAMPLES_PER_TASK]
    return chosen


def write_jsonl(records, path: Path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records -> {path}")


def main():
    gsm8k_records = build_gsm8k_sample()
    write_jsonl(gsm8k_records, config.BENCHMARKS_DIR / "gsm8k.jsonl")

    mmlu_records = build_mmlu_stem_sample()
    write_jsonl(mmlu_records, config.BENCHMARKS_DIR / "mmlu_stem.jsonl")

    print("\nDone. These two files are now the FIXED evaluation sets.")
    print("Every model in 03_run_experiments.py reads from these exact files —")
    print("no re-sampling happens per model.")


if __name__ == "__main__":
    main()
