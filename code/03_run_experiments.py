"""
03_run_experiments.py
======================
For every model x every task x every example, this script:
  1. Generates the model's own CoT explanation + answer (one pass, cached).
  2. Runs all causal-intervention conditions defined in 00_config.CONDITIONS,
     using the SAME fixed dataset sample for every model (see 02_download_data.py).
  3. Extracts + normalizes answers using utils/answer_extraction.py (explicit,
     documented rules -- no silent guessing on unparsable output).
  4. Writes one JSONL line per (example, condition, seed) to
     results/raw/<model_name>__<task>.jsonl -- the FULL prompt and generated
     text are logged, not just a boolean, so every result is auditable.

New things implemented here:
  - Removal no longer changes instruction wording (uses utils/prompts.py).
  - Swap donor explanations are answer-stripped before use (leakage fix).
  - Word-shuffle and step-reorder are both run, each across multiple seeds.
  - An irrelevant-filler control and an original-reinjection baseline are
    run alongside the "real" interventions, to let 04_compute_metrics... report
    a noise-floor-adjusted ACR.
  - Chat templates are applied for instruct models, raw prompts for base models.

GPU / inference backend:
  - Inference now runs on vLLM instead of raw transformers .generate().
  - A run_manifest.json captures exact checkpoint revisions, seeds, and
    sample counts for reproducibility.

Usage:
    python 03_run_experiments.py                          # everything
    python 03_run_experiments.py --tier small              # only small tier
    python 03_run_experiments.py --models Qwen3-8B-Base     # one model
    python 03_run_experiments.py --tasks gsm8k               # one task
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_GPUS = 1
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
config = importlib.import_module("00_config")

from utils.prompts import (
    cot_elicitation_prompt, with_explanation_prompt, no_explanation_prompt,
    apply_chat_template_if_needed, strip_thinking_blocks,
)
from utils.perturbations import (
    word_shuffle, step_reorder, strip_answer_leakage, irrelevant_filler,
)
from utils.answer_extraction import extract_answer

NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_explanation_and_answer(text: str):
    """Parse elicitation output into (explanation, answer).

    Handles four output styles in priority order:
      1. Explicit 'Answer:' marker (few-shot base models, some instruct).
      2. Instruct-model verbal MCQ patterns: 'the answer is B', '**B**'.
      3. Instruct-model verbal GSM8K patterns: 'therefore ... $X'.
      4. Bare number/dollar-amount on last line (base model GSM8K fallback).
    Returns (explanation_or_None, answer_string).
    """
    # Primary: explicit "Answer:" -- rfind skips few-shot header occurrences
    if "Answer:" in text:
        idx = text.rfind("Answer:")
        expl = text[:idx].strip()
        ans = text[idx + len("Answer:"):].strip().split("\n")[0].strip()
        return (expl if expl else None), ans

    # Instruct MCQ: "the answer is B", "**B**", "therefore ... answer is B"
    for pattern in [
        r"(?:the (?:correct |best )?answer is)[:\s]*\**\(?([A-D])\)?\**",
        r"\*\*([A-D])\*\*",
        r"(?:therefore|thus)[,.]?\s+(?:the )?(?:correct )?answer is[:\s]*\(?([A-D])\)?",
        r"(?:option|choice)\s+\**\(?([A-D])\)?\**\s+is correct",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            expl = text[:m.start()].strip()
            return (expl if expl else None), m.group(1).upper()

    # Instruct GSM8K: verbal numeric answer markers
    for pattern in [
        r"(?:therefore|thus)[,.]?\s+(?:the (?:final )?answer is\s+)?(\$?-?[\d,]+(?:\.\d+)?)",
        r"(?:profit|total|result|cost)[^\n]{0,40}?=\s*(\$?-?[\d,]+(?:\.\d+)?)\s*\.?\s*$",
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            expl = text[:m.start()].strip()
            return (expl if expl else None), m.group(1).strip()

    # Last-line bare number fallback (base model GSM8K without Answer: token)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if re.fullmatch(r"\$?-?[\d,]+(?:\.\d+)?\.?", last):
            expl = "\n".join(lines[:-1]).strip()
            return (expl if expl else None), last

    return None, text.strip()


def load_model_and_tokenizer(model_cfg: dict):
    local_path = config.MODELS_DIR / model_cfg["name"]
    repo_or_path = str(local_path) if local_path.exists() else model_cfg["hf_repo"]

    tokenizer = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Instruct models produce longer reasoning chains; give them more headroom.
    max_tokens = 1024 if model_cfg.get("use_chat_template") else 512

    llm_kwargs = dict(
        model=repo_or_path,
        tokenizer=repo_or_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        max_model_len=4096,
    )

    if model_cfg.get("quantize_4bit"):
        quant_method = model_cfg.get("quant_method", "bitsandbytes")
        llm_kwargs["quantization"] = quant_method
        if quant_method == "bitsandbytes":
            llm_kwargs["load_format"] = "bitsandbytes"

    llm = LLM(**llm_kwargs)
    return tokenizer, llm, max_tokens


def generate_batch(llm, prompts: list, max_new_tokens: int,
                   extra_stop: list = None):
    stop_strings = ["\nQuestion:", "\n\nQuestion:", "\nExplanation:"]
    if extra_stop:
        stop_strings = list(dict.fromkeys(stop_strings + extra_stop))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=stop_strings,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return [out.outputs[0].text.strip() for out in outputs]


def shutdown_vllm(llm):
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


import re

def build_explanation_text(text: str) -> str:
    """Builds the explanation text by stripping out any answer-announcing
    clause, regardless of whether it appears at the start, middle, or end
    of the generation -- unlike the old 'everything before the first
    Answer:' split, which broke whenever a model announced the answer
    first and explained afterward, or used phrasing other than the
    literal 'Answer:' token.

    Falls back to the full text if no recognizable answer-announcing
    pattern is found (over-including is far better than silently
    discarding real reasoning)."""
    patterns = [
        r"(the correct answer is|the final answer is|the answer is)\s*\(?[A-Za-z0-9\-\.]+\)?\.?",
        r"answer:\s*\(?[A-Za-z0-9\-\.]+\)?\.?",
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text.strip()

def run_one_model_task(model_cfg, task, dataset, out_path):
    print(f"\n=== {model_cfg['name']}  |  {task}  |  n={len(dataset)} ===")
    tokenizer, llm, max_new_tokens = load_model_and_tokenizer(model_cfg)
    use_chat = model_cfg["use_chat_template"]
    disable_thinking = model_cfg.get("disable_thinking", False)

    def wrap(raw_prompt):
        return apply_chat_template_if_needed(
            tokenizer, raw_prompt, use_chat,
            disable_thinking=disable_thinking,
        )

    # ---- Pass 1: elicit CoT explanations ----
    elicitation_prompts = [
        wrap(cot_elicitation_prompt(ex["question"], ex.get("choices_text")))
        for ex in dataset
    ]
    elicitation_outputs = generate_batch(llm, elicitation_prompts, max_new_tokens)

    explanations, original_answers = [], []
    n_missing_expl = 0
    for raw_out in elicitation_outputs:
        # Belt-and-suspenders: strip <think> blocks before parsing.
        # Handles Qwen3 when enable_thinking=False silently failed.
        out = strip_thinking_blocks(raw_out)
        _, ans = split_explanation_and_answer(out)        # answer-extraction path: unchanged, already works
        expl_text = build_explanation_text(out)            # NEW: robust explanation construction from full text
        expl = strip_answer_leakage(expl_text) or None      # also dedupe leakage-stripping here, not just at swap time
        explanations.append(expl)
        original_answers.append(ans)
        if expl is None:
            n_missing_expl += 1

    if n_missing_expl:
        pct = 100 * n_missing_expl / len(dataset)
        print(f"  WARNING: {n_missing_expl}/{len(dataset)} ({pct:.1f}%) examples "
              f"produced no parseable explanation.")

    # ---- Pass 2: build all condition prompts ----
    jobs = []  # (example_index, condition, seed, expl_missing, raw_prompt)

    for i, ex in enumerate(dataset):
        expl = explanations[i]
        expl_str = expl or ""
        q = ex["question"]
        choices_text = ex.get("choices_text")
        target_len = max(len(expl_str.split()), 5)
        expl_missing = expl is None

        jobs.append((i, "removal", None, expl_missing,
                     no_explanation_prompt(q, choices_text)))

        for seed in config.PERTURBATION_SEEDS:
            jobs.append((i, "perturb_word_shuffle", seed, expl_missing,
                         with_explanation_prompt(q, word_shuffle(expl_str, seed), choices_text)))

        for seed in config.PERTURBATION_SEEDS:
            jobs.append((i, "perturb_step_reorder", seed, expl_missing,
                         with_explanation_prompt(q, step_reorder(expl_str, seed), choices_text)))

        for offset, seed in zip([1, 2, 3], config.PERTURBATION_SEEDS):
            donor_raw = explanations[(i + offset) % len(dataset)]
            donor_expl = strip_answer_leakage(donor_raw)
            jobs.append((i, "swap", seed, expl_missing,
                         with_explanation_prompt(q, donor_expl, choices_text)))

        for seed in config.PERTURBATION_SEEDS:
            jobs.append((i, "control_irrelevant", seed, expl_missing,
                         with_explanation_prompt(q, irrelevant_filler(target_len, seed), choices_text)))

        jobs.append((i, "reinject_original", None, expl_missing,
                     with_explanation_prompt(q, expl_str, choices_text)))

    print(f"  Batching {len(jobs)} total prompts...")
    all_prompts = [wrap(p) for (_, _, _, _, p) in jobs]
    all_outputs = generate_batch(
        llm, all_prompts, max_new_tokens,
        extra_stop=["\nAnswer:"],
    )

    # ---- Write results ----
    n_written = 0
    with open(out_path, "w") as fout:
        for (i, condition, seed, expl_missing, prompt), generated_text in zip(jobs, all_outputs):
            ex = dataset[i]

            # Strip thinking blocks from intervention outputs too (Qwen3 safety)
            generated_text_clean = strip_thinking_blocks(generated_text)

            ans_orig = extract_answer(task, original_answers[i], ex.get("choices_text"))
            ans_int  = extract_answer(task, generated_text_clean, ex.get("choices_text"))

            # Validity-filtered 'changed':
            #   True only when BOTH sides are valid AND answers differ.
            #   This prevents invalid-baseline rows (original_answer_valid=False)
            #   from inflating ACR -- an empty baseline vs any answer always
            #   looks like a "change" under the raw string comparison.
            # 'changed_raw' preserves the unfiltered comparison for auditing.
            changed_raw = (ans_int.normalized != ans_orig.normalized)
            changed = (
                ans_orig.is_valid
                and ans_int.is_valid
                and changed_raw
            )

            record = {
                "model": model_cfg["name"],
                "task": task,
                "example_index": i,
                "gold_answer": ex["gold_answer"],
                "condition": condition,
                "seed": seed,
                "explanation_missing": expl_missing,
                "original_explanation": explanations[i],
                "original_answer_raw": original_answers[i],
                "original_answer_norm": ans_orig.normalized,
                "original_answer_valid": ans_orig.is_valid,
                "original_answer_correct": (
                    ans_orig.normalized == str(ex["gold_answer"])
                    if ans_orig.is_valid else None
                ),
                "prompt": prompt,
                "generated_text": generated_text_clean,
                "intervened_answer_norm": ans_int.normalized,
                "intervened_answer_valid": ans_int.is_valid,
                "changed": changed,           # validity-filtered -- use for ACR
                "changed_raw": changed_raw,   # unfiltered -- for audit only
            }
            fout.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records -> {out_path}")
    shutdown_vllm(llm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["small", "large"], default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=config.TASKS)
    args = parser.parse_args()

    models = config.get_models(tier=args.tier, names=args.models)
    if not models:
        print("No models matched filters.")
        return

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "global_seed": config.GLOBAL_SEED,
        "perturbation_seeds": config.PERTURBATION_SEEDS,
        "n_samples_per_task": config.N_SAMPLES_PER_TASK,
        "gpus_used": os.environ["CUDA_VISIBLE_DEVICES"],
        "inference_backend": "vllm",
        "models": models,
        "tasks": args.tasks,
    }

    for task in args.tasks:
        data_path = config.BENCHMARKS_DIR / f"{task}.jsonl"
        if not data_path.exists():
            print(f"Missing {data_path} -- run 02_download_data.py first.")
            continue
        dataset = load_jsonl(data_path)

        for model_cfg in models:
            out_path = config.RAW_DIR / f"{model_cfg['name']}__{task}.jsonl"
            run_one_model_task(model_cfg, task, dataset, out_path)

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(config.METRICS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nRun manifest written -> {config.METRICS_DIR / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
