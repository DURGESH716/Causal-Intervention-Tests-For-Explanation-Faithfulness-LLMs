"""
00_config.py
=============
Single source of truth for the whole pipeline. Every other script imports
this file instead of hardcoding paths, model names, or sample sizes.
"""

import os
from pathlib import Path

# -------------------------------------------------------------------
# Paths (relative to repo root, i.e. one level above code/)
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
RESULTS_DIR = ROOT_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [MODELS_DIR, BENCHMARKS_DIR, RAW_DIR, METRICS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
GLOBAL_SEED = 42                 # master seed for sampling questions
PERTURBATION_SEEDS = [1, 2, 3]   # multiple seeds for shuffle/swap -> variance estimates
N_SAMPLES_PER_TASK = 500         # fixed N, same indices reused across ALL models
MAX_NEW_TOKENS = 320
GEN_KWARGS = dict(do_sample=False, temperature=None, top_p=None)  # greedy, deterministic

# -------------------------------------------------------------------
# Model registry
# -------------------------------------------------------------------
# tier: "small" (original 6, kept for continuity with prior submission)
#       "large" (added per Y9xk: "I wonder if we still observe these behaviors
#                with larger models")
# type: "base" or "instruct"
# pair_id: matched base/instruct siblings share a pair_id so the metrics
#          script can run a direct paired comparison instead of an n=1 claim.
# use_chat_template: True for instruct models -> applies tokenizer chat template
#          instead of the raw string templates (reviewers flagged that base vs
#          instruct prompts were being treated identically, which is incorrect).
# quantize_4bit: load with bitsandbytes 4-bit NF4 to fit large models in 96GB VRAM.

MODELS = [
    # ---------------- SMALL TIER: matched base/instruct pairs ----------------
    dict(name="Qwen3-8B-Base", hf_repo="Qwen/Qwen3-8B-Base", tier="small",
         family="Qwen3", type="base", pair_id="qwen3_8b", use_chat_template=False,
         quantize_4bit=False),
    dict(name="Qwen3-8B-Instruct", hf_repo="Qwen/Qwen3-8B", tier="small",
         family="Qwen3", type="instruct", pair_id="qwen3_8b", use_chat_template=True,
         quantize_4bit=False, disable_thinking=True,),

    dict(name="Gemma-7B-Base", hf_repo="google/gemma-7b", tier="small",
         family="Gemma", type="base", pair_id="gemma_7b", use_chat_template=False,
         quantize_4bit=False),
    dict(name="Gemma-7B-Instruct", hf_repo="google/gemma-7b-it", tier="small",
         family="Gemma", type="instruct", pair_id="gemma_7b", use_chat_template=True,
         quantize_4bit=False),

    dict(name="Llama-3.1-8B-Base", hf_repo="meta-llama/Llama-3.1-8B", tier="small",
         family="Llama-3.1", type="base", pair_id="llama31_8b", use_chat_template=False,
         quantize_4bit=False),
    dict(name="Llama-3.1-8B-Instruct", hf_repo="meta-llama/Llama-3.1-8B-Instruct",
         tier="small", family="Llama-3.1", type="instruct", pair_id="llama31_8b",
         use_chat_template=True, quantize_4bit=False),

    # ---------------- LARGE TIER: does the effect hold at scale? ----------------
    dict(name="Qwen2.5-32B-Base", hf_repo="Qwen/Qwen2.5-32B", tier="large",
         family="Qwen2.5", type="base", pair_id="qwen25_32b", use_chat_template=False,
         quantize_4bit=False),   # bf16 ~64GB, fits 96GB single GPU
    dict(name="Qwen2.5-32B-Instruct", hf_repo="Qwen/Qwen2.5-32B-Instruct", tier="large",
         family="Qwen2.5", type="instruct", pair_id="qwen25_32b", use_chat_template=True,
         quantize_4bit=False),

    dict(name="Gemma-2-27B-Base", hf_repo="google/gemma-2-27b", tier="large",
         family="Gemma-2", type="base", pair_id="gemma2_27b", use_chat_template=False,
         quantize_4bit=False),
    dict(name="Gemma-2-27B-Instruct", hf_repo="google/gemma-2-27b-it", tier="large",
         family="Gemma-2", type="instruct", pair_id="gemma2_27b", use_chat_template=True,
         quantize_4bit=False),

    dict(name="Llama-3.1-70B-Base", hf_repo="meta-llama/Llama-3.1-70B", tier="large",
         family="Llama-3.1", type="base", pair_id="llama31_70b", use_chat_template=False,
         quantize_4bit=True, quant_method="bitsandbytes"),    # 70B needs 4-bit (~35-40GB) to fit comfortably
    dict(name="Llama-3.1-70B-Instruct", hf_repo="meta-llama/Llama-3.1-70B-Instruct",
         tier="large", family="Llama-3.1", type="instruct", pair_id="llama31_70b",
         use_chat_template=True, quantize_4bit=True, quant_method="bitsandbytes"),
]

# Convenience filters used by 03_run_experiments.py (--tier / --models flags)
def get_models(tier=None, names=None):
    sel = MODELS
    if tier:
        sel = [m for m in sel if m["tier"] == tier]
    if names:
        sel = [m for m in sel if m["name"] in names]
    return sel

# -------------------------------------------------------------------
# Tasks
# -------------------------------------------------------------------
MMLU_STEM_SUBJECTS = [
    "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics",
    "computer_security", "electrical_engineering", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning",
]

TASKS = ["gsm8k", "mmlu_stem"]

# -------------------------------------------------------------------
# Intervention / control conditions
# Addressing reviewer comments directly:
#   - "removal" no longer changes instruction wording (me / nJoK fix).
#   - "perturb_word_shuffle" = original word-level shuffle, explicitly NOT
#     subword-tokenizer shuffle (nJoK: "define exactly what token-level
#     shuffling means").
#   - "perturb_step_reorder" = milder structural perturbation (Ehmi request).
#   - "swap" uses an ANSWER-STRIPPED donor explanation (nJoK leakage concern).
#   - "control_irrelevant" = fluent but off-topic filler, matched length, to
#     separate "sensitive to explanation content" from "sensitive to any
#     change in the prompt" (construct-validity concern, me / nJoK).
#   - "reinject_original" = re-inject the model's own unperturbed explanation
#     through the SAME forced-explanation template, to measure the decoding
#     noise floor (Ehmi: "what's the ACR when you re-inject the original?").
# -------------------------------------------------------------------
CONDITIONS = [
    "removal",
    "perturb_word_shuffle",
    "perturb_step_reorder",
    "swap",
    "control_irrelevant",
    "reinject_original",
]

HF_TOKEN_ENV_VAR = "HF_TOKEN"  # set via `.env` or `export HF_TOKEN=...`; 
