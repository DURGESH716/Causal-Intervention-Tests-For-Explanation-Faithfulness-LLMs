# Faithfulness via Causal Interventions — Evaluation Pipeline

This pipeline implements the causal-intervention evaluation for LLM
explanation faithfulness described in the accompanying paper, including two
controls: a content-free specificity check and a decoding-noise floor.
## Setup

```bash
cd Health3
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste your OWN HF token
```

Your `.env` should contain a single line:

```
HF_TOKEN=hf_your_own_token_here
```

`.env` is already gitignored — never commit it, and never paste a real token into
a README, notebook, issue, or chat.

Your HF token must belong to an account that has accepted the gated-model
licenses on the Hugging Face pages for: `meta-llama/Llama-3.1-8B(-Instruct)`,
`meta-llama/Llama-3.1-70B(-Instruct)`, and the Gemma model cards.


## Pipeline order

```bash
cd code
python 01_download_models.py            # downloads all 12 model checkpoints
python 02_download_data.py              # fixed N=500/task sample, cached once
python 03_run_experiments.py            # main experiment loop (long-running)
python 04_compute_metrics_and_figures.py  # ACR tables, CIs, McNemar tests, figures
```

Useful flags for partial / incremental runs:
```bash
python 01_download_models.py --tier small
python 03_run_experiments.py --tier small --tasks gsm8k
python 03_run_experiments.py --models Qwen3-8B-Base Qwen3-8B-Instruct
```

## Design overview

| Design point | Implementation | Where |
|---|---|---|
| Removal must not confound explanation absence with an instruction-wording change | `with_explanation_prompt` and `no_explanation_prompt` differ **only** in the presence of the `Explanation:` line | `code/utils/prompts.py` |
| Swap must not leak the final answer through the donor explanation | `strip_answer_leakage()` regex-removes trailing answer-revealing clauses before a donor explanation is used | `code/utils/perturbations.py` |
| Raw ACR conflates "sensitive to explanation content" with "sensitive to any prompt change" (construct validity) | Added `control_irrelevant` (fluent, off-topic, length-matched filler) and `reinject_original` (noise-floor baseline) conditions; ACR is reported both raw and noise-floor-adjusted | `code/utils/perturbations.py`, `code/03_run_experiments.py`, `acr_adjusted_pct` column in `04_compute_metrics_and_figures.py` |
| ACR alone doesn't distinguish right→wrong / wrong→right / wrong→wrong | `transition_breakdown()` reports the full confusion-style breakdown per model/task/condition | `code/utils/stats.py`, `results/metrics/transition_breakdown.csv` |
| Small N, single run, no CIs understates uncertainty | N = 500/task; `bootstrap_ci()` reports 95% CIs on every ACR estimate; `mcnemar_test()` tests whether intervention differences are actually significant | `code/00_config.py`, `code/utils/stats.py` |
| "Word-level" vs. subword-level shuffling must be unambiguous | Explicit **word**-level shuffle (`word_shuffle`); added a milder **step-reorder** structural perturbation as a separate, comparable condition | `code/utils/perturbations.py` |
| MMLU answer extraction/normalization must be explicit | Documented, ordered extraction rules (letter pattern → choice-text match → invalid); invalid/unparsable rate logged and reported, never silently guessed | `code/utils/answer_extraction.py` |
| Sample must be identical across every model, not re-drawn per run | Sample is drawn **once**, with a fixed global seed, and cached to `data/benchmarks/*.jsonl`; every model in `03_run_experiments.py` loads the identical file | `code/02_download_data.py` |
| A single instruction-tuned model can't support a general instruction-tuning claim | Matched base/instruct **pairs** from three families at two scale tiers (Qwen3-8B, Gemma-7B, Llama-3.1-8B, Qwen2.5-32B, Gemma-2-27B, Llama-3.1-70B); paired McNemar comparisons per `pair_id` | `code/00_config.py`, `build_base_vs_instruct_table()` in `code/04_compute_metrics_and_figures.py` |
| Results should hold at scale, not just 7–8B | Added a `large` tier (27B–70B, 4-bit quantized for the 70B pair to fit a single 96GB GPU) | `code/00_config.py` |
| Need a baseline for ACR when re-injecting the unperturbed explanation | `reinject_original` condition; used as the noise floor for adjusted ACR | `code/utils/prompts.py`, `code/03_run_experiments.py` |
| Ordering claims (e.g. removal > perturbation > swap) must be backed by significance tests | `mcnemar_intervention_pairs.csv` gives a significance test for every pairwise comparison, per model/task — only report an ordering claim where it's actually significant | `code/04_compute_metrics_and_figures.py` |
| Base and instruct models must not be prompted identically | `use_chat_template` flag per model; instruct models go through `tokenizer.apply_chat_template`, base models get raw string prompts | `code/utils/prompts.py`, `code/03_run_experiments.py` |
| Checkpoints, seeds, and decoding settings must be fully specified | `run_manifest.json` records exact HF repo IDs, seeds, sample counts, and timestamps for every run | `code/03_run_experiments.py` |

## Outputs

- `results/raw/<model>__<task>.jsonl` — every single prompt + generation, fully auditable (one line per example x condition x seed).
- `results/metrics/acr_table.csv` — ACR%, 95% CI, invalid rate, noise-floor-adjusted ACR.
- `results/metrics/transition_breakdown.csv` — right→wrong / wrong→right / etc.
- `results/metrics/mcnemar_intervention_pairs.csv` — significance tests between interventions.
- `results/metrics/base_vs_instruct.csv` — paired base-vs-instruct comparisons per family.
- `results/metrics/run_manifest.json` — exact reproducibility record of the run.
- `results/figures/acr_<task>_<tier>.png` — bar charts with CI error bars.
