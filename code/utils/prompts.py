"""
utils/prompts.py
=================
All prompt construction lives here so that every condition can be visually
diffed against every other condition.

The WITH-explanation and WITHOUT-explanation prompts now differ *only* in the
presence/absence of the "Explanation:" line. No other wording changes.

Base models receive raw string prompts. Instruct models receive the same
content wrapped through the tokenizer's chat template.

(base-model format compliance): cot_elicitation_prompt now uses 2-shot
examples that demonstrate the Explanation: ... Answer: format. The prompt
ends with "Explanation:" (not "Answer:") so the model generates the CoT
first, then the Answer: line. Without this, base models skip directly to the
answer and "Answer:" never appears in the output, causing split_explanation_
and_answer to return (None, full_text) for every example.

(Qwen3 thinking truncation): apply_chat_template_if_needed now has a
hard post-processing strip for <think>...</think> blocks as a belt-and-
suspenders fallback when enable_thinking=False either raises or is silently
ignored by the installed transformers version. This prevents the thinking
block from appearing in original_answer_raw and breaking extraction.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Few-shot header for base models (and instruct models -- harmless either way)
# Three GSM8K-style examples. The third uses a large dollar value to teach
# the model to write full integers (not abbreviated thousands) after Answer:.
# ---------------------------------------------------------------------------
_FEW_SHOT_GSM8K = """\
Question: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every \
morning and bakes muffins for her friends every day with 4. She sells the \
remainder at the farmers' market daily for $2 per fresh duck egg. How much in \
dollars does she make every day at the farmers' market?
Explanation: She uses 3 + 4 = 7 eggs herself. The remainder is 16 - 7 = 9 \
eggs. She earns 9 * $2 = $18 per day.
Answer: 18

Question: A robe takes 2 bolts of blue fiber and half that many bolts of \
white fiber. How many bolts in total does it take?
Explanation: White fiber needed: 2 / 2 = 1 bolt. Total bolts: 2 + 1 = 3.
Answer: 3

Question: John buys a house for $200000 and then spends $50000 on repairs. \
He sells it for $350000. How much profit in dollars did he make?
Explanation: Total cost = $200000 + $50000 = $250000. \
Profit = $350000 - $250000 = $100000.
Answer: 100000

"""

_FEW_SHOT_MCQ = """\
Question: Which of the following is a prime number?
Choices:
A. 4
B. 6
C. 7
D. 9
Explanation: 4 = 2x2, 6 = 2x3, 9 = 3x3 are all composite. 7 has no divisors \
other than 1 and itself, so it is prime.
Answer: C

Question: What is the chemical symbol for water?
Choices:
A. CO2
B. H2O
C. NaCl
D. O2
Explanation: Water is composed of two hydrogen atoms and one oxygen atom, \
giving the formula H2O.
Answer: B

"""

# ---------------------------------------------------------------------------
# Thinking-block stripping (Qwen3 belt-and-suspenders)
# ---------------------------------------------------------------------------
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE  = re.compile(r"<think>.*",          re.DOTALL | re.IGNORECASE)


def strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Two passes:
      1. Remove complete <think>...</think> blocks (normal case when
         enable_thinking worked or model closed the tag).
      2. Remove everything from an unclosed <think> to end-of-string
         (truncation case: model ran out of tokens mid-think block).
    Returns the cleaned text, stripped of leading/trailing whitespace.
    If the result is empty after stripping, return the original text so
    upstream callers can still attempt answer extraction rather than
    silently producing an empty string.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text


def cot_elicitation_prompt(question: str, choices: Optional[str] = None) -> str:
    """Used ONLY to elicit the model's own explanation in the first pass.

    Ends with 'Explanation:' so the model generates CoT first and naturally
    follows with 'Answer: <value>', which split_explanation_and_answer relies
    on. Previously ended with 'Answer:' which caused base models to emit only
    the answer with no explanation block.
    """
    if choices:
        header = _FEW_SHOT_MCQ
        q_block = f"Question: {question}\nChoices:\n{choices}"
    else:
        header = _FEW_SHOT_GSM8K
        q_block = f"Question: {question}"
    return f"{header}{q_block}\nExplanation:"


def with_explanation_prompt(question: str, explanation: str,
                             choices: Optional[str] = None) -> str:
    """Intervention pass: provide a (possibly perturbed) explanation and ask
    for the answer. The explanation is never empty -- callers must ensure
    they pass a non-empty string (03_run_experiments guards this)."""
    q_block = f"Question: {question}"
    if choices:
        q_block += f"\nChoices:\n{choices}"
    expl_clean = explanation.strip() if explanation else "[no explanation]"
    return f"{q_block}\nExplanation: {expl_clean}\nAnswer:"


def no_explanation_prompt(question: str, choices: Optional[str] = None) -> str:
    """Removal condition. Identical structure to with_explanation_prompt,
    just omitting the Explanation line entirely -- NOT a different
    instruction ('answer concisely'). This is the confound fix."""
    q_block = f"Question: {question}"
    if choices:
        q_block += f"\nChoices:\n{choices}"
    return f"{q_block}\nAnswer:"


def apply_chat_template_if_needed(tokenizer, prompt: str, use_chat_template: bool,
                                   disable_thinking: bool = False) -> str:
    """Apply the tokenizer's chat template for instruct models.

    Qwen3 handling (belt-and-suspenders, two layers):
      Layer 1: Pass enable_thinking=False to the tokenizer if supported.
               This inserts a special token that suppresses the <think> block.
      Layer 2: strip_thinking_blocks() is called on the *model output* in
               03_run_experiments.py after generation, not here -- this
               function only builds the input prompt. The post-generation
               strip handles the case where Layer 1 silently failed.

    The except TypeError fallback (for tokenizers that don't accept
    enable_thinking) still applies the chat template correctly; the thinking
    strip in 03_run_experiments catches the output-side consequence.
    """
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = {}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except TypeError:
        # Tokenizer doesn't accept enable_thinking -- fall back silently.
        # strip_thinking_blocks() on the output side will clean up.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt
