"""
utils/perturbations.py
=======================
Implements the intervention conditions, each tied to a specific design:

  - word_shuffle: original perturbation, but explicitly WORD-level (not
    tokenizer-subword level).
  - step_reorder: milder structural perturbation
  - strip_answer_leakage: removes trailing answer-revealing phrases from a
    donor explanation before it is used in the swap condition.
  - irrelevant_filler: generates fluent, topic-unrelated text of matched
    length for the control_irrelevant condition (construct-validity:
    separates "sensitive to explanation CONTENT" from "sensitive to ANY
    prompt perturbation").
"""

import random
import re
from typing import List, Optional

# A small bank of neutral, fluent, topic-irrelevant sentences used to build
# the irrelevant-filler control. Kept deliberately mundane and unrelated to
# math/science so it cannot accidentally leak task-relevant content.
_FILLER_BANK = [
    "The library closes early on weekends during the summer months.",
    "Many cities have expanded their public transportation networks recently.",
    "The weather in coastal regions tends to be milder than inland areas.",
    "Local markets often sell fresh produce early in the morning.",
    "Several museums have introduced free admission days this year.",
    "The committee will meet again next month to discuss the proposal.",
    "Train schedules can vary slightly depending on the day of the week.",
    "A new community garden opened near the old town hall recently.",
    "The bakery on the corner is known for its sourdough bread.",
    "Volunteers organized a cleanup event at the riverside park.",
]


def _safe(explanation: Optional[str]) -> str:
    """Coerce None to empty string. All public functions call this first so
    callers never need to guard individually. Returns "" for None/empty input,
    which perturbation functions treat as a degenerate (< 5 token) explanation
    and return unchanged (i.e. still "")."""
    return explanation.strip() if explanation else ""


def word_shuffle(explanation: Optional[str], seed: int) -> str:
    """Randomly shuffles the words in the explanation.
    Returns "" unchanged when explanation is None or has fewer than 5 words."""
    text = _safe(explanation)
    rng = random.Random(seed)
    tokens = text.split()
    if len(tokens) < 5:
        return text          # includes "" case -- caller sees empty string
    rng.shuffle(tokens)
    return " ".join(tokens)


def step_reorder(explanation: Optional[str], seed: int) -> str:
    """Splits the explanation into sentence-like 'steps' and shuffles the
    step ORDER while keeping each step's internal wording intact. This is
    a strictly milder perturbation than word_shuffle: it disrupts logical
    sequencing without destroying local fluency.
    Returns "" unchanged when explanation is None or has fewer than 2 steps."""
    text = _safe(explanation)
    rng = random.Random(seed)
    steps = re.split(r"(?<=[.!?])\s+", text)
    steps = [s for s in steps if s]
    if len(steps) < 2:
        return text          # includes "" case
    rng.shuffle(steps)
    return " ".join(steps)


# ---------------------------------------------------------------------------
# Answer-leakage stripping (for swap donors)
# ---------------------------------------------------------------------------
_ANSWER_LEAK_PATTERNS = [
    r"the answer is[:\s]*.*$",
    r"answer[:\s]*.*$",
    r"####.*$",
    r"so,?\s*(the )?(final )?(answer|result) (is|=).*$",
    r"\bfinal answer\b.*$",
    r"\bx\s*=\s*-?\d+\.?\d*\s*\.?$",
]


def strip_answer_leakage(explanation: Optional[str]) -> str:
    """Removes trailing sentences/clauses that reveal the final answer,
    so the swap condition tests reliance on REASONING, not a copied answer.

    Heuristic regex-based; documented as a best-effort, not a perfect
    guarantee (noted explicitly in the paper's limitations).

    FIX: previously called explanation.strip() without guarding against None,
    causing AttributeError when a base model explanation slot was None. Now
    returns "" for None input so the caller gets a well-typed string. The
    caller (03_run_experiments) should handle empty-string donors gracefully
    (e.g. skip the swap condition or log a warning)."""
    text = _safe(explanation)
    if not text:
        return ""            # nothing to strip; caller decides what to do

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s_lower = s.lower()
        if any(re.search(p, s_lower) for p in _ANSWER_LEAK_PATTERNS):
            continue
        # also strip bare trailing numbers/letters that look like a final answer
        if re.fullmatch(r"-?\d+\.?\d*\.?", s.strip()):
            continue
        if re.fullmatch(r"\(?[A-D]\)?\.?", s.strip()):
            continue
        kept.append(s)

    result = " ".join(kept).strip()
    # Never return empty string when we started with real content -- that
    # would silently drop all reasoning. Fall back to full text so the
    # swap condition at least has *something* (leakage risk logged by caller).
    return result if result else text


# ---------------------------------------------------------------------------
# Irrelevant filler control
# ---------------------------------------------------------------------------
def irrelevant_filler(target_word_count: int, seed: int) -> str:
    """Generates fluent, topic-unrelated text of approximately target_word_count
    words by cycling through _FILLER_BANK sentences.
    target_word_count is guaranteed >= 5 by the caller (03_run_experiments uses
    max(len(expl.split()), 5)), so this never returns an empty string."""
    rng = random.Random(seed)
    pool = _FILLER_BANK.copy()
    rng.shuffle(pool)
    out_words: List[str] = []
    i = 0
    while len(out_words) < target_word_count:
        out_words.extend(pool[i % len(pool)].split())
        i += 1
    return " ".join(out_words[:target_word_count])
