"""
utils/answer_extraction.py
===========================
This module makes every extraction rule explicit, deterministic, and logged.
Unparsable outputs are NEVER silently coerced into a guess -- they are
flagged `is_valid=False` and excluded from ACR (with the exclusion rate
reported as its own metric in 04_compute_metrics_and_figures.py).

If the model repeats the question or continues generating after the answer, the last number is
from that continuation, not the answer. The fixed strategy is:

  1. If the text contains "Answer:" take only what follows it (first line).
  2. Otherwise use only the first non-empty line of the generated text.
  3. Extract the first number from that narrowed scope.
  4. Fall back to last number in the full text only if step 1-3 yield nothing.

This matches how a human reader would parse "Answer: 50" -- they read the
first number after the marker, not scan the entire output.
"""
import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedAnswer:
    raw_text: str
    normalized: str
    is_valid: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_number(val: str) -> Optional[str]:
    """Convert a digit string to a canonical integer-or-float string.
    Returns None if val is not a finite number."""
    try:
        f = float(val)
        if not math.isfinite(f):
            return None
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _first_answer_line(text: str) -> str:
    """Return the most answer-relevant substring of text for number extraction.

    Priority:
      1. Text after the LAST "Answer:" marker (first line only) -- handles
         few-shot prompts that contain "Answer:" in the header examples.
      2. Text after common verbal markers ("the answer is", "profit is", etc.)
         on the first matching line.
      3. First non-empty line of the full text.

    Never returns an empty string -- falls back to full text if nothing else
    matches, so the outer function can still attempt extraction.
    """
    # 1. Explicit "Answer:" marker -- use rfind to skip few-shot occurrences
    if "Answer:" in text:
        after = text[text.rfind("Answer:") + len("Answer:"):].strip()
        first_line = after.split("\n")[0].strip()
        if first_line:
            return first_line

    # 2. Verbal markers on any line
    for pattern in [
        r"(?:the )?(?:final )?answer is[:\s]+(.+)",
        r"(?:total|profit|result|cost|value)[^\n]{0,30}?=\s*(\$?-?[\d,]+(?:\.\d+)?)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).split("\n")[0].strip()

    # 3. First non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line

    return text  # ultimate fallback


# ---------------------------------------------------------------------------
# GSM8K: numeric answers
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> ExtractedAnswer:
    """Extract and normalise a numeric answer from GSM8K model output.

    Extraction scope is narrowed to the answer line before scanning for
    numbers, so spurious numbers in a repeated question or continued
    generation don't override the actual answer.
    """
    if not text or not text.strip():
        return ExtractedAnswer(raw_text=text, normalized="", is_valid=False)

    # Narrow to the most answer-relevant line
    answer_line = _first_answer_line(text)

    # Clean punctuation that isn't part of a number
    cleaned = answer_line.strip().lower()
    cleaned = cleaned.replace(",", "").replace("$", "")

    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    if not numbers:
        # Nothing on the answer line -- fall back to full text, last number
        full_cleaned = text.strip().lower().replace(",", "").replace("$", "")
        numbers = re.findall(r"-?\d+\.?\d*", full_cleaned)
        if not numbers:
            return ExtractedAnswer(raw_text=text, normalized="", is_valid=False)
        val = numbers[-1].rstrip(".")
    else:
        # Use FIRST number on the answer line (model writes the answer first)
        val = numbers[0].rstrip(".")

    norm = _normalize_number(val)
    if norm is None:
        return ExtractedAnswer(raw_text=text, normalized="", is_valid=False)

    return ExtractedAnswer(raw_text=text, normalized=norm, is_valid=True)


def extract_gsm8k_gold(answer_field: str) -> str:
    """GSM8K gold answers are formatted '... #### 42'."""
    if "####" in answer_field:
        tail = answer_field.split("####")[-1].strip().replace(",", "")
        return tail
    return answer_field.strip().replace(",", "")


# ---------------------------------------------------------------------------
# MMLU-STEM: multiple choice A-D
# ---------------------------------------------------------------------------
_LETTER_PATTERNS = [
    r"answer is[:\s]*\(?([A-D])\)?",      # "the answer is (B)" -- check first,
                                           # more specific than bare letter
    r"\(([A-D])\)",                        # "(B)"
    r"^([A-D])[).:\s]",                   # "B." or "B)" or "B:" at line start
    r"\b([A-D])\b",                        # bare letter anywhere -- last resort
]


def extract_mmlu_answer(text: str, choices_text: str = None) -> ExtractedAnswer:
    """
    Extraction priority (documented, deterministic):
      1. Look for an explicit letter pattern (A-D) in the first 200 chars.
         Patterns are ordered most-to-least specific to avoid false positives
         (e.g. "A" appearing as an article shouldn't match before a real marker).
      2. If no letter found and choices_text provided, substring-match the
         generated text against each choice's content string.
      3. Otherwise mark invalid -- do NOT guess.
    """
    snippet = text.strip()[:200]

    for pat in _LETTER_PATTERNS:
        m = re.search(pat, snippet, re.IGNORECASE | re.MULTILINE)
        if m:
            return ExtractedAnswer(
                raw_text=text, normalized=m.group(1).upper(), is_valid=True
            )

    if choices_text:
        # choices_text format: "A. foo\nB. bar\nC. baz\nD. qux"
        for line in choices_text.split("\n"):
            line = line.strip()
            if not line or ". " not in line:
                continue
            letter, content = line.split(". ", 1)
            if content.strip().lower() in snippet.lower():
                return ExtractedAnswer(
                    raw_text=text, normalized=letter.strip().upper(), is_valid=True
                )

    return ExtractedAnswer(raw_text=text, normalized="", is_valid=False)


def extract_mmlu_gold(answer_index: int) -> str:
    return chr(65 + answer_index)  # 0 -> 'A', 1 -> 'B', ...


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

def extract_answer(task: str, text: str, choices_text: str = None) -> ExtractedAnswer:
    if not isinstance(text, str):
        # Guard against None being passed (e.g. from a failed elicitation)
        text = "" if text is None else str(text)
    if task == "gsm8k":
        return extract_gsm8k_answer(text)
    elif task == "mmlu_stem":
        return extract_mmlu_answer(text, choices_text)
    raise ValueError(f"Unknown task: {task}")
