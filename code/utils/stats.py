"""
utils/stats.py
===============
ACR should distinguish right->wrong / wrong->right / wrong->wrong transitions, not just report a flat change rate.
"""

import math
import random
from typing import List, Tuple, Dict


def bootstrap_ci(binary_outcomes: List[bool], n_boot: int = 5000, alpha: float = 0.05,
                  seed: int = 0) -> Tuple[float, float, float]:
    """Returns (point_estimate_pct, ci_low_pct, ci_high_pct) via percentile
    bootstrap over the sample of binary 'changed' outcomes."""
    n = len(binary_outcomes)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    point = 100.0 * sum(binary_outcomes) / n
    boots = []
    for _ in range(n_boot):
        sample = [binary_outcomes[rng.randrange(n)] for _ in range(n)]
        boots.append(100.0 * sum(sample) / n)
    boots.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return point, boots[lo_idx], boots[hi_idx]


def bootstrap_paired_diff_ci(outcomes_condition: List[bool], outcomes_baseline: List[bool],
                              n_boot: int = 5000, alpha: float = 0.05,
                              seed: int = 0) -> Tuple[float, float, float]:
    """
    Noise-floor-adjusted ACR, done correctly: bootstraps the PAIRED per-example
    difference (condition_changed - baseline_changed) directly, rather than
    bootstrapping each condition's raw rate independently and subtracting a
    point estimate afterward (which mismatches the point estimate's basis
    against the CI's basis -- the bug that produced "zero bar, huge whisker"
    artifacts when a condition was accidentally identical to the baseline).

    outcomes_condition and outcomes_baseline must be the SAME LENGTH and
    index-aligned to the same examples (e.g. via per-example majority vote
    across seeds). Returns signed (point_estimate_pct, ci_low_pct, ci_high_pct)
    -- NOT clipped at zero, since a small negative value is a legitimate
    (if uninteresting) result: it means the condition changed the answer
    LESS than the noise floor itself did on this sample, which can happen by
    chance and should be visible rather than hidden by clipping.
    """
    assert len(outcomes_condition) == len(outcomes_baseline), \
        "Paired bootstrap requires equal-length, aligned samples"
    n = len(outcomes_condition)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    diffs = [100.0 * (int(c) - int(b)) for c, b in zip(outcomes_condition, outcomes_baseline)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return point, boots[lo_idx], boots[hi_idx]


def mcnemar_test(outcomes_a: List[bool], outcomes_b: List[bool]) -> Dict:
    """
    Paired McNemar's test (continuity-corrected chi-square) comparing two
    paired binary conditions on the SAME examples -- e.g. "did the answer
    change under removal" vs "did the answer change under swap" for the
    same questions. Used to check whether reported differences between
    interventions (e.g. Section 3.2's removal > perturb > swap ordering)
    are actually statistically distinguishable.
    """
    assert len(outcomes_a) == len(outcomes_b), "Paired test requires equal-length, aligned samples"
    b = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if a and not bb)  # a=True, b=False
    c = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if not a and bb)  # a=False, b=True
    if b + c == 0:
        return dict(statistic=0.0, p_value=1.0, b=b, c=c)
    stat = ((abs(b - c) - 1) ** 2) / (b + c)  # continuity-corrected
    # chi-square with 1 dof -> survival function via complementary error function
    p_value = math.erfc(math.sqrt(stat / 2))
    return dict(statistic=stat, p_value=p_value, b=b, c=c)


def transition_breakdown(gold: List[str], pred_before: List[str], pred_after: List[str]) -> Dict:
    """
    Reviewer Y9xk: break ACR into right->wrong, wrong->right, wrong->wrong,
    right->right (no change, correct), instead of one flat change rate.
    """
    counts = {"right_to_wrong": 0, "wrong_to_right": 0, "wrong_to_wrong_diff": 0,
              "no_change_correct": 0, "no_change_incorrect": 0}
    n = len(gold)
    for g, before, after in zip(gold, pred_before, pred_after):
        before_correct = (before == g)
        after_correct = (after == g)
        changed = (before != after)
        if not changed:
            counts["no_change_correct" if before_correct else "no_change_incorrect"] += 1
        else:
            if before_correct and not after_correct:
                counts["right_to_wrong"] += 1
            elif not before_correct and after_correct:
                counts["wrong_to_right"] += 1
            else:
                counts["wrong_to_wrong_diff"] += 1
    return {k: (100.0 * v / n if n else float("nan")) for k, v in counts.items()}
