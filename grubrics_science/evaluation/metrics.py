"""Evaluation metrics for rubric quality.

All metrics operate on a single rubric evaluated against a set of answers
with known gold_scores. Higher is better for all metrics except where noted.
"""

import re
from typing import List

import numpy as np
from scipy.stats import spearmanr


def alignment_score(
    rubric_scores: List[float],
    gold_scores: List[float],
) -> float:
    """Spearman rank correlation between rubric scores and gold scores.

    This is the primary metric: does the rubric rank answers the same way
    as the gold standard (human rubric or programmatic correctness)?

    Returns:
        Value in [-1, 1]. 1.0 = perfect agreement. 0.0 = no correlation.
    """
    if len(rubric_scores) < 2 or len(rubric_scores) != len(gold_scores):
        return 0.0
    try:
        corr, _ = spearmanr(rubric_scores, gold_scores)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def discrimination_score(rubric_scores: List[float]) -> float:
    """Standard deviation of rubric scores across answers.

    A rubric that gives the same score to all answers is useless.
    Higher std = more discriminative.

    Returns:
        Value in [0, inf). 0.0 = degenerate (all same score).
    """
    if len(rubric_scores) < 2:
        return 0.0
    return float(np.std(rubric_scores))


def format_validity(rubric_text: str) -> float:
    """Fraction of lines that match the required format.

    Required format: ``Points: <number>, Item: <text>``

    Returns:
        Value in [0, 1]. 1.0 = all lines valid.
    """
    lines = [l.strip() for l in rubric_text.strip().split("\n") if l.strip()]
    if not lines:
        return 0.0

    pattern = re.compile(r"^Points:\s*[\d.]+\s*,\s*Item:\s*.+")
    valid = sum(1 for l in lines if pattern.match(l))
    return valid / len(lines)


def points_sum(rubric_text: str) -> float:
    """Sum of all Points values in the rubric.

    Target is 10.0. Useful for checking rubric completeness.
    """
    pattern = re.compile(r"Points:\s*([\d.]+)")
    matches = pattern.findall(rubric_text)
    return sum(float(m) for m in matches)


def info_value(rubric_scores: List[float], threshold: float = 0.5) -> float:
    """Measures how discriminative the rubric's criteria are.

    ``4 * p * (1 - p)`` where p = fraction of answers scoring above threshold.
    Maximised at p = 0.5 (half pass, half fail).

    Returns:
        Value in [0, 1]. 1.0 = maximally discriminative.
    """
    if len(rubric_scores) < 2:
        return 0.0
    p = sum(1 for s in rubric_scores if s >= threshold) / len(rubric_scores)
    return 4.0 * p * (1.0 - p)


def rubric_length(rubric_text: str) -> int:
    """Character length of the rubric."""
    return len(rubric_text)


def compute_all_metrics(
    rubric_text: str,
    rubric_scores: List[float],
    gold_scores: List[float],
) -> dict:
    """Compute all metrics for a single rubric on a single question.

    Args:
        rubric_text: The rubric text.
        rubric_scores: Scores assigned by the Judge using this rubric (one per answer).
        gold_scores: Gold standard scores (one per answer).

    Returns:
        Dict with keys: alignment, discrimination, format_validity,
        points_sum, info_value, length.
    """
    return {
        "alignment": alignment_score(rubric_scores, gold_scores),
        "discrimination": discrimination_score(rubric_scores),
        "format_validity": format_validity(rubric_text),
        "points_sum": points_sum(rubric_text),
        "info_value": info_value(rubric_scores),
        "length": rubric_length(rubric_text),
    }
