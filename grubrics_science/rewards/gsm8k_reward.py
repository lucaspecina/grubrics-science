"""Local reward function for verifiable domains (GSM8K, MATH, olympiad_math).

This reward checks whether a generated rubric has correct format and
basic coherence, without requiring any external API calls.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_rubric_items(rubric_text: str) -> List[Tuple[float, str]]:
    """Parse rubric text into (points, item_text) tuples.

    Expects lines like: ``Points: 1.5, Item: The answer derives ...``

    Returns:
        List of (points, item_text) tuples.  Empty list if parsing fails.
    """
    pattern = r"Points:\s*([\d.]+)\s*,\s*Item:\s*(.+)"
    items = []
    for match in re.finditer(pattern, rubric_text):
        try:
            points = float(match.group(1))
            text = match.group(2).strip()
            items.append((points, text))
        except ValueError:
            continue
    return items


def format_score(rubric_text: str) -> float:
    """Score how well the rubric follows the expected format.

    Returns a score in [0, 1]:
        - 0.0 if no valid items found
        - partial credit for having items but wrong total
        - 1.0 if items sum to exactly 10.0 (within tolerance)
    """
    items = parse_rubric_items(rubric_text)

    if not items:
        return 0.0

    total_points = sum(p for p, _ in items)
    num_items = len(items)

    score = 0.0

    # Credit for having valid items (0 to 0.3)
    item_credit = min(num_items / 5.0, 1.0) * 0.3
    score += item_credit

    # Credit for reasonable number of items: 3-10 is ideal (0 to 0.2)
    if 3 <= num_items <= 10:
        score += 0.2
    elif num_items >= 2:
        score += 0.1

    # Credit for points summing to 10.0 (0 to 0.5)
    if abs(total_points - 10.0) < 0.01:
        score += 0.5
    elif abs(total_points - 10.0) < 1.0:
        score += 0.3
    elif abs(total_points - 10.0) < 2.0:
        score += 0.1

    return min(score, 1.0)


def coherence_score(rubric_text: str, question: str) -> float:
    """Score basic coherence of rubric items relative to the question.

    Simple heuristic checks:
        - Items are non-empty and not trivially short
        - Items mention something related to the question domain
        - No duplicate items

    Returns a score in [0, 1].
    """
    items = parse_rubric_items(rubric_text)
    if not items:
        return 0.0

    scores = []

    # Check item text quality
    for _, text in items:
        item_score = 0.0
        # Non-trivially short
        if len(text) > 20:
            item_score += 0.5
        elif len(text) > 10:
            item_score += 0.25
        # Contains meaningful evaluative language
        eval_words = [
            "answer", "solution", "correct", "derive", "compute",
            "calculate", "explain", "identify", "show", "prove",
            "result", "formula", "equation", "value", "step",
        ]
        if any(w in text.lower() for w in eval_words):
            item_score += 0.5
        scores.append(min(item_score, 1.0))

    # Check for duplicates
    texts_lower = [t.lower().strip() for _, t in items]
    unique_ratio = len(set(texts_lower)) / len(texts_lower) if texts_lower else 0.0

    avg_item_quality = sum(scores) / len(scores) if scores else 0.0
    return avg_item_quality * 0.7 + unique_ratio * 0.3


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = "",
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute reward for a rubric generated for a verifiable-domain question.

    This is the reward function used during Phase 0 training with veRL.
    It does NOT require external API calls.

    Args:
        data_source: Dataset identifier (e.g. ``"gsm8k"``, ``"math"``).
        solution_str: The generated rubric text.
        ground_truth: The correct answer (not directly used for scoring,
            but available for future extensions).
        extra_info: Dict with ``"question"`` and other metadata.

    Returns:
        Reward in [0, 1].
    """
    extra_info = extra_info or {}
    rubric_text = solution_str
    question = extra_info.get("question", "")

    # Format check (weight: 0.6)
    fmt = format_score(rubric_text)

    # Coherence check (weight: 0.4)
    coh = coherence_score(rubric_text, question)

    reward = 0.6 * fmt + 0.4 * coh
    return reward
