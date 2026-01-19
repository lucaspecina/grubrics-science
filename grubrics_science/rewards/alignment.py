"""Alignment metrics and reward computation."""

from typing import List
import numpy as np
from scipy.stats import spearmanr, pearsonr


def pairwise_accuracy(scores: List[float], gold_scores: List[float]) -> float:
    """
    Compute pairwise accuracy: fraction of pairs where ordering matches.
    
    Args:
        scores: Predicted scores
        gold_scores: Ground truth scores
    
    Returns:
        Pairwise accuracy in [0, 1]
    """
    if len(scores) != len(gold_scores):
        raise ValueError("scores and gold_scores must have same length")
    
    if len(scores) < 2:
        return 1.0  # Trivial case
    
    scores = np.array(scores)
    gold_scores = np.array(gold_scores)
    
    # Count pairs where ordering matches
    matches = 0
    total_pairs = 0
    
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            # Skip ties in gold scores
            if gold_scores[i] == gold_scores[j]:
                continue
            
            total_pairs += 1
            gold_order = gold_scores[i] > gold_scores[j]
            pred_order = scores[i] > scores[j]
            
            if gold_order == pred_order:
                matches += 1
    
    if total_pairs == 0:
        return 1.0  # All ties, consider perfect
    
    return matches / total_pairs


def spearman_correlation(scores: List[float], gold_scores: List[float]) -> float:
    """
    Compute Spearman rank correlation.
    
    Args:
        scores: Predicted scores
        gold_scores: Ground truth scores
    
    Returns:
        Spearman correlation in [-1, 1] (returns 0.0 if correlation cannot be computed)
    """
    if len(scores) != len(gold_scores):
        raise ValueError("scores and gold_scores must have same length")
    
    if len(scores) < 2:
        return 1.0
    
    try:
        corr, _ = spearmanr(scores, gold_scores)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def pearson_correlation(scores: List[float], gold_scores: List[float]) -> float:
    """
    Compute Pearson correlation.
    
    Args:
        scores: Predicted scores
        gold_scores: Ground truth scores
    
    Returns:
        Pearson correlation in [-1, 1] (returns 0.0 if correlation cannot be computed)
    """
    if len(scores) != len(gold_scores):
        raise ValueError("scores and gold_scores must have same length")
    
    if len(scores) < 2:
        return 1.0
    
    try:
        corr, _ = pearsonr(scores, gold_scores)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def compute_alignment(
    scores: List[float],
    gold_scores: List[float],
    metric: str = "spearman"
) -> float:
    """
    Compute alignment metric between scores and gold scores.
    
    Args:
        scores: Predicted scores from a rubric
        gold_scores: Ground truth scores
        metric: Metric to use ("spearman", "pairwise", "pearson")
    
    Returns:
        Alignment score (higher is better, typically in [-1, 1] or [0, 1])
    """
    if metric == "spearman":
        return spearman_correlation(scores, gold_scores)
    elif metric == "pairwise":
        return pairwise_accuracy(scores, gold_scores)
    elif metric == "pearson":
        return pearson_correlation(scores, gold_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def length_penalty(rubric_text: str, penalty_type: str = "characters") -> float:
    """
    Compute length penalty for a rubric.
    
    Args:
        rubric_text: The rubric text
        penalty_type: "characters" or "tokens"
    
    Returns:
        Penalty value (positive, higher = longer = more penalty)
    """
    if penalty_type == "characters":
        return len(rubric_text)
    elif penalty_type == "tokens":
        # Rough estimate: ~4 characters per token
        return len(rubric_text) / 4.0
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")


def compute_reward(
    scores: List[float],
    gold_scores: List[float],
    rubric_text: str,
    alignment_metric: str = "spearman",
    lambda_len: float = 0.01,
    length_penalty_type: str = "characters"
) -> float:
    """
    Compute reward for a rubric.
    
    Reward = alignment(scores, gold_scores) - lambda_len * length_penalty(rubric_text)
    
    Args:
        scores: Scores produced by this rubric
        gold_scores: Ground truth scores
        rubric_text: The rubric text
        alignment_metric: Metric to use for alignment
        lambda_len: Length penalty coefficient
        length_penalty_type: Type of length penalty
    
    Returns:
        Reward value
    """
    alignment = compute_alignment(scores, gold_scores, metric=alignment_metric)
    penalty = length_penalty(rubric_text, penalty_type=length_penalty_type)
    
    reward = alignment - lambda_len * penalty
    
    return reward

