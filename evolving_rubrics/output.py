"""
Output module for DR-Tulu Evolving Rubrics.

Handles saving evolution history and results to files for analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def save_evolution_history(
    result: Dict[str, Any],
    output_path: Optional[str] = None,
    output_dir: str = "outputs"
) -> str:
    """
    Save complete evolution history to a JSON file.
    
    The output file contains:
    - Initial rubrics
    - All iterations with:
      - Responses generated
      - Scores from Judge
      - Good vs bad response classification
      - Adaptive rubrics generated
      - Updated rubrics after each iteration
    - Final rubrics
    
    Args:
        result: Result dictionary from evolve_rubrics_for_example
        output_path: Optional custom output path. If None, auto-generates based on timestamp
        output_dir: Directory to save outputs (default: "outputs")
    
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir_path / f"evolution_history_{timestamp}.json"
    else:
        output_path = Path(output_path)
    
    # Prepare structured output
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_iterations": len(result.get("iterations", [])),
            "initial_rubrics_count": len(result.get("initial_ground_truth", {}).get("rubrics", [])),
            "final_rubrics_count": len(result.get("final_ground_truth", {}).get("rubrics", [])),
        },
        "question": result.get("initial_ground_truth", {}).get("query", ""),
        "initial_rubrics": _format_rubrics_for_comparison(result.get("initial_ground_truth", {})),
        "iterations": [],
        "final_rubrics": _format_rubrics_for_comparison(result.get("final_ground_truth", {})),
        "rubrics_evolution": _extract_rubrics_evolution(result)
    }
    
    # Process each iteration
    for iteration in result.get("iterations", []):
        iteration_data = {
            "iteration_number": iteration.get("iteration") or iteration.get("iteration_number"),
            "responses": [
                {
                    "response_num": eval_result.get("response_num"),
                    "response": eval_result.get("response", ""),
                    "total_reward": eval_result.get("total_reward"),
                    "scores_per_rubric": eval_result.get("scores_per_rubric", {}),
                    "classification": _classify_response(eval_result, iteration.get("evaluations", []))
                }
                for eval_result in iteration.get("evaluations", [])
            ],
            "rubrics_before": _get_rubrics_before_iteration(result, iteration.get("iteration") or iteration.get("iteration_number")),
            "adaptive_rubrics_generated": {
                "positive_rubrics": iteration.get("adaptive_rubrics", {}).get("positive_rubrics", []),
                "negative_rubrics": iteration.get("adaptive_rubrics", {}).get("negative_rubrics", [])
            },
            "rubrics_after": _format_rubrics_for_comparison(iteration.get("updated_ground_truth", {})),
            "summary": {
                "num_responses": len(iteration.get("responses", [])),
                "num_adaptive_rubrics": (
                    len(iteration.get("adaptive_rubrics", {}).get("positive_rubrics", [])) +
                    len(iteration.get("adaptive_rubrics", {}).get("negative_rubrics", []))
                ),
                "best_response_reward": max(
                    [e.get("total_reward", 0) for e in iteration.get("evaluations", [])],
                    default=0
                ),
                "worst_response_reward": min(
                    [e.get("total_reward", 0) for e in iteration.get("evaluations", [])],
                    default=0
                ),
                "avg_response_reward": sum(
                    [e.get("total_reward", 0) for e in iteration.get("evaluations", [])]
                ) / len(iteration.get("evaluations", [])) if iteration.get("evaluations") else 0
            }
        }
        output_data["iterations"].append(iteration_data)
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return str(output_path)


def _format_rubrics_for_comparison(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format rubrics in a way that's easy to compare.
    
    Returns:
        Dictionary with rubrics organized by type and with metadata
    """
    rubrics = ground_truth.get("rubrics", [])
    rubric_types = ground_truth.get("rubrics_types", [])
    
    formatted = {
        "total_count": len(rubrics),
        "persistent_count": sum(1 for t in rubric_types if t == "persistent"),
        "adaptive_count": sum(1 for t in rubric_types if t == "adaptive"),
        "rubrics": []
    }
    
    for i, (rubric, rubric_type) in enumerate(zip(rubrics, rubric_types)):
        formatted["rubrics"].append({
            "index": i,
            "type": rubric_type,
            "title": rubric.get("title", ""),
            "description": rubric.get("description", ""),
            "weight": rubric.get("weight", 1.0),
            "is_positive": rubric.get("weight", 1.0) > 0
        })
    
    return formatted


def _get_rubrics_before_iteration(result: Dict[str, Any], iteration_num: Optional[int]) -> Dict[str, Any]:
    """
    Get rubrics that existed before a given iteration.
    
    Args:
        result: Result dictionary from evolve_rubrics_for_example
        iteration_num: Iteration number (1-indexed)
    
    Returns:
        Formatted rubrics dictionary
    """
    if iteration_num is None or iteration_num == 1:
        return _format_rubrics_for_comparison(result.get("initial_ground_truth", {}))
    else:
        # Get rubrics from previous iteration
        iterations_list = result.get("iterations", [])
        if iteration_num - 2 < len(iterations_list):
            prev_iteration = iterations_list[iteration_num - 2]
            return _format_rubrics_for_comparison(prev_iteration.get("updated_ground_truth", {}))
        else:
            return _format_rubrics_for_comparison(result.get("initial_ground_truth", {}))


def _classify_response(eval_result: Dict[str, Any], all_evaluations: list) -> str:
    """
    Classify a response as good, bad, or medium based on its reward.
    
    Args:
        eval_result: Evaluation result for one response
        all_evaluations: All evaluation results for comparison
    
    Returns:
        "good", "bad", or "medium"
    """
    if not all_evaluations:
        return "unknown"
    
    rewards = [e.get("total_reward", 0) for e in all_evaluations]
    sorted_rewards = sorted(rewards, reverse=True)
    
    reward = eval_result.get("total_reward", 0)
    num_responses = len(rewards)
    
    # Top third = good, bottom third = bad, middle = medium
    if num_responses <= 2:
        return "good" if reward >= sorted_rewards[0] else "bad"
    
    threshold_good = sorted_rewards[num_responses // 3]
    threshold_bad = sorted_rewards[2 * num_responses // 3]
    
    if reward >= threshold_good:
        return "good"
    elif reward <= threshold_bad:
        return "bad"
    else:
        return "medium"


def _extract_rubrics_evolution(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract rubrics evolution across iterations for easy comparison.
    
    Returns:
        Dictionary with rubrics organized by iteration
    """
    evolution = {
        "iteration_0": {
            "description": "Initial rubrics",
            "rubrics": _format_rubrics_for_comparison(result.get("initial_ground_truth", {}))
        }
    }
    
    for iteration in result.get("iterations", []):
        iter_num = iteration.get("iteration") or iteration.get("iteration_number")
        evolution[f"iteration_{iter_num}"] = {
            "description": f"After iteration {iter_num}",
            "rubrics": _format_rubrics_for_comparison(iteration.get("updated_ground_truth", {})),
            "new_rubrics_this_iteration": {
                "positive": iteration.get("adaptive_rubrics", {}).get("positive_rubrics", []),
                "negative": iteration.get("adaptive_rubrics", {}).get("negative_rubrics", [])
            }
        }
    
    return evolution


def load_evolution_history(file_path: str) -> Dict[str, Any]:
    """
    Load evolution history from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dictionary with evolution history
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_rubrics_across_iterations(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare rubrics across iterations to see how they evolve.
    
    Args:
        history: Loaded evolution history
    
    Returns:
        Dictionary with comparison analysis
    """
    comparison = {
        "rubrics_count_evolution": [],
        "new_rubrics_per_iteration": [],
        "rubrics_by_type": {
            "persistent": [],
            "adaptive": []
        }
    }
    
    # Track rubrics count
    initial_count = history.get("initial_rubrics", {}).get("total_count", 0)
    comparison["rubrics_count_evolution"].append({
        "iteration": 0,
        "count": initial_count,
        "persistent": initial_count,
        "adaptive": 0
    })
    
    # Process iterations
    for iteration in history.get("iterations", []):
        iter_num = iteration.get("iteration_number")
        rubrics_after = iteration.get("rubrics_after", {})
        
        comparison["rubrics_count_evolution"].append({
            "iteration": iter_num,
            "count": rubrics_after.get("total_count", 0),
            "persistent": rubrics_after.get("persistent_count", 0),
            "adaptive": rubrics_after.get("adaptive_count", 0)
        })
        
        # Track new rubrics
        adaptive = iteration.get("adaptive_rubrics_generated", {})
        comparison["new_rubrics_per_iteration"].append({
            "iteration": iter_num,
            "positive": len(adaptive.get("positive_rubrics", [])),
            "negative": len(adaptive.get("negative_rubrics", [])),
            "total": (
                len(adaptive.get("positive_rubrics", [])) +
                len(adaptive.get("negative_rubrics", []))
            )
        })
    
    return comparison

