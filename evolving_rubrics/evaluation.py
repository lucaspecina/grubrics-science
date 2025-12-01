"""
Evaluation module for DR-Tulu Evolving Rubrics.

Contains functions for evaluating responses against rubrics using LLM judges.
"""

import asyncio
from typing import Dict, Any, Optional

from .helpers import call_llm, extract_json_from_response
from .config import RUBRIC_JUDGE_MODEL
from .prompts import (
    RUBRIC_EVALUATION_SYSTEM_PROMPT,
    get_rubric_evaluation_prompt
)


async def evaluate_rubric(
    response: str,
    question: str,
    rubric_description: str,
    model: Optional[str] = None,
    client: Optional[Any] = None
) -> float:
    """
    Evaluate a response against a specific rubric criterion using an LLM judge.
    
    Args:
        response: The response to evaluate
        question: The original question
        rubric_description: Description of the criterion to evaluate
        model: Optional model name override
        client: Optional client instance
    
    Returns:
        Score between 0.0 and 1.0
    """
    system_prompt = RUBRIC_EVALUATION_SYSTEM_PROMPT
    user_prompt = get_rubric_evaluation_prompt(question, response, rubric_description)
    
    try:
        llm_response = await call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=model or RUBRIC_JUDGE_MODEL,
            client=client
        )
        
        result = extract_json_from_response(llm_response)
        
        if result and "score" in result:
            score = float(result["score"])
            # Normalize from 0-2 to 0-1
            return score / 2.0
        else:
            return 0.0
            
    except Exception as e:
        print(f"⚠️  Error evaluating rubric: {e}")
        return 0.0


async def evaluate_complete_response(
    response: str,
    ground_truth: Dict[str, Any],
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate a response against all rubrics and calculate total reward.
    
    Args:
        response: The response to evaluate
        ground_truth: Ground truth with all rubrics
        client: Optional client instance
    
    Returns:
        Dictionary with scores per rubric and total reward
    """
    question = ground_truth["query"]
    rubrics = ground_truth["rubrics"]
    
    # Evaluate each rubric
    scores_per_rubric = {}
    tasks = []
    
    for rubric in rubrics:
        description = rubric["description"]
        # Create unique key for the rubric
        key = rubric.get("title", description[:30])
        tasks.append((key, evaluate_rubric(response, question, description, client=client)))
    
    # Execute all evaluations in parallel
    results = await asyncio.gather(*[task[1] for task in tasks])
    
    for (key, _), score in zip(tasks, results):
        scores_per_rubric[key] = score
    
    # Calculate total reward (weighted average)
    total_reward = 0.0
    total_weight = 0.0
    
    for i, rubric in enumerate(rubrics):
        key = rubric.get("title", rubric["description"][:30])
        weight = abs(rubric["weight"])  # Use absolute value for weight
        score = scores_per_rubric[key]
        
        # Multiply by the sign of the weight (positive or negative)
        total_reward += score * rubric["weight"] * weight
        total_weight += weight
    
    final_reward = total_reward / total_weight if total_weight > 0 else 0.0
    
    return {
        "total_reward": final_reward,
        "scores_per_rubric": scores_per_rubric,
        "num_rubrics": len(rubrics)
    }

