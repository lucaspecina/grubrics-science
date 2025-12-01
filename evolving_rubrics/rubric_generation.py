"""
Rubric generation module for DR-Tulu Evolving Rubrics.

Contains functions for generating initial rubrics and adaptive rubrics.
"""

from typing import List, Dict, Any, Optional

from .helpers import call_llm, extract_json_from_response
from .config import RUBRIC_GENERATION_MODEL
from .prompts import (
    get_original_rubrics_prompt,
    get_adaptive_rubrics_prompt
)


async def generate_original_rubrics(
    question: str,
    model: Optional[str] = None,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generate initial rubrics for a question using an LLM.
    
    Args:
        question: The question to generate rubrics for
        model: Optional model name override
        client: Optional client instance
    
    Returns:
        Dictionary with 'query' and 'rubrics' keys
    """
    if model is None:
        model = RUBRIC_GENERATION_MODEL
    
    prompt = get_original_rubrics_prompt(question)

    try:
        llm_response = await call_llm(
            prompt=prompt,
            model=model,
            client=client
        )
        
        rubrics = extract_json_from_response(llm_response)
        
        if rubrics and "rubrics" in rubrics:
            return rubrics
        else:
            print(f"⚠️  Could not extract valid rubrics")
            print(f"Response received: {llm_response[:300]}...")
            # Return basic structure if extraction fails
            return {
                "query": question,
                "rubrics": [
                    {
                        "title": "Relevant response",
                        "description": "The response must be relevant to the question",
                        "weight": 1.0
                    }
                ]
            }
    except Exception as e:
        print(f"❌ Error generating original rubrics: {e}")
        return {
            "query": question,
            "rubrics": [
                {
                    "title": "Relevant response",
                    "description": "The response must be relevant to the question",
                    "weight": 1.0
                }
            ]
        }


async def generate_adaptive_rubrics(
    question: str,
    responses: List[str],
    existing_rubrics: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate adaptive rubrics based on differences between model responses.
    
    Args:
        question: The original question
        responses: List of model responses to analyze
        existing_rubrics: Optional existing rubrics to avoid duplicating
        model: Optional model name override
        client: Optional client instance
    
    Returns:
        Dictionary with 'positive_rubrics' and 'negative_rubrics' keys, or None if failed
    """
    full_prompt = get_adaptive_rubrics_prompt(question, responses, existing_rubrics)
    
    try:
        # Call the LLM
        llm_response = await call_llm(
            prompt=full_prompt,
            model=model or RUBRIC_GENERATION_MODEL,
            client=client
        )
        
        # Extract JSON
        rubrics = extract_json_from_response(llm_response)
        
        if rubrics:
            return rubrics
        else:
            print(f"⚠️  Could not extract JSON from LLM response")
            print(f"Response received: {llm_response[:200]}...")
            return None
            
    except Exception as e:
        print(f"❌ Error generating adaptive rubrics: {e}")
        return None


def update_ground_truth(
    initial_ground_truth: Dict[str, Any],
    adaptive_rubrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update ground truth by combining initial rubrics with adaptive rubrics.
    
    Args:
        initial_ground_truth: Initial ground truth with persistent rubrics
        adaptive_rubrics: New adaptive rubrics to add
    
    Returns:
        Updated ground truth dictionary
    """
    updated_ground_truth = initial_ground_truth.copy()
    
    # Get original persistent rubrics
    persistent_rubrics = initial_ground_truth.get("rubrics", [])
    
    # Convert adaptive rubrics to correct format
    new_rubrics = []
    
    # Positive rubrics (weight +1.0)
    for rubric in adaptive_rubrics.get("positive_rubrics", []):
        new_rubrics.append({
            "description": rubric["description"],
            "weight": 1.0,
            "title": rubric.get("title", "Untitled")
        })
    
    # Negative rubrics (weight -1.0)
    for rubric in adaptive_rubrics.get("negative_rubrics", []):
        new_rubrics.append({
            "description": rubric["description"],
            "weight": -1.0,
            "title": rubric.get("title", "Untitled")
        })
    
    # Combine: persistent first, then adaptive
    updated_ground_truth["rubrics"] = persistent_rubrics + new_rubrics
    
    # Add types (optional, for tracking)
    types = ["persistent"] * len(persistent_rubrics) + ["adaptive"] * len(new_rubrics)
    updated_ground_truth["rubrics_types"] = types
    
    return updated_ground_truth

