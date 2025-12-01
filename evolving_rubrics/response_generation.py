"""
Response generation module for DR-Tulu Evolving Rubrics.

Contains functions for generating model responses to questions.
"""

from typing import List, Optional, Any

from .helpers import call_llm
from .config import RUBRIC_GENERATION_MODEL
from .prompts import (
    RESPONSE_GENERATION_INSTRUCTIONS,
    get_response_generation_prompt
)


async def generate_model_responses(
    question: str,
    num_responses: int = 4,
    model: Optional[str] = None,
    client: Optional[Any] = None
) -> List[str]:
    """
    Generate multiple model responses to a question with varied instructions.
    
    Args:
        question: The question to answer
        num_responses: Number of responses to generate
        model: Optional model name override
        client: Optional client instance
    
    Returns:
        List of response strings
    """
    if model is None:
        model = RUBRIC_GENERATION_MODEL
    
    responses = []
    
    for i in range(num_responses):
        instruction = RESPONSE_GENERATION_INSTRUCTIONS[i % len(RESPONSE_GENERATION_INSTRUCTIONS)]
        prompt = get_response_generation_prompt(question, instruction)

        try:
            response = await call_llm(
                prompt=prompt,
                model=model,
                client=client
            )
            responses.append(response.strip())
        except Exception as e:
            print(f"Warning: Error generating response {i+1}: {e}")
            responses.append(f"Error generating response {i+1}")
    
    return responses

