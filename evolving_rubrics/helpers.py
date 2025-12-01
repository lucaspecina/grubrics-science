"""
Helper functions for DR-Tulu Evolving Rubrics.

Contains utility functions for JSON extraction and LLM communication.
"""

import json
import re
from typing import Dict, Optional, Any

from .config import (
    USE_AZURE,
    RUBRIC_GENERATION_MODEL,
    get_client
)


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract a JSON object from a text response.
    
    Tries multiple strategies:
    1. Extract JSON from code blocks (```json ... ```)
    2. Extract JSON directly from the text
    
    Args:
        response: Text response that may contain JSON
    
    Returns:
        Extracted JSON dictionary, or None if extraction fails
    """
    # Search for JSON in code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Search for direct JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


async def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None
) -> str:
    """
    Call an LLM asynchronously (supports OpenAI and Azure OpenAI).
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        model: Model name (defaults to RUBRIC_GENERATION_MODEL)
        client: Optional client instance (creates new one if not provided)
    
    Returns:
        LLM response text
    
    Raises:
        Exception: If the API call fails
    """
    if model is None:
        model = RUBRIC_GENERATION_MODEL
    
    if client is None:
        client = get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    if USE_AZURE:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=4000
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=4000
        )
    
    return response.choices[0].message.content

