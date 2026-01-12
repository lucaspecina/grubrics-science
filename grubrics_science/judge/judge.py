"""Fixed Judge wrapper for evaluating answers with rubrics."""

import json
import re
from typing import List, Dict, Any, Optional
import asyncio

from ..llm.client import AzureOpenAIClient
from ..llm.prompts import JUDGE_SYSTEM_PROMPT, get_judge_prompt


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON directly
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


class Judge:
    """Fixed Judge for evaluating answers with rubrics."""
    
    def __init__(
        self,
        client: Optional[AzureOpenAIClient] = None,
        model: str = "gpt-4o-mini",
        use_azure: bool = True
    ):
        """
        Initialize Judge.
        
        Args:
            client: Optional Azure OpenAI client (creates new if None)
            model: Model name for judge
            use_azure: Whether to use Azure OpenAI
        """
        self.client = client or AzureOpenAIClient(model=model, use_azure=use_azure)
    
    async def evaluate_batch(
        self,
        question: str,
        answer: str,
        rubrics: List[str],
        answer_id: str = "a1"
    ) -> List[float]:
        """
        Evaluate an answer against multiple rubrics in a single call.
        
        Args:
            question: The original question
            answer: The answer to evaluate
            rubrics: List of rubric strings
            answer_id: Identifier for this answer
        
        Returns:
            List of scores (one per rubric), in order
        """
        prompt = get_judge_prompt(question, answer, rubrics, answer_id)
        
        response = await self.client.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=2000,  # Enough for multiple rubric scores
            temperature=0.0  # Deterministic scoring
        )
        
        # Parse JSON response
        result = extract_json_from_response(response)
        
        if not result or 'rubric_scores' not in result:
            # Fallback: return zeros if parsing fails
            print(f"Warning: Failed to parse judge response. Response: {response[:200]}...")
            return [0.0] * len(rubrics)
        
        # Extract scores in order
        rubric_scores = result['rubric_scores']
        scores_dict = {item['rubric_id']: item['score'] for item in rubric_scores}
        
        # Map to list in order (r1, r2, ...)
        scores = []
        for i in range(len(rubrics)):
            rubric_id = f"r{i+1}"
            score = scores_dict.get(rubric_id, 0.0)
            # Ensure score is in [0, 1]
            score = max(0.0, min(1.0, float(score)))
            scores.append(score)
        
        return scores
    
    async def evaluate_multiple_answers(
        self,
        question: str,
        answers: List[str],
        rubrics: List[str]
    ) -> List[List[float]]:
        """
        Evaluate multiple answers against multiple rubrics.
        
        Returns a matrix of shape [num_answers, num_rubrics].
        Each row corresponds to one answer, each column to one rubric.
        
        Args:
            question: The original question
            answers: List of answers to evaluate
            rubrics: List of rubric strings
        
        Returns:
            List of lists: scores[i][j] = score of answer i under rubric j
        """
        # Evaluate each answer independently (can be parallelized)
        tasks = [
            self.evaluate_batch(question, answer, rubrics, answer_id=f"a{i+1}")
            for i, answer in enumerate(answers)
        ]
        
        results = await asyncio.gather(*tasks)
        return results

