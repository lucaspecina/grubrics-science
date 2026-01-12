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
        answer_id: str = "a1",
        return_details: bool = True
    ) -> tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate an answer against multiple rubrics in a single call.
        
        Args:
            question: The original question
            answer: The answer to evaluate
            rubrics: List of rubric strings
            answer_id: Identifier for this answer
            return_details: If True, return detailed explanations for each rubric and item
        
        Returns:
            Tuple of (scores, details) where:
            - scores: List of total scores (one per rubric), in order
            - details: List of dicts with 'rubric_id', 'total_score', 'item_scores' for each rubric
              Each 'item_scores' is a list of dicts with 'item_id', 'item_description', 'max_points', 'score', 'notes'
        """
        prompt = get_judge_prompt(question, answer, rubrics, answer_id)
        
        response = await self.client.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=5000,  # Increased for detailed item-by-item explanations
            temperature=0.0  # Deterministic scoring
        )
        
        # Parse JSON response
        result = extract_json_from_response(response)
        
        if not result or 'rubric_evaluations' not in result:
            # Fallback: return zeros if parsing fails
            print(f"Warning: Failed to parse judge response. Response: {response[:500]}...")
            fallback_details = [
                {
                    "rubric_id": f"r{i+1}",
                    "total_score": 0.0,
                    "item_scores": [],
                    "notes": "Failed to parse judge response"
                }
                for i in range(len(rubrics))
            ]
            return [0.0] * len(rubrics), fallback_details
        
        # Extract scores and details in order
        rubric_evaluations = result['rubric_evaluations']
        evaluations_dict = {eval_item['rubric_id']: eval_item for eval_item in rubric_evaluations}
        
        # Map to lists in order (r1, r2, ...)
        scores = []
        details = []
        for i in range(len(rubrics)):
            rubric_id = f"r{i+1}"
            eval_item = evaluations_dict.get(rubric_id, {
                "total_score": 0.0,
                "item_scores": [],
                "notes": "Rubric not found in response"
            })
            
            total_score = float(eval_item.get('total_score', 0.0))
            # Ensure score is in [0, 1]
            total_score = max(0.0, min(1.0, total_score))
            scores.append(total_score)
            
            if return_details:
                item_scores = eval_item.get('item_scores', [])
                details.append({
                    "rubric_id": rubric_id,
                    "total_score": total_score,
                    "item_scores": item_scores,
                    "notes": f"Evaluated {len(item_scores)} items"
                })
            else:
                details.append({
                    "rubric_id": rubric_id,
                    "total_score": total_score,
                    "item_scores": []
                })
        
        return scores, details
    
    async def evaluate_multiple_answers(
        self,
        question: str,
        answers: List[str],
        rubrics: List[str],
        return_details: bool = True
    ) -> tuple[List[List[float]], List[List[Dict[str, Any]]]]:
        """
        Evaluate multiple answers against multiple rubrics.
        
        Returns a matrix of shape [num_answers, num_rubrics].
        Each row corresponds to one answer, each column to one rubric.
        
        Args:
            question: The original question
            answers: List of answers to evaluate
            rubrics: List of rubric strings
            return_details: If True, return detailed explanations
        
        Returns:
            Tuple of (score_matrix, details_matrix) where:
            - score_matrix: List of lists: scores[i][j] = score of answer i under rubric j
            - details_matrix: List of lists: details[i][j] = detailed evaluation for answer i, rubric j
        """
        # Evaluate each answer independently (can be parallelized)
        tasks = [
            self.evaluate_batch(question, answer, rubrics, answer_id=f"a{i+1}", return_details=return_details)
            for i, answer in enumerate(answers)
        ]
        
        results = await asyncio.gather(*tasks)
        # Unpack scores and details
        score_matrix = [scores for scores, _ in results]
        details_matrix = [details for _, details in results]
        return score_matrix, details_matrix

