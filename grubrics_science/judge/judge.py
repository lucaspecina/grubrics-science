"""Judge wrapper for evaluating answers with rubrics.

Includes rate limiting, retry with backoff, and response caching
for use inside RL training loops.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..llm.client import AzureOpenAIClient
from ..llm.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_BATCHED_SYSTEM_PROMPT,
    get_judge_prompt,
    get_judge_batched_prompt,
)

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling truncation and extra text."""
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

    # Handle truncated JSON: find opening brace and try to repair
    brace_pos = response.find('{')
    if brace_pos >= 0:
        fragment = response[brace_pos:]
        # Try closing truncated arrays/objects incrementally
        for suffix in (']]}', ']}', '}'):
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                continue

    return None


def _cache_key(question: str, answer: str, rubrics: List[str]) -> str:
    """Deterministic cache key for a (question, answer, rubrics) tuple."""
    content = json.dumps(
        {"q": question, "a": answer, "r": rubrics},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(content.encode()).hexdigest()


class Judge:
    """Judge for evaluating answers with rubrics.

    Features:
        - Rate limiting via asyncio.Semaphore (max concurrent API calls)
        - Retry with exponential backoff on transient failures
        - In-memory cache to avoid duplicate API calls within a run
    """

    def __init__(
        self,
        client: Optional[AzureOpenAIClient] = None,
        model: str = "gpt-4o-mini",
        use_azure: bool = True,
        max_concurrent: int = 10,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_cache_size: int = 0,
    ):
        """
        Args:
            client: Optional Azure OpenAI client (creates new if None).
            model: Model name for judge.
            use_azure: Whether to use Azure OpenAI.
            max_concurrent: Max parallel API calls (semaphore size).
            max_retries: Number of retry attempts on failure.
            timeout: Timeout in seconds per API call.
            max_cache_size: Max cache entries (0 = disabled). Use 0 during RL
                training to avoid unbounded memory growth; rubrics are unique each step.
        """
        self.client = client or AzureOpenAIClient(model=model, use_azure=use_azure)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._timeout = timeout
        self._max_cache_size = max_cache_size
        self._cache: Dict[str, Tuple[List[float], List[Dict[str, Any]]]] = {}
        # Timing stats for training diagnostics
        self._call_timings: List[Tuple[float, float]] = []  # (sem_wait, api_time)

    async def _call_with_retry(
        self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 5000,
    ) -> str:
        """Call the LLM with rate limiting, timeout, and retry."""
        system_prompt = system_prompt or JUDGE_SYSTEM_PROMPT
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                _t_enter = time.perf_counter()
                async with self._semaphore:
                    _t_acquired = time.perf_counter()
                    response = await asyncio.wait_for(
                        self.client.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                        ),
                        timeout=self._timeout,
                    )
                    _t_done = time.perf_counter()
                    self._call_timings.append((
                        _t_acquired - _t_enter,  # sem_wait
                        _t_done - _t_acquired,   # api_time
                    ))
                return response
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Judge API call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Judge API call failed after {self._max_retries} retries: {last_error}"
        )

    def get_and_reset_timings(self) -> List[Tuple[float, float]]:
        """Return accumulated (sem_wait, api_time) tuples and reset.

        Used by the reward function to log per-step timing summaries.
        """
        timings = list(self._call_timings)
        self._call_timings.clear()
        return timings

    def _parse_response(
        self, response: str, num_rubrics: int, return_details: bool
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Parse LLM JSON response into scores and details."""
        result = extract_json_from_response(response)

        if not result or "rubric_evaluations" not in result:
            logger.warning(
                "Failed to parse judge response: %s...", response[:300]
            )
            fallback_details = [
                {
                    "rubric_id": f"r{i + 1}",
                    "total_score": 0.0,
                    "item_scores": [],
                    "notes": "Failed to parse judge response",
                }
                for i in range(num_rubrics)
            ]
            return [0.0] * num_rubrics, fallback_details

        rubric_evaluations = result["rubric_evaluations"]
        evaluations_dict = {
            e["rubric_id"]: e for e in rubric_evaluations
        }

        scores: List[float] = []
        details: List[Dict[str, Any]] = []

        for i in range(num_rubrics):
            rubric_id = f"r{i + 1}"
            eval_item = evaluations_dict.get(
                rubric_id,
                {"total_score": 0.0, "item_scores": [], "notes": "Rubric not found"},
            )

            total_score = float(eval_item.get("total_score", 0.0))
            total_score = max(0.0, min(1.0, total_score))
            scores.append(total_score)

            if return_details:
                details.append({
                    "rubric_id": rubric_id,
                    "total_score": total_score,
                    "item_scores": eval_item.get("item_scores", []),
                    "notes": f"Evaluated {len(eval_item.get('item_scores', []))} items",
                })
            else:
                details.append({
                    "rubric_id": rubric_id,
                    "total_score": total_score,
                    "item_scores": [],
                })

        return scores, details

    async def evaluate_batch(
        self,
        question: str,
        answer: str,
        rubrics: List[str],
        answer_id: str = "a1",
        return_details: bool = True,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Evaluate an answer against multiple rubrics in a single call.

        Args:
            question: The original question.
            answer: The answer to evaluate.
            rubrics: List of rubric strings.
            answer_id: Identifier for this answer.
            return_details: If True, return detailed item-level explanations.

        Returns:
            (scores, details) — scores[j] is the total score for rubric j.
        """
        if self._max_cache_size > 0:
            key = _cache_key(question, answer, rubrics)
            if key in self._cache:
                return self._cache[key]

        prompt = get_judge_prompt(question, answer, rubrics, answer_id)
        response = await self._call_with_retry(prompt)
        scores, details = self._parse_response(response, len(rubrics), return_details)

        if self._max_cache_size > 0:
            key = _cache_key(question, answer, rubrics)
            if len(self._cache) < self._max_cache_size:
                self._cache[key] = (scores, details)
        return scores, details

    async def evaluate_multiple_answers(
        self,
        question: str,
        answers: List[str],
        rubrics: List[str],
        return_details: bool = True,
    ) -> Tuple[List[List[float]], List[List[Dict[str, Any]]]]:
        """Evaluate multiple answers against multiple rubrics.

        Returns matrices of shape [num_answers, num_rubrics].

        Args:
            question: The original question.
            answers: List of answers to evaluate.
            rubrics: List of rubric strings.
            return_details: If True, return detailed explanations.

        Returns:
            (score_matrix, details_matrix)
        """
        tasks = [
            self.evaluate_batch(
                question, answer, rubrics,
                answer_id=f"a{i + 1}",
                return_details=return_details,
            )
            for i, answer in enumerate(answers)
        ]

        results = await asyncio.gather(*tasks)
        score_matrix = [scores for scores, _ in results]
        details_matrix = [details for _, details in results]
        return score_matrix, details_matrix

    async def evaluate_answers_batched(
        self,
        question: str,
        answers: List[str],
        rubric: str,
    ) -> List[float]:
        """Evaluate multiple answers against one rubric in a single API call.

        More efficient than evaluate_multiple_answers for the common case
        of one rubric + many answers (1 call instead of N). The model can
        also compare answers in context, producing more consistent rankings.

        Args:
            question: The original question.
            answers: List of answers to evaluate.
            rubric: Single rubric string.

        Returns:
            List of total_scores, one per answer.
        """
        if self._max_cache_size > 0:
            key = _cache_key(question, json.dumps(answers), [rubric])
            if key in self._cache:
                cached_scores, _ = self._cache[key]
                return cached_scores

        prompt = get_judge_batched_prompt(question, answers, rubric)

        response = await self._call_with_retry(
            prompt,
            system_prompt=JUDGE_BATCHED_SYSTEM_PROMPT,
            max_tokens=4000,
        )

        scores = self._parse_batched_response(response, len(answers))

        if self._max_cache_size > 0:
            key = _cache_key(question, json.dumps(answers), [rubric])
            if len(self._cache) < self._max_cache_size:
                self._cache[key] = (scores, [])
        return scores

    def _parse_batched_response(
        self, response: str, num_answers: int
    ) -> List[float]:
        """Parse batched Judge response into a list of scores."""
        result = extract_json_from_response(response)

        if not result or "evaluations" not in result:
            logger.warning(
                "Failed to parse batched judge response: %s...", response[:300]
            )
            return [0.0] * num_answers

        evaluations = result["evaluations"]

        # Build dict by answer_id for robust lookup
        eval_dict = {}
        for e in evaluations:
            aid = e.get("answer_id", "")
            eval_dict[aid] = float(e.get("total_score", 0.0))

        scores = []
        for i in range(num_answers):
            aid = f"a{i + 1}"
            score = eval_dict.get(aid, 0.0)
            score = max(0.0, min(1.0, score))
            scores.append(score)

        return scores
