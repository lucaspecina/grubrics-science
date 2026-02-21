"""GSM8K dataset adapter (Grade School Math 8K)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class GSM8KAdapter(DatasetAdapter):
    """Adapter for OpenAI's GSM8K dataset.

    Source: ``openai/gsm8k`` on HuggingFace.
    Domain: grade-school math with step-by-step solutions.
    Size: ~7.5K train / ~1.3K test.
    """

    data_source = "gsm8k"
    domain_type = "verifiable"

    def __init__(self, cache_path: Optional[str] = None):
        """
        Args:
            cache_path: Optional path to precompute cache JSONL with
                pre-generated answers and programmatic gold scores.
        """
        self._cache_path = cache_path
        self._cache: Optional[Dict[str, Any]] = None

    def _load_cache(self) -> Dict[str, Any]:
        """Load the precompute cache (answers + gold scores per question)."""
        if self._cache is not None:
            return self._cache

        self._cache = {}
        if self._cache_path and Path(self._cache_path).exists():
            with open(self._cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        qid = entry.get("question_id", "")
                        if qid:
                            self._cache[qid] = entry
        return self._cache

    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load GSM8K from HuggingFace datasets."""
        from datasets import load_dataset

        ds_name = path or "openai/gsm8k"
        ds = load_dataset(ds_name, "main", split="train")

        items = []
        for row in ds:
            # GSM8K format: "question" and "answer" fields.
            # The answer contains step-by-step reasoning ending with
            # "#### <final_number>".
            answer_text = row["answer"]
            final_answer = ""
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()

            items.append({
                "question": row["question"],
                "solution": answer_text,
                "final_answer": final_answer,
            })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a GSM8K item to veRL row format.

        If a precompute cache is available, attaches answers + gold_scores
        into extra_info for functional alignment reward.
        """
        question = item["question"]

        # Try to load cached answers and gold scores
        # Cache key format: gsm8k_{index} — matches precompute_verifiable.py
        cache = self._load_cache()
        # Find by question text match (more robust than index)
        cached = {}
        for entry in cache.values():
            if entry.get("question") == question:
                cached = entry
                break

        answers = cached.get("answers", [])
        gold_scores = cached.get("gold_scores", [])

        # Build contrastive excerpts if answers are available and enabled
        best_excerpt = None
        worst_excerpt = None
        if answers and gold_scores:
            from . import use_contrastive
            if use_contrastive():
                best_idx = gold_scores.index(max(gold_scores))
                worst_idx = gold_scores.index(min(gold_scores))
                if gold_scores[best_idx] != gold_scores[worst_idx]:
                    best_excerpt = answers[best_idx][:500]
                    worst_excerpt = answers[worst_idx][:500]

        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context="This is a grade-school math problem with a unique numerical answer.",
            best_answer_excerpt=best_excerpt,
            worst_answer_excerpt=worst_excerpt,
        )

        return {
            "data_source": self.data_source,
            "prompt": prompt_messages,
            "reward_model": {
                "ground_truth": item["final_answer"],
                "style": "rubric_for_verifiable",
            },
            "extra_info": {
                "domain_type": self.domain_type,
                "solution": item["solution"],
                "final_answer": item["final_answer"],
                "question": question,
                "answers": answers,
                "gold_scores": gold_scores,
            },
        }
