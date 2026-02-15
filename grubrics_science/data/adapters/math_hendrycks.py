"""MATH dataset adapter (Hendrycks competition math)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class MATHAdapter(DatasetAdapter):
    """Adapter for Hendrycks' MATH dataset.

    Source: ``hendrycks/competition_math`` on HuggingFace.
    Domain: competition math (AMC 10/12, AIME, etc.), 7 subjects, 5 levels.
    Size: ~12K train / 500 test.
    """

    data_source = "math"
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
        """Load MATH from HuggingFace datasets."""
        from datasets import load_dataset

        ds_name = path or "hendrycks/competition_math"
        ds = load_dataset(ds_name, split="train")

        items = []
        for row in ds:
            # MATH format: "problem", "solution", "level", "type"
            # Solution contains step-by-step reasoning with \\boxed{answer}.
            solution = row["solution"]
            final_answer = ""
            if "\\boxed{" in solution:
                # Extract content inside \boxed{...}
                start = solution.rfind("\\boxed{") + len("\\boxed{")
                depth = 1
                end = start
                while end < len(solution) and depth > 0:
                    if solution[end] == "{":
                        depth += 1
                    elif solution[end] == "}":
                        depth -= 1
                    end += 1
                final_answer = solution[start:end - 1]

            items.append({
                "question": row["problem"],
                "solution": solution,
                "final_answer": final_answer,
                "level": row.get("level", ""),
                "subject": row.get("type", ""),
            })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a MATH item to veRL row format.

        If a precompute cache is available, attaches answers + gold_scores
        into extra_info for functional alignment reward.
        """
        question = item["question"]
        subject = item.get("subject", "mathematics")
        level = item.get("level", "")

        # Try to load cached answers and gold scores
        cache = self._load_cache()
        cached = {}
        for entry in cache.values():
            if entry.get("question") == question:
                cached = entry
                break

        answers = cached.get("answers", [])
        gold_scores = cached.get("gold_scores", [])

        # Build contrastive excerpts if answers are available
        best_excerpt = None
        worst_excerpt = None
        if answers and gold_scores:
            best_idx = gold_scores.index(max(gold_scores))
            worst_idx = gold_scores.index(min(gold_scores))
            if gold_scores[best_idx] != gold_scores[worst_idx]:
                best_excerpt = answers[best_idx][:500]
                worst_excerpt = answers[worst_idx][:500]

        context = f"This is a competition math problem ({subject}"
        if level:
            context += f", {level}"
        context += ") with a precise answer."

        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context=context,
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
                "level": item.get("level", ""),
                "subject": item.get("subject", ""),
                "question": question,
                "answers": answers,
                "gold_scores": gold_scores,
            },
        }
