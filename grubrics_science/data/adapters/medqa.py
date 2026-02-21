"""MedQA-USMLE dataset adapter (medical MCQ, verifiable).

Source: ``GBaker/MedQA-USMLE-4-options`` on HuggingFace.
Domain: USMLE-style medical multiple-choice questions.
Size: ~10K train / ~1.3K test.

For the curriculum, MedQA provides verifiable medical questions where
gold_scores are programmatic (correct=1.0, incorrect=0.0). The 4 MCQ
options serve as natural answer diversity (1 correct + 3 incorrect).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class MedQAAdapter(DatasetAdapter):
    """Adapter for MedQA-USMLE 4-option dataset.

    Source: ``GBaker/MedQA-USMLE-4-options`` on HuggingFace.
    Domain: USMLE medical exam questions (verifiable, MCQ).
    """

    data_source = "medqa"
    domain_type = "verifiable"

    def __init__(self, cache_path: Optional[str] = None):
        self._cache_path = cache_path
        self._cache: Optional[Dict[str, Any]] = None

    def _load_cache(self) -> Dict[str, Any]:
        """Load precompute cache (answers + gold scores)."""
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
        """Load MedQA from HuggingFace.

        Args:
            path: HuggingFace dataset name override.
        """
        from datasets import load_dataset

        ds_name = path or "GBaker/MedQA-USMLE-4-options"
        ds = load_dataset(ds_name, split="train")

        items = []
        for idx, row in enumerate(ds):
            question = row["question"]
            options = row.get("options", {})

            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except (json.JSONDecodeError, TypeError):
                    options = {}

            answer_letter = row.get("answer_idx", "") or row.get("answer", "")
            if len(answer_letter) > 1:
                answer_letter = row.get("answer_idx", "")
            correct_text = options.get(answer_letter, "")

            items.append({
                "question_id": f"medqa_{idx}",
                "question": question,
                "options": options,
                "answer_letter": answer_letter,
                "correct_text": correct_text,
                "final_answer": correct_text,
            })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a MedQA item to veRL row format.

        The prompt asks GRubrics to generate a rubric for the medical question
        (without showing the MCQ options, so the rubric is general).
        """
        question = item["question"]
        question_id = item["question_id"]

        cache = self._load_cache()
        cached = {}
        for entry in cache.values():
            if entry.get("question_id") == question_id:
                cached = entry
                break

        answers = cached.get("answers", [])
        gold_scores = cached.get("gold_scores", [])

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
            context=(
                "This is a USMLE-style medical exam question. "
                "The rubric should evaluate medical knowledge accuracy, "
                "clinical reasoning, and completeness of the explanation."
            ),
            best_answer_excerpt=best_excerpt,
            worst_answer_excerpt=worst_excerpt,
        )

        return {
            "data_source": self.data_source,
            "prompt": prompt_messages,
            "reward_model": {
                "ground_truth": item.get("correct_text", ""),
                "style": "rubric_for_verifiable",
            },
            "extra_info": {
                "domain_type": self.domain_type,
                "question_id": question_id,
                "question": question,
                "options": item.get("options", {}),
                "answer_letter": item.get("answer_letter", ""),
                "correct_text": item.get("correct_text", ""),
                "final_answer": item.get("final_answer", ""),
                "answers": answers,
                "gold_scores": gold_scores,
            },
        }
