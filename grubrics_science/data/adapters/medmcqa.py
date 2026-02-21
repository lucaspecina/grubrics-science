"""MedMCQA dataset adapter (medical MCQ, verifiable).

Source: ``openlifescienceai/medmcqa`` on HuggingFace.
Domain: medical multiple-choice questions across 21 specialties.
Size: ~183K train / ~6K validation / ~4K test.

Similar to MedQA but much larger and covering more medical specialties.
Gold_scores are programmatic (correct=1.0, incorrect=0.0).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter

_COP_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}
_COP_TO_KEY = {1: "opa", 2: "opb", 3: "opc", 4: "opd"}


class MedMCQAAdapter(DatasetAdapter):
    """Adapter for MedMCQA dataset.

    Source: ``openlifescienceai/medmcqa`` on HuggingFace.
    Domain: medical MCQ across 21 specialties (verifiable).
    """

    data_source = "medmcqa"
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
        """Load MedMCQA from HuggingFace.

        Args:
            path: HuggingFace dataset name override.
        """
        from datasets import load_dataset

        ds_name = path or "openlifescienceai/medmcqa"
        ds = load_dataset(ds_name, split="train")

        items = []
        for idx, row in enumerate(ds):
            question = row.get("question", "")
            cop = row.get("cop", 0)  # correct option: 1=A, 2=B, 3=C, 4=D

            options = {
                "A": row.get("opa", ""),
                "B": row.get("opb", ""),
                "C": row.get("opc", ""),
                "D": row.get("opd", ""),
            }

            answer_letter = _COP_TO_LETTER.get(cop, "")
            correct_key = _COP_TO_KEY.get(cop, "")
            correct_text = row.get(correct_key, "") if correct_key else ""

            subject = row.get("subject_name", "")
            topic = row.get("topic_name", "")

            items.append({
                "question_id": f"medmcqa_{idx}",
                "question": question,
                "options": options,
                "answer_letter": answer_letter,
                "correct_text": correct_text,
                "final_answer": correct_text,
                "subject": subject,
                "topic": topic,
            })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a MedMCQA item to veRL row format."""
        question = item["question"]
        question_id = item["question_id"]
        subject = item.get("subject", "")

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

        context = "This is a medical multiple-choice question."
        if subject:
            context = f"This is a medical question in {subject}."
        context += (
            " The rubric should evaluate medical knowledge accuracy, "
            "clinical reasoning, and completeness of the explanation."
        )

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
                "subject": item.get("subject", ""),
                "topic": item.get("topic", ""),
                "answers": answers,
                "gold_scores": gold_scores,
            },
        }
