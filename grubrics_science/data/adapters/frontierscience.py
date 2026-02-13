"""FrontierScience Research dataset adapter."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class FrontierScienceAdapter(DatasetAdapter):
    """Adapter for OpenAI's FrontierScience Research track.

    Source: ``data/frontierscience-research/test.jsonl`` in the repo.
    Domain: open-ended physics research questions with PhD-authored rubrics.
    Size: ~60 subtasks.
    """

    data_source = "frontierscience"
    domain_type = "open_rubric"

    def __init__(self, cache_path: Optional[str] = None):
        """
        Args:
            cache_path: Optional path to precompute_cache.jsonl with
                pre-generated answers and gold scores.
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
        """Load FrontierScience from JSONL file.

        Args:
            path: Path to test.jsonl.  Falls back to the default repo location.
        """
        if path is None:
            # Default repo location
            repo_root = Path(__file__).parent.parent.parent.parent
            path = str(repo_root / "data" / "frontierscience-research" / "test.jsonl")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                items.append({
                    "question_id": str(idx),
                    "problem": record["problem"],
                    "golden_rubric": record["answer"],
                    "subject": record.get("subject", "physics"),
                    "task_group_id": record.get("task_group_id"),
                    "metadata": {
                        k: v
                        for k, v in record.items()
                        if k not in ("problem", "answer", "subject", "task_group_id")
                    },
                })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a FrontierScience item to veRL row format.

        If a precompute cache is available, this attaches the pre-generated
        answers and gold scores into ``extra_info``.
        """
        question = item["problem"]
        golden_rubric = item["golden_rubric"]
        question_id = item["question_id"]

        # Try to load cached answers and gold scores
        cache = self._load_cache()
        cached = cache.get(question_id, {})

        answers = cached.get("answers", [])
        gold_scores = cached.get("gold_scores", [])

        # Build contrastive excerpts if answers are available
        best_excerpt = None
        worst_excerpt = None
        if answers and gold_scores:
            import numpy as np

            scores_arr = np.array(gold_scores)
            best_idx = int(scores_arr.argmax())
            worst_idx = int(scores_arr.argmin())
            # Take first 500 chars as excerpt
            best_excerpt = answers[best_idx][:500]
            worst_excerpt = answers[worst_idx][:500]

        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context=(
                f"This is an open-ended {item.get('subject', 'physics')} research question. "
                "The rubric should evaluate scientific reasoning, methodology, "
                "derivations, and correctness of conclusions."
            ),
            best_answer_excerpt=best_excerpt,
            worst_answer_excerpt=worst_excerpt,
        )

        return {
            "data_source": self.data_source,
            "prompt": prompt_messages,
            "reward_model": {
                "ground_truth": "",  # open-ended, no single answer
                "style": "rubric_for_open",
            },
            "extra_info": {
                "domain_type": self.domain_type,
                "question_id": question_id,
                "question": question,
                "golden_rubric": golden_rubric,
                "subject": item.get("subject", "physics"),
                "task_group_id": item.get("task_group_id"),
                "answers": answers,
                "gold_scores": gold_scores,
                "best_answer_excerpt": best_excerpt or "",
                "worst_answer_excerpt": worst_excerpt or "",
            },
        }
