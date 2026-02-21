"""HealthBench dataset adapter (OpenAI, 5000 medical conversations).

Source: ``openai/healthbench`` on HuggingFace (MIT license).
Domain: open-ended medical conversations with physician-authored rubrics.
Size: ~5000 conversations, 48,562 unique rubric criteria from 262 physicians.

The meta_eval file contains model responses already evaluated by physicians
(binary_labels). We use those responses as pre-generated answers (saving
Answer Policy cost), but gold_scores must come from our own Judge to avoid
evaluator mismatch.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


def _rubrics_to_text(rubrics: List[Dict[str, Any]]) -> str:
    """Convert HealthBench rubric JSON list to our text format.

    HealthBench format: [{"criterion": "...", "points": 10, "tags": [...]}]
    Our format: "Points: 10, Item: ..."
    """
    lines = []
    for r in rubrics:
        pts = r.get("points", 0)
        criterion = r.get("criterion", "")
        if criterion:
            lines.append(f"Points: {pts}, Item: {criterion}")
    return "\n".join(lines)


def _extract_question_text(prompt: List[Dict[str, str]]) -> str:
    """Extract a flat question string from HealthBench multi-turn prompt.

    HealthBench prompts are lists of {role, content} messages.
    We concatenate them into a readable conversation for the rubric generator.
    """
    parts = []
    for msg in prompt:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


class HealthBenchAdapter(DatasetAdapter):
    """Adapter for OpenAI's HealthBench dataset.

    Source: ``openai/healthbench`` on HuggingFace.
    Domain: open-ended medical conversations with physician-authored rubrics.
    Size: ~5000 conversations.

    Args:
        cache_path: Path to precompute cache JSONL (gold_scores from our Judge).
        meta_eval_path: Path to oss_meta_eval.jsonl (for pre-generated answers).
        dataset_path: Path to oss_eval.jsonl (main dataset). If None, downloads
            from HuggingFace.
    """

    data_source = "healthbench"
    domain_type = "open_rubric"

    def __init__(
        self,
        cache_path: Optional[str] = None,
        meta_eval_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ):
        self._cache_path = cache_path
        self._meta_eval_path = meta_eval_path
        self._dataset_path = dataset_path
        self._cache: Optional[Dict[str, Any]] = None
        self._meta_eval: Optional[Dict[str, Any]] = None

    def _load_cache(self) -> Dict[str, Any]:
        """Load precompute cache (gold_scores from our Judge)."""
        if self._cache is not None:
            return self._cache

        self._cache = {}
        if self._cache_path and Path(self._cache_path).exists():
            with open(self._cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        pid = entry.get("prompt_id", "")
                        if pid:
                            self._cache[pid] = entry
        return self._cache

    def _load_meta_eval(self) -> Dict[str, Any]:
        """Load meta_eval answers (model responses pre-evaluated by physicians).

        Returns dict keyed by prompt_id with lists of answer texts.
        """
        if self._meta_eval is not None:
            return self._meta_eval

        self._meta_eval = {}
        if not self._meta_eval_path or not Path(self._meta_eval_path).exists():
            return self._meta_eval

        with open(self._meta_eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                pid = entry.get("prompt_id", "")
                if not pid:
                    continue

                if pid not in self._meta_eval:
                    self._meta_eval[pid] = []

                completion = entry.get("completion", "")
                if completion:
                    self._meta_eval[pid].append(completion)

        return self._meta_eval

    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load HealthBench from local JSONL or HuggingFace.

        Args:
            path: Path to oss_eval.jsonl. If None, uses dataset_path from
                constructor, or downloads from HuggingFace.
        """
        data_path = path or self._dataset_path

        if data_path and Path(data_path).exists():
            return self._load_from_jsonl(data_path)

        return self._load_from_huggingface()

    def _load_from_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load from a local JSONL file."""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                items.append(self._parse_record(record))
        return items

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Download and load from HuggingFace."""
        try:
            import blobfile as bf
        except ImportError:
            from datasets import load_dataset
            ds = load_dataset("openai/healthbench", split="train")
            items = []
            for row in ds:
                items.append(self._parse_record(dict(row)))
            return items

        url = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
        items = []
        with bf.BlobFile(url, "rb") as f:
            for line in f:
                record = json.loads(line)
                items.append(self._parse_record(record))
        return items

    def _parse_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single HealthBench record into our internal format."""
        prompt = record.get("prompt", [])
        rubrics = record.get("rubrics", [])
        prompt_id = record.get("prompt_id", "")

        # Extract ideal completions if available
        ideal_data = record.get("ideal_completions_data") or {}
        ideal_completion = ideal_data.get("ideal_completion", "")
        ref_completions = ideal_data.get(
            "ideal_completions_ref_completions", []
        ) or []

        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "rubrics": rubrics,
            "golden_rubric": _rubrics_to_text(rubrics),
            "question": _extract_question_text(prompt),
            "category": record.get("category", ""),
            "example_tags": record.get("example_tags", []),
            "ideal_completion": ideal_completion,
            "ref_completions": ref_completions,
        }

    def _get_answers_for_question(self, prompt_id: str, item: Dict[str, Any]) -> List[str]:
        """Get pre-generated answers for a question.

        Priority: cache > meta_eval > ideal_completions from oss_eval.
        """
        cache = self._load_cache()
        cached = cache.get(prompt_id, {})
        if cached.get("answers"):
            return cached["answers"]

        meta = self._load_meta_eval()
        meta_answers = meta.get(prompt_id, [])
        if meta_answers:
            return meta_answers

        answers = []
        if item.get("ideal_completion"):
            answers.append(item["ideal_completion"])
        answers.extend(item.get("ref_completions", []))
        return answers

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a HealthBench item to veRL row format."""
        question = item["question"]
        golden_rubric = item["golden_rubric"]
        prompt_id = item["prompt_id"]

        cache = self._load_cache()
        cached = cache.get(prompt_id, {})

        answers = self._get_answers_for_question(prompt_id, item)
        gold_scores = cached.get("gold_scores", [])

        best_excerpt = None
        worst_excerpt = None
        if answers and gold_scores and len(gold_scores) == len(answers):
            from . import use_contrastive
            if use_contrastive():
                import numpy as np
                scores_arr = np.array(gold_scores)
                best_idx = int(scores_arr.argmax())
                worst_idx = int(scores_arr.argmin())
                if scores_arr[best_idx] != scores_arr[worst_idx]:
                    best_excerpt = answers[best_idx][:500]
                    worst_excerpt = answers[worst_idx][:500]

        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context=(
                "This is a medical conversation between a patient and an AI assistant. "
                "The rubric should evaluate medical accuracy, completeness, safety, "
                "communication quality, and instruction following."
            ),
            best_answer_excerpt=best_excerpt,
            worst_answer_excerpt=worst_excerpt,
        )

        return {
            "data_source": self.data_source,
            "prompt": prompt_messages,
            "reward_model": {
                "ground_truth": "",
                "style": "rubric_for_open",
            },
            "extra_info": {
                "domain_type": self.domain_type,
                "prompt_id": prompt_id,
                "question": question,
                "golden_rubric": golden_rubric,
                "rubrics_json": item.get("rubrics", []),
                "category": item.get("category", ""),
                "example_tags": item.get("example_tags", []),
                "answers": answers,
                "gold_scores": gold_scores,
                "best_answer_excerpt": best_excerpt or "",
                "worst_answer_excerpt": worst_excerpt or "",
            },
        }
