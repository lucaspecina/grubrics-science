"""Adapter for the verifiable-math-problems.csv in the repo (olympiad math)."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DatasetAdapter


class VerifiableMathAdapter(DatasetAdapter):
    """Adapter for the olympiad math problems already in the repository.

    Source: ``data/primeintellect-synthetic-1/verifiable-math-problems.csv``
    Domain: olympiad-level math with gold-standard solutions.
    """

    data_source = "olympiad_math"
    domain_type = "verifiable"

    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load verifiable math problems from CSV.

        Args:
            path: Path to CSV file.  Falls back to the default repo location.
        """
        if path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            path = str(
                repo_root
                / "data"
                / "primeintellect-synthetic-1"
                / "verifiable-math-problems.csv"
            )

        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append({
                    "source": row.get("source", ""),
                    "task_type": row.get("task_type", ""),
                    "question": row.get("prompt", ""),
                    "solution": row.get("gold_standard_solution", ""),
                    "verification_info": row.get("verification_info", ""),
                })

        return items

    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert an olympiad math item to veRL row format."""
        question = item["question"]
        source = item.get("source", "olympiad")

        prompt_messages = self.build_rubric_generation_prompt(
            question=question,
            context=f"This is an olympiad-level math problem (source: {source}).",
        )

        return {
            "data_source": self.data_source,
            "prompt": prompt_messages,
            "reward_model": {
                "ground_truth": item.get("solution", ""),
                "style": "rubric_for_verifiable",
            },
            "extra_info": {
                "domain_type": self.domain_type,
                "source": source,
                "task_type": item.get("task_type", ""),
                "solution": item.get("solution", ""),
                "verification_info": item.get("verification_info", ""),
                "question": question,
            },
        }
