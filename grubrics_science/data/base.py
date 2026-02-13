"""Base dataset adapter for converting any dataset to veRL parquet format."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class DatasetAdapter(ABC):
    """Abstract base adapter: any dataset -> veRL parquet format.

    To add a new dataset, subclass this and implement ``load_raw`` and
    ``to_verl_format``.  Register the adapter in ``adapters/__init__.py``.
    """

    # --- subclasses MUST set these ---
    data_source: str = ""
    domain_type: str = ""  # "verifiable" | "open_rubric" | "open_no_rubric"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load raw items from the original source.

        Args:
            path: Optional path override (file or HuggingFace dataset name).

        Returns:
            List of dicts, one per example.
        """
        ...

    @abstractmethod
    def to_verl_format(self, item: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
        """Convert a single raw item to the veRL parquet schema.

        Must return a dict with **at least** these keys::

            {
                "data_source": str,
                "prompt": List[Dict],  # chat messages for the tokenizer
                "reward_model": {
                    "ground_truth": str,   # for verifiable; "" otherwise
                    "style": str,
                },
                "extra_info": {
                    "domain_type": str,
                    ...  # adapter-specific metadata
                },
            }

        Args:
            item: One dict from ``load_raw()``.
            tokenizer: Optional HuggingFace tokenizer (used to apply chat
                template if needed).

        Returns:
            Dict in veRL row format.
        """
        ...

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_rubric_generation_prompt(
        question: str,
        context: Optional[str] = None,
        best_answer_excerpt: Optional[str] = None,
        worst_answer_excerpt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build a chat-format prompt asking the model to generate a rubric.

        Returns a list of message dicts (system + user) suitable for
        ``tokenizer.apply_chat_template``.
        """
        system_msg = (
            "You are a RUBRIC WRITER. Given a QUESTION, produce a scoring rubric "
            "in the format: Points: <number>, Item: <text>. "
            "The sum of all Points must be exactly 10.0. "
            "Each item must be actionable, weighted by importance, and discriminative."
        )

        user_parts = []

        if context:
            user_parts.append(f"CONTEXT:\n{context}\n")

        if best_answer_excerpt or worst_answer_excerpt:
            user_parts.append(
                "Below are answer excerpts to help you create discriminative criteria:"
            )
            if best_answer_excerpt:
                user_parts.append(
                    f"High-quality answer excerpt:\n{best_answer_excerpt}\n"
                )
            if worst_answer_excerpt:
                user_parts.append(
                    f"Low-quality answer excerpt:\n{worst_answer_excerpt}\n"
                )

        user_parts.append(f"QUESTION:\n{question}\n")
        user_parts.append("YOUR RUBRIC:")

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    # ------------------------------------------------------------------
    # Parquet serialisation
    # ------------------------------------------------------------------
    def to_parquet(
        self,
        output_dir: str,
        tokenizer: Any = None,
        path: Optional[str] = None,
        max_items: Optional[int] = None,
        split: str = "train",
    ) -> Path:
        """Load raw data, convert every item, and write a parquet file.

        Args:
            output_dir: Directory where the parquet will be written.
            tokenizer: HuggingFace tokenizer for chat template application.
            path: Optional path override passed to ``load_raw()``.
            max_items: If set, truncate the dataset to this many items.
            split: Name tag appended to the output filename.

        Returns:
            Path to the written parquet file.
        """
        items = self.load_raw(path)
        if max_items is not None:
            items = items[:max_items]

        rows: List[Dict[str, Any]] = []
        for item in items:
            row = self.to_verl_format(item, tokenizer=tokenizer)
            # Serialise nested dicts/lists to JSON strings for parquet
            for key in ("prompt", "reward_model", "extra_info"):
                if key in row and not isinstance(row[key], str):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
            rows.append(row)

        df = pd.DataFrame(rows)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        parquet_file = out_path / f"{self.data_source}_{split}.parquet"
        df.to_parquet(parquet_file, index=False)

        print(f"[{self.data_source}] Wrote {len(df)} rows -> {parquet_file}")
        return parquet_file
