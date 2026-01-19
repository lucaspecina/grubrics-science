"""FrontierScience dataset loader and task definition."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from ..utils.io import load_cache, save_cache_entry


@dataclass
class TaskExample:
    """Single example from the dataset."""
    question_id: str
    problem: str
    golden_rubric: str
    subject: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FrontierScienceTask:
    """Task wrapper for FrontierScience dataset."""
    
    def __init__(self, dataset_path: str, cache_dir: str = "grubrics_science/data/cache"):
        """
        Initialize task.
        
        Args:
            dataset_path: Path to test.jsonl file (relative to repo root)
            cache_dir: Directory for caching precomputed answers
        """
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / "precompute_cache.jsonl"
        
        # Load dataset
        self.examples = self._load_dataset()
    
    def _load_dataset(self) -> List[TaskExample]:
        """Load dataset from JSONL file."""
        examples = []
        
        # Handle both absolute and relative paths
        if not Path(self.dataset_path).is_absolute():
            repo_root = Path(__file__).parent.parent.parent
            full_path = repo_root / self.dataset_path
            if full_path.exists():
                self.dataset_path = full_path
            elif not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path} or {full_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    record = json.loads(line)
                    example = TaskExample(
                        question_id=str(idx),
                        problem=record['problem'],
                        golden_rubric=record['answer'],
                        subject=record.get('subject'),
                        metadata={
                            'task_group_id': record.get('task_group_id'),
                            **{k: v for k, v in record.items() if k not in ['problem', 'answer', 'subject', 'task_group_id']}
                        }
                    )
                    examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> TaskExample:
        return self.examples[idx]
    
    def load_cache(self) -> Dict[str, Any]:
        """Load precomputed cache."""
        return load_cache(self.cache_path)
    
    def save_cache_entry(
        self,
        question_id: str,
        question: str,
        answers: List[str],
        gold_scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        gold_details: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Save a cache entry.
        
        Args:
            question_id: Unique identifier for the question
            question: The question text
            answers: List of generated answers
            gold_scores: List of gold scores (one per answer)
            metadata: Optional metadata dict
            gold_details: Optional list of detailed evaluations (one per answer) with item-by-item breakdowns
        """
        save_cache_entry(
            question_id=question_id,
            question=question,
            answers=answers,
            gold_scores=gold_scores,
            cache_path=self.cache_path,
            metadata=metadata,
            gold_details=gold_details
        )
    
    def get_cached_data(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data for a question."""
        cache = self.load_cache()
        return cache.get(question_id)

