"""IO utilities for cache management."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load cache from JSONL file."""
    cache = {}
    if not cache_path.exists():
        return cache
    
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                q_id = record['question_id']
                cache[q_id] = record
    return cache


def save_cache(cache: Dict[str, Any], cache_path: Path):
    """Save cache to JSONL file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        for q_id, record in cache.items():
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_cache_entry(
    question_id: str,
    question: str,
    answers: List[str],
    gold_scores: List[float],
    cache_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    gold_details: Optional[List[Dict[str, Any]]] = None
):
    """
    Save a single cache entry.
    
    Args:
        question_id: Unique identifier for the question
        question: The question text
        answers: List of generated answers
        gold_scores: List of gold scores (one per answer)
        cache_path: Path to cache file
        metadata: Optional metadata dict
        gold_details: Optional list of detailed evaluations (one per answer) with item-by-item breakdowns
    """
    entry = {
        'question_id': question_id,
        'question': question,
        'answers': answers,
        'gold_scores': gold_scores,
    }
    if metadata:
        entry['metadata'] = metadata
    if gold_details is not None:
        entry['gold_details'] = gold_details
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

