"""End-to-end test: functional alignment reward for verifiable domains.

Loads precomputed GSM8K cache, sends good/bad/degenerate rubrics through
the full reward pipeline (Judge API → Spearman → reward), and verifies
that good rubric > bad rubric > degenerate rubric.

Usage:
    python scripts/test_verifiable_reward_e2e.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grubrics_science.rewards.grubrics_reward import compute_score


def load_cache(path: str):
    """Load precompute cache entries."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def main():
    cache_path = "data/cache/gsm8k_precompute_test.jsonl"
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found at {cache_path}")
        print("Run: python -m grubrics_science.data.precompute_verifiable --dataset gsm8k --limit 5")
        sys.exit(1)

    entries = load_cache(cache_path)
    print(f"Loaded {len(entries)} cached questions")
    print(f"Judge model: {os.environ.get('JUDGE_MODEL', 'gpt-4o-mini')}")
    print("=" * 70)

    # Test rubrics
    rubrics = {
        "good": (
            "Points: 3.0, Item: The answer correctly identifies the mathematical operation needed\n"
            "Points: 3.0, Item: The answer shows clear step-by-step arithmetic work\n"
            "Points: 2.0, Item: The answer arrives at the correct final numerical result\n"
            "Points: 2.0, Item: The answer is clearly stated and well-organized"
        ),
        "bad": (
            "Points: 5.0, Item: The answer is written in English\n"
            "Points: 5.0, Item: The answer mentions numbers"
        ),
        "degenerate": (
            "Points: 2.0, Item: The response exists\n"
            "Points: 2.0, Item: The response contains text\n"
            "Points: 2.0, Item: The response is not empty\n"
            "Points: 2.0, Item: The response has characters\n"
            "Points: 2.0, Item: The response was generated"
        ),
    }

    # Test on first 2 questions
    for qi, entry in enumerate(entries[:2]):
        question = entry["question"]
        answers = entry["answers"]
        gold_scores = entry["gold_scores"]

        print(f"\nQ{qi}: {question[:80]}...")
        print(f"  Gold scores: {gold_scores}")
        print(f"  Answers: {len(answers)} (lengths: {[len(a) for a in answers]})")
        print("-" * 70)

        extra_info = {
            "question": question,
            "answers": answers,
            "gold_scores": gold_scores,
            "domain_type": "verifiable",
        }

        scores = {}
        for rubric_name, rubric_text in rubrics.items():
            score = compute_score(
                data_source="gsm8k",
                solution_str=rubric_text,
                ground_truth=entry.get("gold_answer", ""),
                extra_info=extra_info,
            )
            scores[rubric_name] = score
            print(f"  [{rubric_name:12s}] reward = {score:.4f}")

        # Check ordering
        print()
        if scores["good"] > scores["bad"]:
            print(f"  ✓ good ({scores['good']:.3f}) > bad ({scores['bad']:.3f})")
        else:
            print(f"  ✗ UNEXPECTED: good ({scores['good']:.3f}) <= bad ({scores['bad']:.3f})")

        if scores["good"] > scores["degenerate"]:
            print(f"  ✓ good ({scores['good']:.3f}) > degenerate ({scores['degenerate']:.3f})")
        else:
            print(f"  ✗ UNEXPECTED: good ({scores['good']:.3f}) <= degenerate ({scores['degenerate']:.3f})")

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("If good > bad > degenerate, functional alignment works for verifiable domains.")
    print("The Judge can distinguish rubric quality even with programmatic gold_scores.")


if __name__ == "__main__":
    main()
