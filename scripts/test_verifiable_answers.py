"""Test: can we generate a mix of correct/incorrect answers for GSM8K?

Generates 4 answers per question using different instruction types,
then checks correctness programmatically.

Usage:
    python scripts/test_verifiable_answers.py
"""

import asyncio
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grubrics_science.llm.client import AzureOpenAIClient


# ---------------------------------------------------------------------------
# 3 GSM8K questions (hand-picked for simplicity)
# ---------------------------------------------------------------------------

QUESTIONS = [
    # MATH Level 2 (should be easy for GPT-5.2)
    {
        "question": "What is the remainder when 2007 is divided by 25?",
        "gold_answer": "7",
        "level": "L2",
    },
    # MATH Level 3 (medium - might get some wrong)
    {
        "question": "How many three-digit numbers are multiples of neither 5 nor 7?",
        "gold_answer": "617",
        "level": "L3",
    },
    {
        "question": "What is the greatest common divisor of 1407 and 903?",
        "gold_answer": "21",
        "level": "L3",
    },
    # MATH Level 4 (harder - more errors expected)
    {
        "question": "How many integers between 1 and 200 are multiples of both 3 and 5 but not of either 4 or 7?",
        "gold_answer": "9",
        "level": "L4",
    },
    {
        "question": "The number 236! is divisible by 12^n. What is the greatest integer value of n?",
        "gold_answer": "116",
        "level": "L4",
    },
    # MATH Level 5 (competition - GPT likely fails)
    {
        "question": "What is the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square?",
        "gold_answer": "80",
        "level": "L5",
    },
    # GSM8K for reference
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
        "gold_answer": "18",
        "level": "GSM8K",
    },
]


# ---------------------------------------------------------------------------
# Instruction types for answer diversity
# ---------------------------------------------------------------------------

ANSWER_INSTRUCTIONS = {
    "rigorous": (
        "Solve this math problem step by step. Show all your work clearly. "
        "End your answer with the final number on a line by itself after '#### '."
    ),
    "shallow": (
        "Solve this math problem. Give a quick answer without too much detail. "
        "You may skip some steps. End with '#### ' followed by the final number."
    ),
    "overconfident": (
        "Solve this math problem. Be confident and direct. Don't second-guess yourself. "
        "If you're unsure about a step, just pick the most likely interpretation and go with it. "
        "End with '#### ' followed by the final number."
    ),
    "careless": (
        "Solve this math problem quickly. Don't worry too much about double-checking. "
        "Just do the arithmetic in your head if you can. "
        "End with '#### ' followed by the final number."
    ),
}


def extract_final_answer(response: str) -> str:
    """Extract the number after #### from the response."""
    # Look for #### pattern
    match = re.search(r'####\s*([\d,.\-]+)', response)
    if match:
        # Clean up: remove commas, strip
        return match.group(1).replace(",", "").strip().rstrip(".")
    return ""


def normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def check_correct(response: str, gold: str) -> bool:
    """Check if the response contains the correct answer."""
    extracted = extract_final_answer(response)
    if not extracted:
        return False
    return normalize_number(extracted) == normalize_number(gold)


async def generate_answer(client, question: str, instruction_type: str) -> str:
    """Generate one answer for a question."""
    instruction = ANSWER_INSTRUCTIONS[instruction_type]
    prompt = f"{instruction}\n\nQuestion: {question}\n\nAnswer:"

    response = await client.generate(
        prompt=prompt,
        max_tokens=1024,
    )
    return response.strip()


async def main():
    model = os.environ.get("JUDGE_MODEL", "gpt-5.2-chat")
    print(f"Model: {model}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Instruction types: {list(ANSWER_INSTRUCTIONS.keys())}")
    print("=" * 70)

    client = AzureOpenAIClient(model=model, use_azure=True)

    for qi, q in enumerate(QUESTIONS):
        print(f"\n{'='*70}")
        print(f"Q{qi} [{q.get('level', '?')}]: {q['question'][:80]}...")
        print(f"Gold answer: {q['gold_answer']}")
        print("-" * 70)

        correct_count = 0
        incorrect_count = 0

        for inst_type in ANSWER_INSTRUCTIONS:
            response = await generate_answer(client, q["question"], inst_type)
            extracted = extract_final_answer(response)
            is_correct = check_correct(response, q["gold_answer"])

            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

            status = "CORRECT" if is_correct else "WRONG"
            print(f"  [{inst_type:15s}] extracted={extracted:>10s}  gold={q['gold_answer']:>10s}  -> {status}")

            # Show first 150 chars of response for debugging
            preview = response[:150].replace("\n", " ")
            print(f"    preview: {preview}...")

        print(f"\n  Summary: {correct_count} correct, {incorrect_count} incorrect out of 4")

        if incorrect_count == 0:
            print("  WARNING: All answers correct! GPT is too good at this question.")
            print("  May need perturbation strategy for this difficulty level.")
        elif correct_count == 0:
            print("  WARNING: All answers incorrect! Question may be too hard or instructions confusing.")

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("If most questions have a mix of correct/incorrect, the LLM-based")
    print("strategy works. If all are correct, we need perturbations.")


if __name__ == "__main__":
    asyncio.run(main())
