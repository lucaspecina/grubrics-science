"""
Test script for debugging evolve_rubrics_for_example function.

This script reads the question from evolving_rubrics/example_question.txt
and optionally loads golden answer from a dataset CSV file.
Can be used with VS Code debugger to test the evolving rubrics functionality.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from evolving_rubrics import evolve_rubrics_for_example


async def main():
    """Main test function."""
    # Read question from example_question.txt
    example_file = Path(__file__).parent / "evolving_rubrics" / "example_question.txt"
    
    if not example_file.exists():
        raise FileNotFoundError(f"Example question file not found: {example_file}")
    
    with open(example_file, "r", encoding="utf-8") as f:
        question = f.read().strip()
    
    print("Starting rubric evolution...")
    print(f"Question loaded from: {example_file}")
    print(f"\nQuestion:\n{question[:200]}...\n")
    print("-" * 70)
    
    # Optional: Load golden answer from file or create manually
    golden_answer = None
    golden_answer_file = Path(__file__).parent / "evolving_rubrics" / "example_golden_answer.txt"
    
    if golden_answer_file.exists():
        with open(golden_answer_file, "r", encoding="utf-8") as f:
            golden_answer = f.read().strip()
        print(f"\nGolden Answer loaded from: {golden_answer_file}")
        print(f"Golden Answer preview: {golden_answer[:200]}...")
    else:
        print("\nNo golden answer file found. You can create 'evolving_rubrics/example_golden_answer.txt' with your golden answer text.")
    
    # Run evolution with 2 iterations for testing
    result = await evolve_rubrics_for_example(
        question=question,
        num_iterations=2,  # Reduced for faster testing
        num_responses_per_iteration=2,  # Reduced for faster testing
        golden_answer=golden_answer  # Optional: use golden answer to guide initial rubrics
    )
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nInitial rubrics: {len(result['initial_ground_truth']['rubrics'])}")
    print(f"Final rubrics: {len(result['final_ground_truth']['rubrics'])}")
    print(f"Total iterations: {len(result['iterations'])}")
    
    if result['iterations']:
        last_iteration = result['iterations'][-1]
        print(f"\nLast iteration evaluations:")
        for eval_result in last_iteration['evaluations']:
            print(f"  Response {eval_result['response_num']}: {eval_result['total_reward']:.3f}")


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())

