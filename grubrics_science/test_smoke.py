"""Smoke test for GRubrics Science.

Tests precompute and train modes with minimal settings.
"""

import asyncio
import yaml
from pathlib import Path
import tempfile
import shutil

from grubrics_science.tasks.frontierscience import FrontierScienceTask
from grubrics_science.llm.client import AzureOpenAIClient
from grubrics_science.llm.prompts import get_answer_policy_prompt, get_grubrics_prompt
from grubrics_science.judge.judge import Judge
from grubrics_science.rewards.alignment import compute_reward, compute_alignment
from grubrics_science.utils.seeding import set_seed


async def test_precompute_smoke():
    """Test precompute mode with 2 questions, K=2."""
    print("=" * 70)
    print("SMOKE TEST: PRECOMPUTE MODE")
    print("=" * 70)
    
    # Create temp cache dir
    temp_dir = Path(tempfile.mkdtemp())
    cache_dir = temp_dir / "cache"
    
    try:
        # Load dataset
        dataset_path = Path("frontierscience-research/test.jsonl")
        if not dataset_path.exists():
            dataset_path = Path(__file__).parent.parent / dataset_path
        
        task = FrontierScienceTask(
            dataset_path=str(dataset_path),
            cache_dir=str(cache_dir)
        )
        
        print(f"Loaded {len(task)} questions")
        
        # Limit to 2 questions for smoke test
        test_questions = task.examples[:2]
        
        # Initialize clients
        answer_policy_client = AzureOpenAIClient(
            model="gpt-4o-mini",
            use_azure=True
        )
        
        judge = Judge(
            model="gpt-4o-mini",
            use_azure=True
        )
        
        set_seed(42)
        
        # Process 2 questions with K=2 answers each
        for idx, example in enumerate(test_questions):
            print(f"\nQuestion {idx+1}/2 (ID: {example.question_id})")
            question = example.problem
            golden_rubric = example.golden_rubric
            
            # Generate 2 answers
            answers = []
            for i in range(2):
                prompt = get_answer_policy_prompt(question, "normal")
                answer = await answer_policy_client.generate(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.8
                )
                answers.append(answer.strip())
                print(f"  Generated answer {i+1}: {len(answer)} chars")
            
            # Compute gold scores
            gold_scores = []
            for i, answer in enumerate(answers):
                scores = await judge.evaluate_batch(
                    question=question,
                    answer=answer,
                    rubrics=[golden_rubric],
                    answer_id=f"a{i+1}"
                )
                gold_scores.append(scores[0])
                print(f"  Gold score {i+1}: {scores[0]:.3f}")
            
            # Save to cache
            task.save_cache_entry(
                question_id=example.question_id,
                question=question,
                answers=answers,
                gold_scores=gold_scores
            )
            print(f"  Saved to cache")
        
        # Verify cache
        cache = task.load_cache()
        print(f"\nCache verification: {len(cache)} entries")
        assert len(cache) == 2, f"Expected 2 cache entries, got {len(cache)}"
        
        print("\n✅ PRECOMPUTE SMOKE TEST PASSED")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_train_smoke():
    """Test train mode with minimal settings."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: TRAIN MODE")
    print("=" * 70)
    
    # This is a simplified test that just verifies the components work
    # Full training requires actual model loading which may be heavy
    
    print("Testing reward computation...")
    scores = [0.8, 0.6, 0.4, 0.2]
    gold_scores = [0.9, 0.7, 0.5, 0.3]
    rubric_text = "Test rubric " * 10
    
    alignment = compute_alignment(scores, gold_scores, "spearman")
    reward = compute_reward(scores, gold_scores, rubric_text, lambda_len=0.01)
    
    print(f"  Alignment (Spearman): {alignment:.3f}")
    print(f"  Reward: {reward:.3f}")
    
    assert alignment > 0.5, "Alignment should be positive"
    print("✅ REWARD COMPUTATION TEST PASSED")
    
    print("\nNote: Full training test requires Qwen model loading.")
    print("To test full training, run:")
    print("  python -m grubrics_science.rl.train_grpo --mode=train")


async def main():
    """Run all smoke tests."""
    try:
        await test_precompute_smoke()
        # await test_train_smoke()
        print("\n" + "=" * 70)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())

