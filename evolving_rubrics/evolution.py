"""
Main evolution module for DR-Tulu Evolving Rubrics.

Contains the main function to evolve rubrics through multiple iterations.
"""

from typing import Dict, Any, Optional

from .config import get_client
from .rubric_generation import (
    generate_original_rubrics,
    generate_adaptive_rubrics,
    update_ground_truth
)
from .response_generation import generate_model_responses
from .evaluation import evaluate_complete_response


async def evolve_rubrics_for_example(
    question: str,
    num_iterations: int = 1,
    num_responses_per_iteration: int = 4,
    initial_rubrics: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Main function to evolve rubrics for a single example through multiple iterations.
    
    This function:
    1. Generates initial rubrics (or uses provided ones)
    2. For each iteration:
       - Generates model responses
       - Generates adaptive rubrics
       - Updates ground truth
       - Evaluates responses
    3. Returns all results
    
    Args:
        question: The question to evolve rubrics for
        num_iterations: Number of evolution iterations to perform
        num_responses_per_iteration: Number of responses to generate per iteration
        initial_rubrics: Optional initial rubrics (if None, will be generated)
        client: Optional client instance
    
    Returns:
        Dictionary containing:
        - 'initial_ground_truth': Initial rubrics
        - 'iterations': List of iteration results, each containing:
          - 'iteration': Iteration number
          - 'adaptive_rubrics': Generated adaptive rubrics
          - 'updated_ground_truth': Updated ground truth after iteration
          - 'responses': Generated responses
          - 'evaluations': Evaluation results for each response
        - 'final_ground_truth': Final ground truth after all iterations
    """
    if client is None:
        client = get_client()
    
    # Step 1: Generate or use initial rubrics
    if initial_rubrics is None:
        print("\n📋 Step 1: Generating initial rubrics with LLM...")
        print("-" * 70)
        initial_ground_truth = await generate_original_rubrics(question, client=client)
        print(f"✓ Generated rubrics: {len(initial_ground_truth['rubrics'])}")
        for i, rubric in enumerate(initial_ground_truth["rubrics"], 1):
            print(f"  {i}. [{rubric['title']}] {rubric['description'][:80]}...")
    else:
        initial_ground_truth = initial_rubrics
        print(f"\n📋 Using provided initial rubrics: {len(initial_ground_truth['rubrics'])}")
    
    iterations = []
    current_ground_truth = initial_ground_truth
    
    # Step 2: Iterate evolution process
    for iteration_num in range(1, num_iterations + 1):
        print(f"\n\n🔄 Iteration {iteration_num}/{num_iterations}")
        print("=" * 70)
        
        # Generate model responses
        print(f"\n🤖 Generating {num_responses_per_iteration} model responses...")
        responses = await generate_model_responses(
            question,
            num_responses=num_responses_per_iteration,
            client=client
        )
        print(f"✓ Generated {len(responses)} responses")
        
        # Generate adaptive rubrics
        print(f"\n🧠 Generating adaptive rubrics...")
        adaptive_rubrics = await generate_adaptive_rubrics(
            question,
            responses,
            existing_rubrics=current_ground_truth["rubrics"],
            client=client
        )
        
        if adaptive_rubrics:
            print("✓ Adaptive rubrics generated:")
            if adaptive_rubrics.get("positive_rubrics"):
                print(f"  Positive rubrics: {len(adaptive_rubrics['positive_rubrics'])}")
            if adaptive_rubrics.get("negative_rubrics"):
                print(f"  Negative rubrics: {len(adaptive_rubrics['negative_rubrics'])}")
        else:
            print("⚠️  Could not generate adaptive rubrics")
            adaptive_rubrics = {"positive_rubrics": [], "negative_rubrics": []}
        
        # Update ground truth
        print(f"\n📝 Updating ground truth...")
        updated_ground_truth = update_ground_truth(
            current_ground_truth,
            adaptive_rubrics
        )
        print(f"✓ Total rubrics: {len(updated_ground_truth['rubrics'])}")
        print(f"  - Persistent: {len(current_ground_truth['rubrics'])}")
        print(f"  - New adaptive: {len(updated_ground_truth['rubrics']) - len(current_ground_truth['rubrics'])}")
        
        # Evaluate responses
        print(f"\n📊 Evaluating responses with all rubrics...")
        evaluations = []
        for i, response in enumerate(responses, 1):
            print(f"  Evaluating Response {i}...", end=" ")
            result = await evaluate_complete_response(response, updated_ground_truth, client=client)
            evaluations.append({
                'response_num': i,
                **result
            })
            print(f"Reward: {result['total_reward']:.3f}")
        
        # Store iteration results
        iterations.append({
            'iteration': iteration_num,
            'adaptive_rubrics': adaptive_rubrics,
            'updated_ground_truth': updated_ground_truth,
            'responses': responses,
            'evaluations': evaluations
        })
        
        # Update current ground truth for next iteration
        current_ground_truth = updated_ground_truth
    
    # Final summary
    print("\n\n" + "=" * 70)
    print("📈 FINAL SUMMARY")
    print("=" * 70)
    
    final_ground_truth = iterations[-1]['updated_ground_truth'] if iterations else initial_ground_truth
    print(f"\nRubrics:")
    print(f"  - Initial: {len(initial_ground_truth['rubrics'])}")
    print(f"  - Final: {len(final_ground_truth['rubrics'])}")
    print(f"  - Total added: {len(final_ground_truth['rubrics']) - len(initial_ground_truth['rubrics'])}")
    
    if iterations:
        last_evaluations = iterations[-1]['evaluations']
        print(f"\nFinal iteration rewards:")
        for eval_result in last_evaluations:
            print(f"  Response {eval_result['response_num']}: {eval_result['total_reward']:.3f}")
        
        best = max(last_evaluations, key=lambda x: x['total_reward'])
        worst = min(last_evaluations, key=lambda x: x['total_reward'])
        print(f"\n✓ Best response: Response {best['response_num']} (reward: {best['total_reward']:.3f})")
        print(f"✗ Worst response: Response {worst['response_num']} (reward: {worst['total_reward']:.3f})")
    
    print("\n" + "=" * 70)
    print("✓ Process completed!")
    print("=" * 70)
    
    return {
        'initial_ground_truth': initial_ground_truth,
        'iterations': iterations,
        'final_ground_truth': final_ground_truth
    }

