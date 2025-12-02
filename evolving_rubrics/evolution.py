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
from .output import save_evolution_history


async def evolve_rubrics_for_example(
    question: str,
    num_iterations: int = 1,
    num_responses_per_iteration: int = 4,
    initial_rubrics: Optional[Dict[str, Any]] = None,
    golden_answer: Optional[str] = None,
    client: Optional[Any] = None,
    save_output: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to evolve rubrics for a single example through multiple iterations.
    
    This function follows the DR-Tulu flow:
    1. Generates initial rubrics (or uses provided ones)
    2. For each iteration:
       - Generates model responses (Policy rollout)
       - Evaluates responses with CURRENT rubrics (Judge)
       - Identifies good vs bad responses based on scores
       - Generates adaptive rubrics from good/bad pairs
       - Updates ground truth with new rubrics
    3. Returns all results
    
    Args:
        question: The question to evolve rubrics for
        num_iterations: Number of evolution iterations to perform
        num_responses_per_iteration: Number of responses to generate per iteration
        initial_rubrics: Optional initial rubrics (if None, will be generated)
        golden_answer: Optional text (golden answer) to guide initial rubric generation (will be generalized)
        client: Optional client instance
        save_output: Whether to save evolution history to a file (default: True)
        output_path: Optional custom path for output file. If None, auto-generates
    
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
        print("\nStep 1: Generating initial rubrics with LLM...")
        if golden_answer:
            print("Using golden answer as reference (will be generalized)...")
        print("-" * 70)
        initial_ground_truth = await generate_original_rubrics(
            question, 
            golden_answer=golden_answer,
            client=client
        )
        print(f"Generated rubrics: {len(initial_ground_truth['rubrics'])}")
        for i, rubric in enumerate(initial_ground_truth["rubrics"], 1):
            print(f"  {i}. [{rubric['title']}] {rubric['description'][:80]}...")
    else:
        initial_ground_truth = initial_rubrics
        print(f"\nUsing provided initial rubrics: {len(initial_ground_truth['rubrics'])}")
    
    iterations = []
    current_ground_truth = initial_ground_truth
    
    # Step 2: Iterate evolution process
    for iteration_num in range(1, num_iterations + 1):
        print(f"\n\nIteration {iteration_num}/{num_iterations}")
        print("=" * 70)
        
        # Generate model responses (Policy rollout)
        print(f"\nGenerating {num_responses_per_iteration} model responses...")
        responses = await generate_model_responses(
            question,
            num_responses=num_responses_per_iteration,
            client=client
        )
        print(f"Generated {len(responses)} responses")
        
        # Judge: Evaluate responses with CURRENT rubrics (before generating new ones)
        print(f"\nJudge: Evaluating responses with current rubrics...")
        evaluations = []
        for i, response in enumerate(responses, 1):
            print(f"  Evaluating Response {i}...", end=" ")
            result = await evaluate_complete_response(response, current_ground_truth, client=client)
            evaluations.append({
                'response_num': i,
                'response': response,
                **result
            })
            print(f"Reward: {result['total_reward']:.3f}")
        
        # Identify good vs bad responses based on scores
        sorted_evaluations = sorted(evaluations, key=lambda x: x['total_reward'], reverse=True)
        num_good = max(1, len(responses) // 2)  # Top half are "good"
        good_responses = [e['response'] for e in sorted_evaluations[:num_good]]
        bad_responses = [e['response'] for e in sorted_evaluations[num_good:]]
        
        print(f"\nResponse ranking:")
        print(f"  Good responses (top {len(good_responses)}): {[e['response_num'] for e in sorted_evaluations[:num_good]]}")
        print(f"  Bad responses (bottom {len(bad_responses)}): {[e['response_num'] for e in sorted_evaluations[num_good:]]}")
        
        # Generate adaptive rubrics based on good vs bad pairs
        print(f"\nGenerating adaptive rubrics from good vs bad responses...")
        adaptive_rubrics = await generate_adaptive_rubrics(
            question,
            responses,
            existing_rubrics=current_ground_truth["rubrics"],
            good_responses=good_responses,
            bad_responses=bad_responses,
            client=client
        )
        
        if adaptive_rubrics:
            print("Adaptive rubrics generated:")
            if adaptive_rubrics.get("positive_rubrics"):
                print(f"  Positive rubrics: {len(adaptive_rubrics['positive_rubrics'])}")
            if adaptive_rubrics.get("negative_rubrics"):
                print(f"  Negative rubrics: {len(adaptive_rubrics['negative_rubrics'])}")
        else:
            print("Warning: Could not generate adaptive rubrics")
            adaptive_rubrics = {"positive_rubrics": [], "negative_rubrics": []}
        
        # Update ground truth
        print(f"\nUpdating ground truth...")
        updated_ground_truth = update_ground_truth(
            current_ground_truth,
            adaptive_rubrics
        )
        print(f"Total rubrics: {len(updated_ground_truth['rubrics'])}")
        print(f"  - Persistent: {len(current_ground_truth['rubrics'])}")
        print(f"  - New adaptive: {len(updated_ground_truth['rubrics']) - len(current_ground_truth['rubrics'])}")
        
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
    print("FINAL SUMMARY")
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
        print(f"\nBest response: Response {best['response_num']} (reward: {best['total_reward']:.3f})")
        print(f"Worst response: Response {worst['response_num']} (reward: {worst['total_reward']:.3f})")
    
    print("\n" + "=" * 70)
    print("Process completed!")
    print("=" * 70)
    
    result = {
        'initial_ground_truth': initial_ground_truth,
        'iterations': iterations,
        'final_ground_truth': final_ground_truth
    }
    
    # Save evolution history to file
    if save_output:
        try:
            output_file = save_evolution_history(result, output_path=output_path)
            print(f"\nEvolution history saved to: {output_file}")
        except Exception as e:
            print(f"\nWarning: Could not save evolution history: {e}")
    
    return result

