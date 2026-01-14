"""Smoke test for GRubrics Science.

Tests precompute and train modes with minimal settings.
You can modify the parameters at the top of this file to customize the test.
"""

import asyncio
import yaml
from pathlib import Path
import tempfile
import shutil
import os

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from grubrics_science.tasks.frontierscience import FrontierScienceTask
from grubrics_science.llm.client import AzureOpenAIClient
from grubrics_science.llm.prompts import get_answer_policy_prompt, get_grubrics_prompt
from grubrics_science.judge.judge import Judge
from grubrics_science.rewards.alignment import compute_reward, compute_alignment
from grubrics_science.utils.seeding import set_seed
from grubrics_science.rl.model_wrap import GRubricsModelWrapper

# ============================================================================
# CONFIGURACIÓN - Modifica estos parámetros según necesites
# ============================================================================

# Execution mode flags
RUN_PRECOMPUTE = False  # Si True, ejecuta precompute (genera respuestas y gold scores)
RUN_TRAIN = True  # Si True, ejecuta train (requiere Qwen cargado y cache existente)

# Precompute settings
NUM_QUESTIONS = 5  # Número de preguntas a procesar
K_ANSWERS = 2  # Número de respuestas por pregunta
USE_TEMP_CACHE = False  # Si True, usa cache temporal (se borra después). Si False, usa cache real. -> todas las respuestas generadas, scores calculados, etc.
MAX_ANSWER_TOKENS = 512  # Máximo de tokens para respuestas generadas (reducir si se cortan mucho)

# Train settings
K_TRAIN = 2  # Subset de respuestas para entrenar
M_RUBRICS = 2  # Número de rúbricas a generar
NUM_STEPS = 2  # Número de steps de entrenamiento

# Model settings (para train)
QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Modelo Qwen a usar
QWEN_MODEL = "Qwen/Qwen3-8B"  # Modelo Qwen a usar

DEVICE = "cuda"  # "cuda" o "cpu"
DTYPE = "bfloat16"  # "bfloat16" o "float32"

# Judge/Answer Policy settings
# These will use environment variables from .env if not specified
# IMPORTANT: In Azure OpenAI, use the DEPLOYMENT NAME, not the model name
# Example: If your deployment is named "gpt-5-chat", use that
# You can override here or set in .env:
#   RUBRIC_JUDGE_MODEL=gpt-5-chat (your Azure deployment name)
#   RUBRIC_GENERATION_MODEL=gpt-5-chat (your Azure deployment name)
JUDGE_MODEL = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-5-chat")
ANSWER_POLICY_MODEL = os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-5-chat")
USE_AZURE = os.environ.get("USE_AZURE_OPENAI", "true").lower() == "true"

# Print configuration for debugging
print(f"Loaded config from .env:")
print(f"  JUDGE_MODEL: {JUDGE_MODEL}")
print(f"  ANSWER_POLICY_MODEL: {ANSWER_POLICY_MODEL}")
print(f"  USE_AZURE: {USE_AZURE}")
print(f"  AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'not set')}")

# ============================================================================


async def test_precompute_smoke(cache_dir: str):
    """
    Test precompute mode with configurable parameters.
    
    Args:
        cache_dir: Path to cache directory (as string)
    """
    print("=" * 70)
    print("SMOKE TEST: PRECOMPUTE MODE")
    print("=" * 70)
    print(f"Configuration: {NUM_QUESTIONS} questions, K={K_ANSWERS} answers per question")
    print(f"Cache directory: {cache_dir}")
    print(f"\nAzure OpenAI Configuration:")
    print(f"  JUDGE_MODEL: {JUDGE_MODEL}")
    print(f"  ANSWER_POLICY_MODEL: {ANSWER_POLICY_MODEL}")
    print(f"  USE_AZURE: {USE_AZURE}")
    print(f"  AZURE_API_BASE: {os.environ.get('AZURE_API_BASE', 'not set')}")
    if USE_AZURE and not os.environ.get('AZURE_API_KEY'):
        print("  ⚠️  WARNING: AZURE_API_KEY not found in environment!")
    
    try:
        # Load dataset
        dataset_path = Path("frontierscience-research/test.jsonl")
        if not dataset_path.exists():
            dataset_path = Path(__file__).parent / dataset_path
        
        task = FrontierScienceTask(
            dataset_path=str(dataset_path),
            cache_dir=str(cache_dir)
        )
        
        print(f"Loaded {len(task)} questions from dataset")
        
        # Limit to configured number of questions
        test_questions = task.examples[:NUM_QUESTIONS]
        
        # Initialize clients
        answer_policy_client = AzureOpenAIClient(
            model=ANSWER_POLICY_MODEL,
            use_azure=USE_AZURE
        )
        
        judge = Judge(
            model=JUDGE_MODEL,
            use_azure=USE_AZURE
        )
        
        set_seed(42)
        
        # Process questions with K answers each
        for idx, example in enumerate(test_questions):
            print(f"\nQuestion {idx+1}/{len(test_questions)} (ID: {example.question_id})")
            question = example.problem
            golden_rubric = example.golden_rubric
            
            # Generate K answers in parallel
            print(f"  Generating {K_ANSWERS} answers in parallel...")
            answer_tasks = [
                answer_policy_client.generate(
                    prompt=get_answer_policy_prompt(question, "normal"),
                    max_tokens=MAX_ANSWER_TOKENS,
                    temperature=0.8
                )
                for _ in range(K_ANSWERS)
            ]
            answer_texts = await asyncio.gather(*answer_tasks)
            answers = [answer.strip() for answer in answer_texts]
            for i, answer in enumerate(answers):
                print(f"  Generated answer {i+1}/{K_ANSWERS}: {len(answer)} chars")
            
            # Compute gold scores in parallel
            print(f"  Evaluating {K_ANSWERS} answers with Judge in parallel...")
            score_matrix, details_matrix = await judge.evaluate_multiple_answers(
                question=question,
                answers=answers,
                rubrics=[golden_rubric],
                return_details=True
            )
            gold_scores = [score_matrix[i][0] for i in range(len(answers))]
            
            # Extract gold details (item-by-item breakdowns) for each answer
            gold_details = []
            for i, detail_list in enumerate(details_matrix):
                # Each detail_list contains evaluations for all rubrics
                # For gold scores, we only have one rubric (the golden one)
                gold_detail = detail_list[0] if detail_list else {}
                gold_details.append(gold_detail)
            
            for i, (score, detail) in enumerate(zip(gold_scores, gold_details)):
                print(f"  Gold score {i+1}: {score:.3f}")
                item_scores = detail.get('item_scores', [])
                if item_scores:
                    print(f"    Item-by-item breakdown:")
                    for item in item_scores[:3]:  # Show first 3 items
                        item_desc = item.get('item_description', 'Unknown item')
                        item_score = item.get('score', 0.0)
                        item_notes = item.get('notes', '')[:100]
                        print(f"      - {item_desc}: {item_score:.3f} - {item_notes}...")
                    if len(item_scores) > 3:
                        print(f"      ... ({len(item_scores) - 3} more items)")
                else:
                    notes = detail.get('notes', 'No explanation')[:150]
                    print(f"    Explanation: {notes}...")
            
            # Save to cache (including item-by-item details)
            task.save_cache_entry(
                question_id=example.question_id,
                question=question,
                answers=answers,
                gold_scores=gold_scores,
                gold_details=gold_details
            )
            print(f"  Saved to cache")
        
        # Verify cache
        cache = task.load_cache()
        print(f"\nCache verification: {len(cache)} entries")
        assert len(cache) == len(test_questions), f"Expected {len(test_questions)} cache entries, got {len(cache)}"
        
        print("\n✅ PRECOMPUTE SMOKE TEST PASSED")
        print(f"Cache saved to: {cache_dir}")
        
    except Exception as e:
        print(f"\n❌ PRECOMPUTE FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_train_smoke(cache_dir: str):
    """Test train mode with configurable parameters."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: TRAIN MODE")
    print("=" * 70)
    print(f"Configuration: k_train={K_TRAIN}, M={M_RUBRICS}, steps={NUM_STEPS}")
    
    # Load cache
    dataset_path = Path("frontierscience-research/test.jsonl")
    if not dataset_path.exists():
        dataset_path = Path(__file__).parent / dataset_path
    
    task = FrontierScienceTask(
        dataset_path=str(dataset_path),
        cache_dir=cache_dir
    )
    
    cache = task.load_cache()
    if len(cache) == 0:
        print("❌ No cache found! Run precompute first.")
        return
    
    print(f"Loaded cache: {len(cache)} questions")
    
    # Initialize GRubrics model
    print(f"\nLoading Qwen model: {QWEN_MODEL}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    try:
        grubrics_model = GRubricsModelWrapper(
            model_name=QWEN_MODEL,
            device=DEVICE,
            dtype=DTYPE
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Skipping train test. Make sure Qwen model is available.")
        return
    
    # Initialize Judge
    judge = Judge(
        model=JUDGE_MODEL,
        use_azure=USE_AZURE
    )
    
    # Get optimizer
    optimizer = grubrics_model.get_optimizer(learning_rate=1e-5)
    
    # Get question IDs
    question_ids = list(cache.keys())[:NUM_STEPS]  # Limit to NUM_STEPS questions
    
    print(f"\nTraining on {len(question_ids)} questions...")
    
    import random
    import numpy as np
    import torch
    from grubrics_science.rewards.alignment import compute_reward, compute_alignment, length_penalty
    
    for step, q_id in enumerate(question_ids):
        print(f"\nStep {step+1}/{len(question_ids)}: Question {q_id}")
        
        cached_data = cache[q_id]
        question = cached_data['question']
        all_answers = cached_data['answers']
        all_gold_scores = cached_data['gold_scores']
        
        # Sample k_train answers
        if len(all_answers) < K_TRAIN:
            selected_indices = list(range(len(all_answers)))
        else:
            selected_indices = random.sample(range(len(all_answers)), K_TRAIN)
        
        selected_answers = [all_answers[i] for i in selected_indices]
        selected_gold_scores = [all_gold_scores[i] for i in selected_indices]
        
        # Get anchors
        best_idx = np.argmax(selected_gold_scores)
        worst_idx = np.argmin(selected_gold_scores)
        best_answer_excerpt = selected_answers[best_idx]
        worst_answer_excerpt = selected_answers[worst_idx]
        
        # Generate M rubrics
        prompt = get_grubrics_prompt(
            question=question,
            # best_answer_excerpt=best_answer_excerpt, # TODO: revisar si esto va o no...
            # worst_answer_excerpt=worst_answer_excerpt
        )
        
        print(f"  Generating {M_RUBRICS} rubrics...")
        rubric_tokens_list = []
        rubrics = []
        for _ in range(M_RUBRICS):
            token_ids, prompt_len, rubric_text = grubrics_model.sample_rubric_tokens(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=1.0,
                top_k=50
            )
            rubric_tokens_list.append((token_ids, prompt_len))
            rubrics.append(rubric_text.strip())
            print(f"    Rubric {len(rubrics)}: {len(rubric_text)} chars")
        
        # Evaluate rubrics with Judge
        print(f"  Evaluating rubrics with Judge...")
        score_matrix, details_matrix = await judge.evaluate_multiple_answers(
            question=question,
            answers=selected_answers,
            rubrics=rubrics,
            return_details=True
        )
        
        # Print detailed evaluations
        print(f"  Judge evaluation details:")
        for j, rubric in enumerate(rubrics):
            print(f"    Rubric {j+1}:")
            for i, answer_details in enumerate(details_matrix):
                detail = answer_details[j]
                total_score = detail.get('total_score', detail.get('score', 0.0))
                print(f"      Answer {i+1}: total_score={total_score:.3f}")
                item_scores = detail.get('item_scores', [])
                if item_scores:
                    print(f"        Item-by-item breakdown:")
                    for item in item_scores[:2]:  # Show first 2 items per answer
                        item_desc = item.get('item_description', 'Unknown item')
                        item_score = item.get('score', 0.0)
                        item_notes = item.get('notes', '')[:80]
                        print(f"          - {item_desc}: {item_score:.3f} - {item_notes}...")
                    if len(item_scores) > 2:
                        print(f"          ... ({len(item_scores) - 2} more items)")
                else:
                    notes = detail.get('notes', 'No explanation')
                    print(f"        Explanation: {notes[:150]}...")
        
        # Compute rewards
        rubric_rewards = []
        for j, rubric in enumerate(rubrics):
            scores = [score_matrix[i][j] for i in range(len(selected_answers))]
            reward = compute_reward(
                scores=scores,
                gold_scores=selected_gold_scores,
                rubric_text=rubric,
                alignment_metric="spearman",
                lambda_len=0.01,
                length_penalty_type="characters"
            )
            rubric_rewards.append(reward)
        
        # Compute advantages
        rewards_tensor = torch.tensor(rubric_rewards, dtype=torch.float32, device=grubrics_model.device)
        mu = rewards_tensor.mean()
        advantages = rewards_tensor - mu
        
        # Compute logprobs
        print(f"  Computing logprobs and updating model...")
        all_logprobs = []
        for token_ids, prompt_len in rubric_tokens_list:
            token_ids = token_ids.to(grubrics_model.device)
            logprobs_per_token = grubrics_model.compute_logprobs_per_token(token_ids, prompt_len)
            all_logprobs.append(logprobs_per_token)
        
        # Pad and compute loss
        max_gen_len = max(len(lp) for lp in all_logprobs)
        padded_logprobs = []
        for lp in all_logprobs:
            padding = torch.zeros(max_gen_len - len(lp), device=grubrics_model.device)
            padded_lp = torch.cat([lp, padding])
            padded_logprobs.append(padded_lp)
        
        logprobs_tensor = torch.stack(padded_logprobs)
        pg_obj = (logprobs_tensor * advantages.unsqueeze(-1)).sum()
        num_valid = sum(len(lp) for lp in all_logprobs)
        pg_obj = pg_obj / max(num_valid, 1)
        loss = -pg_obj
        
        # Backward and step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(grubrics_model.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"  Mean reward: {rewards_tensor.mean():.4f}, Loss: {loss.item():.6f}")
    
    print("\n✅ TRAIN SMOKE TEST PASSED")


async def main():
    """Run smoke tests based on configuration flags."""
    # Setup cache directory (centralized in main)
    if USE_TEMP_CACHE:
        temp_dir = Path(tempfile.mkdtemp())
        cache_dir = str(temp_dir / "cache")
        cleanup_cache = True
    else:
        cache_dir = "grubrics_science/data/cache"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cleanup_cache = False
    
    try:
        # Run precompute if enabled
        if RUN_PRECOMPUTE:
            await test_precompute_smoke(cache_dir)
        else:
            print("Skipping precompute (RUN_PRECOMPUTE=False)")
            # Check if cache exists for train
            cache_path = Path(cache_dir) / "precompute_cache.jsonl"
            if not cache_path.exists() and RUN_TRAIN:
                print(f"⚠️  WARNING: Cache not found at {cache_path}")
                print("   Set RUN_PRECOMPUTE=True to generate cache, or ensure cache exists.")
        
        # Run train if enabled
        if RUN_TRAIN:
            await test_train_smoke(cache_dir)
        else:
            print("Skipping train (RUN_TRAIN=False)")
        
        print("\n" + "=" * 70)
        print("SMOKE TESTS COMPLETED")
        print("=" * 70)
        if not USE_TEMP_CACHE:
            print(f"Cache location: {cache_dir}")
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup temp cache if used
        if cleanup_cache:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())

