"""Main training script for GRubrics Science.

Supports two modes:
- precompute: Generate and cache answers + gold scores
- train: Train GRubrics model with RL
"""

import argparse
import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import random
import numpy as np
import torch
import sys

from ..tasks.frontierscience import FrontierScienceTask
from ..llm.client import AzureOpenAIClient, QwenClient
from ..llm.prompts import (
    get_answer_policy_prompt,
    get_grubrics_prompt
)
from ..judge.judge import Judge
from ..rewards.alignment import compute_reward
from ..rl.model_wrap import GRubricsModelWrapper
from ..utils.io import load_cache
from ..utils.logging import setup_logging, get_logger, log_metrics, DummyWandb
from ..utils.seeding import set_seed, get_deterministic_seed


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


async def precompute_mode(config: Dict[str, Any], logger):
    """Precompute answers and gold scores for all questions."""
    logger.info("=" * 70)
    logger.info("PRECOMPUTE MODE")
    logger.info("=" * 70)
    
    # Setup
    dataset_path = config['dataset_path']
    # Handle relative paths
    if not Path(dataset_path).is_absolute():
        repo_root = Path(__file__).parent.parent.parent
        dataset_path = str(repo_root / dataset_path)
    
    task = FrontierScienceTask(
        dataset_path=dataset_path,
        cache_dir=config['cache_dir']
    )
    
    answer_policy_client = AzureOpenAIClient(
        model=config['answer_policy']['model'],
        use_azure=config['answer_policy']['use_azure']
    )
    
    judge = Judge(
        model=config['judge']['model'],
        use_azure=config['judge']['use_azure']
    )
    
    K = config['precompute']['K']
    seed = config['precompute']['seed']
    set_seed(seed)
    
    logger.info(f"Dataset: {len(task)} questions")
    logger.info(f"K (answers per question): {K}")
    logger.info(f"Seed: {seed}")
    
    # Load existing cache
    cache = task.load_cache()
    logger.info(f"Existing cache entries: {len(cache)}")
    
    # Process each question
    for idx, example in enumerate(task):
        q_id = example.question_id
        
        # Skip if already cached
        if q_id in cache:
            logger.info(f"Question {idx+1}/{len(task)} (ID: {q_id}): Already cached, skipping")
            continue
        
        logger.info(f"Question {idx+1}/{len(task)} (ID: {q_id}): Generating {K} answers...")
        
        question = example.problem
        golden_rubric = example.golden_rubric
        
        # Generate K answers with diversity
        answers = []
        instruction_types = []
        
        # 2 low temp
        for _ in range(2):
            instruction_types.append("low_temp")
        # 2 high temp
        for _ in range(2):
            instruction_types.append("high_temp")
        # 2 failure modes
        instruction_types.append("failure_mode_1")
        instruction_types.append("failure_mode_2")
        # 2 medium
        for _ in range(2):
            instruction_types.append("normal")
        
        # Shuffle for randomness
        random.shuffle(instruction_types)
        
        for i, inst_type in enumerate(instruction_types[:K]):
            prompt = get_answer_policy_prompt(question, inst_type)
            
            # Use deterministic seed for reproducibility
            answer_seed = get_deterministic_seed(seed, q_id, i)
            # Note: Azure OpenAI doesn't support seed directly, but we can vary temperature slightly
            temp = 0.3 if inst_type == "low_temp" else (1.2 if inst_type == "high_temp" else 0.8)
            
            answer = await answer_policy_client.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=temp
            )
            answers.append(answer.strip())
        
        logger.info(f"  Generated {len(answers)} answers")
        
        # Compute gold scores using golden rubric
        logger.info(f"  Computing gold scores with golden rubric...")
        gold_scores = []
        
        for i, answer in enumerate(answers):
            scores = await judge.evaluate_batch(
                question=question,
                answer=answer,
                rubrics=[golden_rubric],
                answer_id=f"a{i+1}"
            )
            gold_scores.append(scores[0])  # Single rubric
        
        logger.info(f"  Gold scores: {[f'{s:.3f}' for s in gold_scores]}")
        
        # Save to cache
        task.save_cache_entry(
            question_id=q_id,
            question=question,
            answers=answers,
            gold_scores=gold_scores,
            metadata={'subject': example.subject}
        )
        
        logger.info(f"  Saved to cache")
    
    logger.info("=" * 70)
    logger.info("PRECOMPUTE COMPLETE")
    logger.info("=" * 70)


async def train_mode(config: Dict[str, Any], logger):
    """Train GRubrics model with RL."""
    logger.info("=" * 70)
    logger.info("TRAIN MODE")
    logger.info("=" * 70)
    
    # Setup
    dataset_path = config['dataset_path']
    # Handle relative paths
    if not Path(dataset_path).is_absolute():
        repo_root = Path(__file__).parent.parent.parent
        dataset_path = str(repo_root / dataset_path)
    
    task = FrontierScienceTask(
        dataset_path=dataset_path,
        cache_dir=config['cache_dir']
    )
    
    # Load cache
    cache = task.load_cache()
    if len(cache) == 0:
        raise ValueError("Cache is empty! Run precompute mode first.")
    
    logger.info(f"Loaded cache: {len(cache)} questions")
    
    # Initialize GRubrics model
    logger.info("Initializing GRubrics model...")
    grubrics_model = GRubricsModelWrapper(
        model_name=config['model']['qwen_model_name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    
    # Initialize Judge
    judge = Judge(
        model=config['judge']['model'],
        use_azure=config['judge']['use_azure']
    )
    
    # Training config
    k_train = config['training']['k_train']
    M = config['training']['M']
    num_epochs = config['training']['num_epochs']
    examples_per_step = config['training']['examples_per_step']
    device_batch_size = config['training']['device_batch_size']
    
    max_new_tokens = config['generation']['max_new_tokens']
    temperature = config['generation']['temperature']
    top_k = config['generation']['top_k']
    
    alignment_metric = config['reward']['alignment_metric']
    lambda_len = config['reward']['lambda_len']
    length_penalty_type = config['reward']['length_penalty_type']
    
    # Optimizer
    optimizer = grubrics_model.get_optimizer(learning_rate=1e-5)
    
    # Logging
    log_every = config['logging']['log_every']
    save_every = config['logging']['save_every']
    eval_every = config['logging']['eval_every']
    use_wandb = config['logging']['use_wandb']
    
    wandb_logger = None
    if use_wandb:
        try:
            import wandb
            wandb_logger = wandb.init(
                project=config['logging']['wandb_project'],
                name="grubrics-train"
            )
        except ImportError:
            logger.warning("wandb not available, using dummy logger")
            wandb_logger = DummyWandb()
    else:
        wandb_logger = DummyWandb()
    
    # Training loop
    step = 0
    num_steps = (len(cache) // examples_per_step) * num_epochs
    
    logger.info(f"Training for {num_steps} steps")
    logger.info(f"k_train={k_train}, M={M}, examples_per_step={examples_per_step}")
    
    # Get list of question IDs
    question_ids = list(cache.keys())
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Shuffle questions
        random.shuffle(question_ids)
        
        for batch_start in range(0, len(question_ids), examples_per_step):
            batch_ids = question_ids[batch_start:batch_start + examples_per_step]
            
            # Process batch
            all_rewards = []
            all_alignments = []
            all_lengths = []
            
            for q_id in batch_ids:
                cached_data = cache[q_id]
                question = cached_data['question']
                all_answers = cached_data['answers']
                all_gold_scores = cached_data['gold_scores']
                
                # Sample k_train answers
                if len(all_answers) < k_train:
                    selected_indices = list(range(len(all_answers)))
                else:
                    selected_indices = random.sample(range(len(all_answers)), k_train)
                
                selected_answers = [all_answers[i] for i in selected_indices]
                selected_gold_scores = [all_gold_scores[i] for i in selected_indices]
                
                # Get anchors (best/worst)
                best_idx = np.argmax(selected_gold_scores)
                worst_idx = np.argmin(selected_gold_scores)
                best_answer_excerpt = selected_answers[best_idx][:300]
                worst_answer_excerpt = selected_answers[worst_idx][:300]
                
                # Generate M rubrics
                prompt = get_grubrics_prompt(
                    question=question,
                    best_answer_excerpt=best_answer_excerpt,
                    worst_answer_excerpt=worst_answer_excerpt
                )
                
                rubrics = grubrics_model.generate_rubrics(
                    prompt=prompt,
                    num_samples=M,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Evaluate rubrics with Judge (batched)
                score_matrix = await judge.evaluate_multiple_answers(
                    question=question,
                    answers=selected_answers,
                    rubrics=rubrics
                )
                
                # Compute rewards for each rubric
                rubric_rewards = []
                for j, rubric in enumerate(rubrics):
                    scores = [score_matrix[i][j] for i in range(len(selected_answers))]
                    reward = compute_reward(
                        scores=scores,
                        gold_scores=selected_gold_scores,
                        rubric_text=rubric,
                        alignment_metric=alignment_metric,
                        lambda_len=lambda_len,
                        length_penalty_type=length_penalty_type
                    )
                    rubric_rewards.append(reward)
                    
                    # Logging metrics
                    from ..rewards.alignment import compute_alignment, length_penalty
                    alignment = compute_alignment(scores, selected_gold_scores, alignment_metric)
                    length = length_penalty(rubric, length_penalty_type)
                    all_rewards.append(reward)
                    all_alignments.append(alignment)
                    all_lengths.append(length)
                
                # Compute advantages
                rewards_tensor = torch.tensor(rubric_rewards, dtype=torch.float32)
                advantages = rewards_tensor - rewards_tensor.mean()
                
                # REINFORCE update (simplified, no actual gradient computation for now)
                # TODO: Implement proper gradient computation with logprobs
                logger.debug(f"  Question {q_id}: Mean reward={rewards_tensor.mean():.4f}, "
                           f"Mean alignment={np.mean(all_alignments[-M:]):.4f}")
            
            # Logging
            if step % log_every == 0:
                metrics = {
                    'mean_reward': np.mean(all_rewards) if all_rewards else 0.0,
                    'mean_alignment': np.mean(all_alignments) if all_alignments else 0.0,
                    'mean_length': np.mean(all_lengths) if all_lengths else 0.0,
                }
                log_metrics(logger, step, metrics, wandb_logger)
            
            step += 1
            
            if step >= num_steps:
                break
        
        if step >= num_steps:
            break
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    if wandb_logger:
        wandb_logger.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GRubrics Science Training")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["precompute", "train"],
        help="Mode: precompute or train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: configs/default.yaml)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load config
    config = load_config(args.config)
    
    # Run mode
    if args.mode == "precompute":
        asyncio.run(precompute_mode(config, logger))
    elif args.mode == "train":
        asyncio.run(train_mode(config, logger))


if __name__ == "__main__":
    main()

