"""Precompute answers and gold scores for open-domain datasets.

For each question in FrontierScience:
  1. Answer Policy generates K diverse answers (long, research-level).
  2. Judge evaluates each answer using the golden rubric → gold_scores.
  3. Results are saved to a JSONL cache file.

This runs ONCE before training. The FrontierScienceAdapter reads the cache
and includes answers + gold_scores in the parquet's extra_info.

Usage:
    # Validate with 2 questions first:
    python -m grubrics_science.data.precompute \
        --model gpt-5-chat --max_tokens 4096 --limit 2

    # Full run (after validation):
    python -m grubrics_science.data.precompute \
        --model gpt-5-chat --max_tokens 4096
"""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def generate_answers(
    client,
    question: str,
    num_answers: int = 6,
    max_tokens: int = 4096,
) -> List[str]:
    """Generate diverse answers using the Answer Policy."""
    from ..llm.prompts import get_answer_policy_prompt

    # Mix of instruction types for quality diversity (all produce long answers)
    instruction_types = [
        "rigorous",       # precise, all derivations
        "conceptual",     # physical intuition, less math
        "exploratory",    # multiple approaches
        "tangential",     # diluted with secondary material
        "overconfident",  # skips justifications
        "shallow",        # broad but not deep
    ]
    if num_answers > len(instruction_types):
        instruction_types = (instruction_types * ((num_answers // len(instruction_types)) + 1))
    instruction_types = instruction_types[:num_answers]
    random.shuffle(instruction_types)

    # GPT-5+ only supports temperature=1 (default).
    # Diversity comes from different instruction types instead.
    tasks = []
    for inst_type in instruction_types:
        prompt = get_answer_policy_prompt(question, inst_type)
        tasks.append(
            client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Answer generation failed: %s", r)
            answers.append("")
        else:
            answers.append(r.strip())

    return answers


async def evaluate_with_golden_rubric(
    judge,
    question: str,
    answers: List[str],
    golden_rubric: str,
    num_evals: int = 1,
) -> List[float]:
    """Evaluate answers using the golden rubric. Returns gold_scores.

    Args:
        num_evals: Number of evaluations to average per answer.
            Higher values reduce noise from stochastic LLMs (e.g. temperature=1).
    """
    import numpy as np

    all_scores = []
    for _ in range(num_evals):
        # Clear cache so each evaluation is independent
        judge._cache = {}
        scores = await judge.evaluate_answers_batched(
            question=question,
            answers=answers,
            rubric=golden_rubric,
        )
        all_scores.append(scores)

    # Average across evaluations
    avg_scores = np.mean(all_scores, axis=0).tolist()
    return avg_scores


async def precompute_dataset(
    dataset_path: str,
    output_cache: str,
    num_answers: int = 6,
    max_tokens: int = 4096,
    model: str = "gpt-4o-mini",
    use_azure: bool = True,
    limit: int = 0,
    num_evals: int = 3,
):
    """Run the full precompute pipeline.

    Args:
        limit: If > 0, only process this many questions (for validation).
        num_evals: Number of Judge evaluations to average per answer.
            Higher values produce more stable gold_scores. Default 3.
    """
    from ..llm.client import AzureOpenAIClient
    from ..judge.judge import Judge

    # Load dataset
    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                items.append({
                    "question_id": str(idx),
                    "problem": record["problem"],
                    "golden_rubric": record["answer"],
                    "subject": record.get("subject", "physics"),
                })

    if limit > 0:
        items = items[:limit]
        logger.info("Limiting to %d questions (validation mode)", limit)

    logger.info("Loaded %d questions from %s", len(items), dataset_path)
    logger.info("Model: %s | max_tokens: %d | answers_per_question: %d | judge_evals: %d", model, max_tokens, num_answers, num_evals)

    # Load existing cache (skip already-computed questions)
    cache_path = Path(output_cache)
    existing: Dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing[entry["question_id"]] = entry
        logger.info("Existing cache: %d entries", len(existing))

    # Initialise clients
    client = AzureOpenAIClient(model=model, use_azure=use_azure)
    judge = Judge(client=client, max_concurrent=5, max_retries=3, timeout=120.0, max_cache_size=500)

    # Process each question
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "a", encoding="utf-8") as f_out:
        for i, item in enumerate(items):
            qid = item["question_id"]
            if qid in existing:
                logger.info("[%d/%d] %s — already cached, skipping.", i + 1, len(items), qid)
                continue

            logger.info("[%d/%d] %s — generating %d answers...", i + 1, len(items), qid, num_answers)

            answers = await generate_answers(client, item["problem"], num_answers, max_tokens)

            # Log answer lengths for sanity check
            lengths = [len(a) for a in answers]
            logger.info(
                "[%d/%d] %s — answer lengths (chars): %s",
                i + 1, len(items), qid, lengths,
            )

            logger.info("[%d/%d] %s — evaluating with golden rubric (%d evals to average)...", i + 1, len(items), qid, num_evals)

            gold_scores = await evaluate_with_golden_rubric(
                judge, item["problem"], answers, item["golden_rubric"],
                num_evals=num_evals,
            )

            # Log gold scores + variance
            import numpy as np
            scores_arr = np.array(gold_scores)
            logger.info(
                "[%d/%d] %s — gold_scores: %s (std=%.3f, range=%.3f-%.3f)",
                i + 1, len(items), qid,
                [f"{s:.3f}" for s in gold_scores],
                scores_arr.std(),
                scores_arr.min(),
                scores_arr.max(),
            )

            entry = {
                "question_id": qid,
                "question": item["problem"],
                "subject": item["subject"],
                "golden_rubric": item["golden_rubric"],
                "answers": answers,
                "gold_scores": gold_scores,
            }

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_out.flush()

    logger.info("Precompute complete. Cache: %s", cache_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute answers + gold scores")
    parser.add_argument(
        "--dataset_path",
        default="data/frontierscience-research/test.jsonl",
        help="Path to FrontierScience JSONL file",
    )
    parser.add_argument(
        "--output_cache",
        default="data/cache/frontierscience_precompute.jsonl",
        help="Path to output cache JSONL",
    )
    parser.add_argument("--num_answers", type=int, default=6, help="Answers per question")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per answer")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for answer generation + judge")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0=all, use for validation)")
    parser.add_argument("--num_evals", type=int, default=3, help="Judge evaluations to average per answer (reduces noise)")
    parser.add_argument("--no_azure", action="store_true", help="Use OpenAI directly instead of Azure")

    args = parser.parse_args()
    asyncio.run(
        precompute_dataset(
            dataset_path=args.dataset_path,
            output_cache=args.output_cache,
            num_answers=args.num_answers,
            max_tokens=args.max_tokens,
            model=args.model,
            use_azure=not args.no_azure,
            limit=args.limit,
            num_evals=args.num_evals,
        )
    )


if __name__ == "__main__":
    main()
