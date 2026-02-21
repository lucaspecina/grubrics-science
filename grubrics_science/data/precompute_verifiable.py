"""Precompute answers + programmatic gold scores for verifiable domains.

For each question in GSM8K/MATH:
  1. Answer Policy generates 1 answer.
  2. Correctness is verified programmatically.
  3. 2-3 perturbations are created to guarantee a mix of correct/incorrect.
  4. Gold_scores are assigned: 1.0 for correct, 0.0 for incorrect.
  5. Results are saved to a JSONL cache file.

Why perturbation? GPT-5.2 with diverse instruction types produces "all or
nothing" results per question (same error across all styles). Perturbation
guarantees variance in gold_scores so Spearman is well-defined.

Usage:
    # Validate with 5 questions first:
    python -m grubrics_science.data.precompute_verifiable \
        --dataset gsm8k --limit 5

    # Full run:
    python -m grubrics_science.data.precompute_verifiable \
        --dataset gsm8k --model gpt-5.2-chat
"""

import argparse
import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Programmatic answer checking
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} in MATH-style solutions."""
    if "\\boxed{" not in text:
        return ""
    start = text.rfind("\\boxed{") + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[start:end - 1].strip()


def extract_hash_answer(text: str) -> str:
    """Extract number after #### in GSM8K-style solutions."""
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    return ""


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    # Try to normalize as number
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        # Leave as-is for LaTeX expressions etc
        return s.lower().strip()


def check_correct(response: str, gold_answer: str, dataset: str) -> Tuple[bool, str]:
    """Check if a response is correct. Returns (is_correct, extracted_answer)."""
    if dataset == "gsm8k":
        extracted = extract_hash_answer(response)
    else:
        # MATH: try boxed first, then hash
        extracted = extract_boxed(response)
        if not extracted:
            extracted = extract_hash_answer(response)

    if not extracted:
        return False, ""

    return normalize_answer(extracted) == normalize_answer(gold_answer), extracted


# ---------------------------------------------------------------------------
# Perturbation strategies
# ---------------------------------------------------------------------------

def perturb_final_number(response: str, gold_answer: str, dataset: str) -> str:
    """Change the final number in the response to a wrong one."""
    gold_norm = normalize_answer(gold_answer)
    try:
        gold_num = float(gold_norm)
    except ValueError:
        # Non-numeric gold answer (LaTeX), use simple text replacement
        return _perturb_text_answer(response, gold_answer)

    # Generate a plausible wrong number
    wrong_numbers = []
    if gold_num != 0:
        wrong_numbers.extend([
            gold_num + 1,
            gold_num - 1,
            gold_num * 2,
            gold_num + 10,
        ])
    else:
        wrong_numbers.extend([1, -1, 2, 10])

    # Filter out the correct answer
    wrong_numbers = [n for n in wrong_numbers if normalize_answer(str(n)) != gold_norm]
    wrong = random.choice(wrong_numbers) if wrong_numbers else gold_num + 1

    # Format wrong number like gold
    if gold_num == int(gold_num) and wrong == int(wrong):
        wrong_str = str(int(wrong))
    else:
        wrong_str = str(wrong)

    # Replace in response
    if dataset == "gsm8k":
        # Replace after ####
        perturbed = re.sub(
            r'(####\s*)(.+?)(\s*$)',
            rf'\g<1>{wrong_str}\3',
            response,
            count=1,
            flags=re.MULTILINE,
        )
        if perturbed == response:
            # No #### found, append one
            perturbed = response + f"\n\n#### {wrong_str}"
    else:
        # MATH: replace boxed content
        if "\\boxed{" in response:
            # Replace last boxed
            idx = response.rfind("\\boxed{")
            start = idx + len("\\boxed{")
            depth = 1
            end = start
            while end < len(response) and depth > 0:
                if response[end] == "{":
                    depth += 1
                elif response[end] == "}":
                    depth -= 1
                end += 1
            perturbed = response[:idx] + f"\\boxed{{{wrong_str}}}" + response[end:]
        else:
            perturbed = response + f"\n\nThe answer is \\boxed{{{wrong_str}}}"

    return perturbed


def _perturb_text_answer(response: str, gold_answer: str) -> str:
    """Perturb a non-numeric answer by modifying it slightly."""
    # For LaTeX/text answers, just change the boxed content
    if "\\boxed{" in response:
        idx = response.rfind("\\boxed{")
        start = idx + len("\\boxed{")
        depth = 1
        end = start
        while end < len(response) and depth > 0:
            if response[end] == "{":
                depth += 1
            elif response[end] == "}":
                depth -= 1
            end += 1
        # Replace with a generic wrong answer
        perturbed = response[:idx] + "\\boxed{\\text{undefined}}" + response[end:]
        return perturbed
    return response + "\n\nThe answer is \\boxed{\\text{undefined}}"


def truncate_solution(response: str) -> str:
    """Truncate the solution to ~60% to simulate incomplete work."""
    lines = response.split("\n")
    keep = max(2, int(len(lines) * 0.6))
    truncated = "\n".join(lines[:keep])
    # Ensure no #### or boxed at the end (they'd be correct)
    truncated = re.sub(r'####\s*.+$', '', truncated, flags=re.MULTILINE).strip()
    if "\\boxed{" in truncated:
        # Remove the last boxed if it ended up in truncated portion
        idx = truncated.rfind("\\boxed{")
        truncated = truncated[:idx].strip()
    return truncated


def swap_final_answer(response: str, gold_answer: str, dataset: str) -> str:
    """Keep the reasoning but replace the final answer with a common mistake."""
    gold_norm = normalize_answer(gold_answer)
    try:
        gold_num = float(gold_norm)
        # Common arithmetic mistakes
        mistakes = [
            gold_num + random.choice([2, 3, 5]),
            gold_num - random.choice([2, 3, 5]),
            abs(gold_num) if gold_num < 0 else -gold_num,
        ]
        wrong = random.choice([m for m in mistakes if normalize_answer(str(m)) != gold_norm] or [gold_num + 7])
        wrong_str = str(int(wrong)) if wrong == int(wrong) else str(wrong)
    except ValueError:
        wrong_str = "0"

    return perturb_final_number(response, gold_answer, dataset).replace(
        # Already handled by perturb_final_number, but ensure we use a DIFFERENT wrong number
        gold_answer, wrong_str
    )


def create_perturbations(
    response: str,
    gold_answer: str,
    is_correct: bool,
    dataset: str,
    num_perturbations: int = 3,
) -> List[Tuple[str, float]]:
    """Create perturbed answers with known correctness.

    Returns list of (perturbed_answer, gold_score) tuples.
    """
    perturbations = []

    if is_correct:
        # Response is correct (gold_score=1.0). Create incorrect perturbations.
        # Perturbation 1: wrong final number
        p1 = perturb_final_number(response, gold_answer, dataset)
        perturbations.append((p1, 0.0))

        if num_perturbations >= 2:
            # Perturbation 2: truncated (incomplete)
            p2 = truncate_solution(response)
            perturbations.append((p2, 0.0))

        if num_perturbations >= 3:
            # Perturbation 3: different wrong number
            p3 = swap_final_answer(response, gold_answer, dataset)
            perturbations.append((p3, 0.0))
    else:
        # Response is incorrect. Create the correct answer artificially.
        # Use the gold answer text as a "correct" reference.
        if dataset == "gsm8k":
            correct_stub = f"The answer is #### {gold_answer}"
        else:
            correct_stub = f"The answer is \\boxed{{{gold_answer}}}"
        perturbations.append((correct_stub, 1.0))

        if num_perturbations >= 2:
            # Another wrong answer (truncated)
            p2 = truncate_solution(response)
            perturbations.append((p2, 0.0))

        if num_perturbations >= 3:
            # Yet another wrong answer variant
            p3 = perturb_final_number(response, gold_answer, dataset)
            perturbations.append((p3, 0.0))

    return perturbations


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_PROMPT = (
    "Solve this math problem step by step. Show your work clearly. "
    "End your answer with the final answer after '#### ' (for arithmetic) "
    "or inside \\boxed{{}} (for expressions).\n\n"
    "Question: {question}\n\nAnswer:"
)


async def generate_answer(client, question: str, max_tokens: int = 1024) -> str:
    """Generate one answer for a question."""
    prompt = ANSWER_PROMPT.format(question=question)
    response = await client.generate(prompt=prompt, max_tokens=max_tokens)
    return response.strip()


# ---------------------------------------------------------------------------
# Main precompute pipeline
# ---------------------------------------------------------------------------

async def precompute_verifiable(
    dataset: str,
    output_cache: str,
    model: str = "gpt-5.2-chat",
    use_azure: bool = True,
    limit: int = 0,
    max_tokens: int = 1024,
    num_perturbations: int = 3,
):
    """Run the precompute pipeline for verifiable domains.

    Args:
        dataset: "gsm8k" or "math".
        output_cache: Path to output JSONL cache.
        model: LLM model for answer generation.
        limit: Process only this many questions (0=all).
        max_tokens: Max tokens per generated answer.
        num_perturbations: Number of perturbations per question.
    """
    from ..llm.client import AzureOpenAIClient

    # Load dataset via adapter
    if dataset == "gsm8k":
        from ..data.adapters.gsm8k import GSM8KAdapter
        adapter = GSM8KAdapter()
    elif dataset == "math":
        from ..data.adapters.math_hendrycks import MATHAdapter
        adapter = MATHAdapter()
    elif dataset == "medqa":
        from ..data.adapters.medqa import MedQAAdapter
        adapter = MedQAAdapter()
    elif dataset == "medmcqa":
        from ..data.adapters.medmcqa import MedMCQAAdapter
        adapter = MedMCQAAdapter()
    else:
        raise ValueError(f"Unknown verifiable dataset: {dataset}")

    logger.info("Loading dataset '%s'...", dataset)
    items = adapter.load_raw()

    if limit > 0:
        items = items[:limit]
        logger.info("Limiting to %d questions (validation mode)", limit)

    logger.info("Loaded %d questions", len(items))
    logger.info("Model: %s | perturbations_per_q: %d", model, num_perturbations)

    # Load existing cache
    cache_path = Path(output_cache)
    existing: Dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing[entry["question_id"]] = entry
        logger.info("Existing cache: %d entries", len(existing))

    # Init client
    client = AzureOpenAIClient(model=model, use_azure=use_azure)

    # Process
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "correct": 0, "incorrect": 0, "skipped": 0}

    is_mcq = dataset in ("medqa", "medmcqa")

    with open(cache_path, "a", encoding="utf-8") as f_out:
        for i, item in enumerate(items):
            qid = item.get("question_id", f"{dataset}_{i}")

            if qid in existing:
                stats["skipped"] += 1
                if (i + 1) % 100 == 0:
                    logger.info("[%d/%d] %s — cached, skipping", i + 1, len(items), qid)
                continue

            question = item["question"]
            gold_answer = item.get("final_answer", "") or item.get("correct_text", "")

            if not gold_answer:
                logger.warning("[%d/%d] %s — no gold answer, skipping", i + 1, len(items), qid)
                continue

            if is_mcq:
                # MCQ: use the 4 options as answers directly
                options = item.get("options", {})
                answer_letter = item.get("answer_letter", "")
                answers = []
                gold_scores_list: List[float] = []
                for letter in sorted(options.keys()):
                    opt_text = options[letter]
                    if opt_text:
                        answers.append(f"{letter}. {opt_text}")
                        gold_scores_list.append(1.0 if letter == answer_letter else 0.0)

                if len(answers) < 2:
                    logger.warning("[%d/%d] %s — fewer than 2 options, skipping", i + 1, len(items), qid)
                    continue

                stats["total"] += 1
                stats["correct"] += 1

                entry = {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "dataset": dataset,
                    "answers": answers,
                    "gold_scores": gold_scores_list,
                    "original_correct": True,
                    "extracted_answer": answer_letter,
                    "answer_letter": answer_letter,
                    "subject": item.get("subject", ""),
                    "topic": item.get("topic", ""),
                }
            else:
                # Math: generate answer + perturbations
                try:
                    response = await generate_answer(client, question, max_tokens)
                except Exception as exc:
                    logger.error("[%d/%d] %s — generation failed: %s", i + 1, len(items), qid, exc)
                    continue

                is_correct, extracted = check_correct(response, gold_answer, dataset)
                stats["total"] += 1
                stats["correct" if is_correct else "incorrect"] += 1

                perturbations = create_perturbations(
                    response, gold_answer, is_correct, dataset, num_perturbations
                )

                answers = [response]
                gold_scores_list = [1.0 if is_correct else 0.0]

                for p_answer, p_score in perturbations:
                    answers.append(p_answer)
                    gold_scores_list.append(p_score)

                entry = {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "dataset": dataset,
                    "answers": answers,
                    "gold_scores": gold_scores_list,
                    "original_correct": is_correct,
                    "extracted_answer": extracted,
                    "level": item.get("level", ""),
                    "subject": item.get("subject", ""),
                }

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_out.flush()

            if (i + 1) % 10 == 0 or (i + 1) <= 5:
                logger.info(
                    "[%d/%d] %s — %d answers (%.0f correct)",
                    i + 1, len(items), qid,
                    len(answers),
                    sum(gold_scores_list),
                )

    logger.info(
        "Done. Total=%d, Correct=%d (%.1f%%), Incorrect=%d, Skipped=%d",
        stats["total"],
        stats["correct"],
        100 * stats["correct"] / max(stats["total"], 1),
        stats["incorrect"],
        stats["skipped"],
    )
    logger.info("Cache: %s", cache_path)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute answers + perturbations + gold scores for verifiable domains"
    )
    parser.add_argument("--dataset", default="gsm8k",
                        choices=["gsm8k", "math", "medqa", "medmcqa"],
                        help="Verifiable dataset to precompute")
    parser.add_argument("--output_cache", default=None,
                        help="Output cache path (default: data/cache/{dataset}_precompute.jsonl)")
    parser.add_argument("--model", default="gpt-5.2-chat",
                        help="LLM model for answer generation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N questions (0=all)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens per answer")
    parser.add_argument("--num_perturbations", type=int, default=3,
                        help="Number of perturbations per question")
    parser.add_argument("--no_azure", action="store_true",
                        help="Use OpenAI directly instead of Azure")

    args = parser.parse_args()

    output_cache = args.output_cache or f"data/cache/{args.dataset}_precompute.jsonl"

    asyncio.run(
        precompute_verifiable(
            dataset=args.dataset,
            output_cache=output_cache,
            model=args.model,
            use_azure=not args.no_azure,
            limit=args.limit,
            max_tokens=args.max_tokens,
            num_perturbations=args.num_perturbations,
        )
    )


if __name__ == "__main__":
    main()
