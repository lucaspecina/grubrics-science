"""Prompt templates for GRubrics Science."""

from typing import Optional, List


# ============================================================================
# ANSWER POLICY PROMPTS
# ============================================================================

def get_answer_policy_prompt(question: str, instruction_type: str = "normal") -> str:
    """
    Generate prompt for answer policy.
    
    Args:
        question: The question to answer
        instruction_type: Type of instruction ("normal", "low_temp", "high_temp", "failure_mode")
    
    Returns:
        Formatted prompt string
    """
    base_instruction = "Answer the following question clearly and completely."
    
    instructions = {
        "normal": base_instruction,
        "low_temp": "Answer the following question in a detailed and precise manner, including all necessary derivations and assumptions.",
        "high_temp": "Answer the following question creatively, exploring different perspectives and approaches.",
        "failure_mode_1": "Answer the following question but omit any derivations or mathematical steps. Just state conclusions.",
        "failure_mode_2": "Answer the following question but omit explicit assumptions or boundary conditions. Be vague about limitations.",
    }
    
    instruction = instructions.get(instruction_type, base_instruction)
    
    return f"""{instruction}

Question: {question}

Answer:"""


# ============================================================================
# GRUBRICS GENERATION PROMPTS
# ============================================================================

def get_grubrics_prompt(
    question: str,
    best_answer_excerpt: Optional[str] = None,
    worst_answer_excerpt: Optional[str] = None
) -> str:
    """
    Generate prompt for GRubrics to generate a scoring rubric.
    
    Args:
        question: The question to generate rubrics for
        best_answer_excerpt: Optional excerpt from best answer (for discriminative guidance)
        worst_answer_excerpt: Optional excerpt from worst answer (for discriminative guidance)
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert evaluator. Generate a scoring rubric for evaluating answers to the following research question.

Question: {question}

The rubric should:
- Be structured as a bullet list with clear criteria
- Include point allocations or scoring guidelines for each criterion
- Be discriminative: able to distinguish between high-quality and low-quality answers
- Focus on scientific rigor, clarity, completeness, and logical reasoning
- NOT reference or copy any existing rubrics you may have seen

"""
    
    if best_answer_excerpt or worst_answer_excerpt:
        prompt += "\n=== CONTEXT FOR DISCRIMINATIVE CRITERIA ===\n"
        prompt += "Below are excerpts from answers to help you understand what distinguishes good from poor responses.\n"
        prompt += "Use these to create criteria that can effectively separate quality levels.\n\n"
        
        if best_answer_excerpt:
            prompt += f"High-quality answer excerpt:\n{best_answer_excerpt[:300]}...\n\n"
        if worst_answer_excerpt:
            prompt += f"Low-quality answer excerpt:\n{worst_answer_excerpt[:300]}...\n\n"
        
        prompt += "Generate a rubric that can distinguish between answers like these.\n"
    
    prompt += "\nRubric:\n"
    
    return prompt


# ============================================================================
# JUDGE PROMPTS (BATCHED)
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. You will be given a question, an answer, and a list of rubrics.

Your task is to evaluate the answer against EACH rubric independently and provide scores.

For each rubric, follow it exactly as written and assign a score between 0.0 and 1.0, where:
- 0.0 = Answer completely fails to meet the criterion
- 0.5 = Answer partially meets the criterion
- 1.0 = Answer fully meets the criterion

You must return a JSON object with the following structure:
{
  "answer_id": "a1",
  "rubric_scores": [
    {"rubric_id": "r1", "score": 0.75, "notes": "Brief reason"},
    {"rubric_id": "r2", "score": 0.50, "notes": "Brief reason"},
    ...
  ]
}

Return ONLY valid JSON, no other text."""


def get_judge_prompt(
    question: str,
    answer: str,
    rubrics: List[str],
    answer_id: str = "a1"
) -> str:
    """
    Generate prompt for Judge to evaluate an answer against multiple rubrics.
    
    Args:
        question: The original question
        answer: The answer to evaluate
        rubrics: List of rubric strings (each will be assigned r1, r2, ...)
        answer_id: Identifier for this answer
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""Question: {question}

Answer to evaluate:
{answer}

Rubrics to evaluate against:

"""
    
    for i, rubric in enumerate(rubrics, 1):
        prompt += f"--- Rubric r{i} ---\n{rubric}\n\n"
    
    prompt += f"""Evaluate the answer against each rubric independently and return scores in JSON format.

Answer ID: {answer_id}
"""
    
    return prompt

