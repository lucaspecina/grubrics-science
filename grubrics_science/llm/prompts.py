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
    base_instruction = """Answer the following question clearly and completely, but be concise. 
Provide a complete answer with a clear conclusion. Do not cut off mid-sentence or leave the answer incomplete."""
    
    instructions = {
        "normal": base_instruction,
        "low_temp": """Answer the following question in a detailed and precise manner, including all necessary derivations and assumptions.
Be thorough but concise. Ensure your answer is complete with a clear conclusion.""",
        "high_temp": """Answer the following question creatively, exploring different perspectives and approaches.
Be concise while covering multiple angles. End with a clear summary or conclusion.""",
        "failure_mode_1": """Answer the following question but omit any derivations or mathematical steps. Just state conclusions.
Be brief and direct. Provide a complete answer with clear conclusions.""",
        "failure_mode_2": """Answer the following question but omit explicit assumptions or boundary conditions. Be vague about limitations.
Keep it concise. Provide a complete answer even if some details are omitted.""",
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

CRITICAL: Each rubric contains multiple ITEMS (each item starts with "Points: X, Item: Y"). You must evaluate EACH ITEM separately within each rubric.

For each ITEM in each rubric:
1. Identify the maximum points for that item (from "Points: X")
2. Evaluate how well the answer satisfies that specific item's criteria
3. Assign a score between 0.0 and 1.0 representing the fraction of points earned
4. Provide a detailed explanation (2-3 sentences) for WHY you assigned that score to that specific item

Scoring guidelines per item:
- 0.0 = Answer completely fails to meet this item's criteria
- 0.5 = Answer partially meets this item's criteria
- 1.0 = Answer fully meets this item's criteria

For items with sub-items (e.g., "(0.5pts) ..."), evaluate each sub-item and aggregate appropriately.

You must return a JSON object with the following structure:
{
  "answer_id": "a1",
  "rubric_evaluations": [
    {
      "rubric_id": "r1",
      "item_scores": [
        {
          "item_id": "item_1",
          "item_description": "Brief description of this item (e.g., 'Background Theory')",
          "max_points": 1.5,
          "score": 0.75,
          "notes": "Detailed explanation (2-3 sentences) explaining what specific aspects of the answer relate to THIS ITEM, why you assigned this score, and what is present or missing."
        },
        {
          "item_id": "item_2",
          "item_description": "Brief description of this item",
          "max_points": 0.5,
          "score": 0.50,
          "notes": "Detailed explanation for this item..."
        },
        ...
      ],
      "total_score": 0.65
    },
    {
      "rubric_id": "r2",
      "item_scores": [...],
      "total_score": 0.80
    },
    ...
  ]
}

IMPORTANT:
- Evaluate each item independently
- The total_score for each rubric should be a weighted average: sum(item_score * max_points) / sum(max_points)
- Provide detailed explanations for EACH item, not just the rubric overall
- Return ONLY valid JSON, no other text."""


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
    
    prompt += f"""Evaluate the answer against each rubric independently. For each rubric, identify all items (each starting with "Points: X, Item: Y") and evaluate each item separately.

Return detailed scores and explanations for EACH ITEM within each rubric.

Answer ID: {answer_id}
"""
    
    return prompt

