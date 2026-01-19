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
    prompt = f"""You are a RUBRIC WRITER.

DEFINITIONS
- QUESTION: the task prompt that future answers will respond to.
- ANSWER: a candidate response to the QUESTION (NOT provided now).
- RUBRIC: a list of scored items used later to grade an ANSWER to the QUESTION.

PIPELINE
Input you receive now: QUESTION only.
Output you must produce now: RUBRIC only.
There is NO ANSWER provided. Do not try to evaluate anything. Do not mention any specific answer content not implied by the QUESTION.

RUBRIC PURPOSE
A grader will later take (QUESTION, ANSWER, RUBRIC) and score the ANSWER by checking each rubric item and summing points.


WHAT A RUBRIC IS (context)
A rubric is a **set of scoring items** that represent the important, weighted properties of a good answer to a question.
Later, a human or model grader will:
1) read an answer,
2) go item-by-item through the rubric,
3) decide whether each item is satisfied (fully/partially/not),
4) sum the points to obtain a final score.

So your rubric must be:
- **actionable** (each item can be judged from the answer text),
- **weighted** (points reflect importance),
- **discriminative** (separates great vs mediocre vs wrong),
- and allow **partial credit** where appropriate.

---

PRIMARY REQUIREMENT (match dataset format)
- Output ONLY rubric lines.
- Each line must start exactly with:
  `Points: <number>, Item: <text>`
- The sum of all Points must be exactly **10.0**.
- Points may be fractional (e.g., 0.25, 0.5, 0.75, 1.0, 1.5, 2.0).
- Keep each Item short and checkable.

OPTIONAL (allowed) structure used in the dataset
- If an item needs internal splits, you may add: `Assign points as follows:` and then sub-bullets with point splits.
- If the question has multiple subquestions, include “Question 1: … / Question 2: …” inside the Item text.

CONTENT GUIDANCE (soft, not strict)
- Prefer grading **invariants** (key results, key intermediate quantities/claims, key constraints).
- Do NOT require a single solution path: accept alternative valid methods if they reach the same essential conclusions.
- Include at least one item for: core correctness, key reasoning, handling constraints/units (if relevant), and clarity/coherence.

---

ONE-SHOT FORMAT EXAMPLE (style only; not this topic) -> pair (QUESTION, RUBRIC)
QUESTION (EXAMPLE):
Context: A sodium adduct of the unknown compound displayed a precursor mass of 846.56 and two major product ions of 363.27 and 337.27 in positive mode.
Question: It was later found that this compound produces a color with LipidTox stain in a concentration-dependent manner. Surprisingly, none of the products corresponded to a diacylglycerol or diacylglycerol derivative of any composition.
What is the exact name (either common or IUPAC) of this compound, and shows steps of how you derive an answer. Additionally, identify which specific sub lipid class this belongs to and derive which exact diacylglycerol (or its derivative) the researcher would have looked for, including its mass. Be sure to identify what the product ions are specifically.
Think step by step and solve the problem below. In your answer, you should include all intermediate derivations, formulas, important steps, and justifications for how you arrived at your answer. Be as detailed as possible in your response.

RUBRIC (EXAMPLE):
Points: 1.0, Item: The answer clearly classified this unknown lipid as a type of phospholipid
Points: 1.0, Item: The answer clearly derives the mass of DAG as a number between 620-630
Points: 1.0, Item: The answer clearly identifies the 337.3 as MAG 18:2 derivative
Points: 1.0, Item: The answer clearly identifies the 363.3 as MAG 20:3 derivative
Points: 1.0, Item: The answer clearly identifies the DAG as a DAG with 18:2 and 20:3 fatty acid chains
Points: 1.0, Item: The answer clearly notes that the mass of sodium will be about 23 Da
Points: 1.0, Item: The answer clearly states that lipidtox stains both neutral lipids and phospholipids
Points: 1.0, Item: The answer clearly states that the neutral mass of the unknown compound is about 823.6 Da
Points: 2.0, Item: The answer specifically states that the lipid is BMP 18:2\/20:3

[END OF ONE-SHOT FORMAT EXAMPLE]

---

"""
    # TODO: revisar si esto va o no...
    if best_answer_excerpt or worst_answer_excerpt:
        prompt += "\n=== CONTEXT FOR DISCRIMINATIVE CRITERIA ===\n"
        prompt += "Below are excerpts from answers (to the question) to help you understand what distinguishes good from poor answers.\n"
        prompt += "Use these to create rubrics that can effectively separate quality levels of answers.\n\n"
        
        if best_answer_excerpt:
            prompt += f"High-quality answer excerpt:\n{best_answer_excerpt}...\n\n"
        if worst_answer_excerpt:
            prompt += f"Low-quality answer excerpt:\n{worst_answer_excerpt}...\n\n"
        
        prompt += "Create a rubric that can distinguish between answers like these.\n"
    
    prompt += f"""\nNow produce the rubric for the QUESTION below, strictly in the required format. Do not include any other text.

QUESTION:
{question}

[END OF QUESTION] 
---

YOUR RUBRIC:
"""
    
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

