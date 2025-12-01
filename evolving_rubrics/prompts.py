"""
Prompts module for DR-Tulu Evolving Rubrics.

Contains all prompt templates used throughout the package.
"""

# ============================================================================
# ORIGINAL RUBRIC GENERATION PROMPT
# ============================================================================

def get_original_rubrics_prompt(question: str) -> str:
    """
    Generate prompt for creating initial rubrics from a question.
    
    Args:
        question: The question to generate rubrics for
    
    Returns:
        Formatted prompt string
    """
    return f"""You are an expert in educational evaluation. Generate evaluation rubrics for the following question.

Question: {question}

Generate 2-4 rubrics that cover the essential aspects for evaluating a response to this question.
Each rubric must have:
- title: A short and descriptive title
- description: A detailed description of what is being evaluated
- weight: A weight (use 1.0 for all)

Respond ONLY with valid JSON in this format:
{{
  "query": "{question}",
  "rubrics": [
    {{
      "title": "Rubric title",
      "description": "Detailed description of what is being evaluated",
      "weight": 1.0
    }}
  ]
}}"""


# ============================================================================
# ADAPTIVE RUBRIC GENERATION PROMPT
# ============================================================================

ADAPTIVE_RUBRIC_GENERATION_PROMPT = """You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality answers. Capture subtle quality differences that existing rubrics miss.

## Output Components
- **Description**: Detailed, specific description of what makes a response excellent/problematic
- **Title**: Concise abstract label (general, not question-specific)

## Categories
1. **Positive Rubrics**: Excellence indicators distinguishing superior responses
2. **Negative Rubrics**: Critical flaws definitively degrading quality

## Core Guidelines

### 1. Discriminative Power
- Focus ONLY on criteria meaningfully separating quality levels
- Each rubric must distinguish between otherwise similar responses
- Exclude generic criteria applying equally to all responses

### 2. Novelty & Non-Redundancy
With existing/ground truth rubrics:
- Never duplicate overlapping rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing ones are broad
- Return empty lists if existing rubrics are comprehensive

### 3. Avoid Mirror Rubrics
Never create positive/negative versions of same criterion:
- ❌ "Provides clear explanations" + "Lacks clear explanations"
- ✅ Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- Response penalized if it exhibits ANY negative rubric behavior
- Focus on active mistakes vs missing features

## Selection Strategy
### Quantity: 1-5 total rubrics (fewer high-quality > many generic)

## Output Format
```json
{
  "question": "<original question verbatim>",
  "positive_rubrics": [
    {"description": "<detailed excellence description>", "title": "<abstract label>"}
  ],
  "negative_rubrics": [
    {"description": "<detailed failure description>", "title": "<abstract label>"}
  ]
}
```

## Critical Reminders
- Each rubric must distinguish between actual provided responses
- Exclude rubrics applying equally to all responses
- Prefer empty lists over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 2 excellent rubrics > 5 mediocre ones

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences."""


def get_adaptive_rubrics_prompt(
    question: str,
    responses: list,
    existing_rubrics: list = None,
    good_responses: list = None,
    bad_responses: list = None
) -> str:
    """
    Build the complete prompt for generating adaptive rubrics.
    
    This follows DR-Tulu: responses are first evaluated with current rubrics (Judge),
    then adaptive rubrics are generated based on what distinguishes good from bad.
    
    Args:
        question: The original question
        responses: List of all model responses to analyze
        existing_rubrics: Optional existing rubrics to avoid duplicating
        good_responses: List of responses identified as "good" (high scores from Judge)
        bad_responses: List of responses identified as "bad" (low scores from Judge)
    
    Returns:
        Complete prompt string
    """
    import json
    
    prompt_suffix = f"Question: {question}\n\n"
    
    # Include information about good vs bad responses if available
    if good_responses and bad_responses:
        prompt_suffix += "=== HIGH-QUALITY RESPONSES (Good examples) ===\n"
        for i, response in enumerate(good_responses, 1):
            prompt_suffix += f"Good Response {i}:\n{response}\n\n"
        
        prompt_suffix += "\n=== LOW-QUALITY RESPONSES (Bad examples) ===\n"
        for i, response in enumerate(bad_responses, 1):
            prompt_suffix += f"Bad Response {i}:\n{response}\n\n"
        
        prompt_suffix += "\n=== ALL RESPONSES (for context) ===\n"
        for i, response in enumerate(responses, 1):
            prompt_suffix += f"Response {i}:\n{response}\n\n"
    else:
        # Fallback: show all responses without labels
        prompt_suffix += "Responses:\n"
        for i, response in enumerate(responses, 1):
            prompt_suffix += f"Response {i}:\n{response}\n\n"
    
    if existing_rubrics:
        prompt_suffix += f"\n\nExisting Rubrics:\n{json.dumps(existing_rubrics, indent=2)}"
    
    return ADAPTIVE_RUBRIC_GENERATION_PROMPT + "\n\n" + prompt_suffix


# ============================================================================
# RESPONSE GENERATION PROMPTS
# ============================================================================

RESPONSE_GENERATION_INSTRUCTIONS = [
    "Answer in a complete and detailed manner, including specific examples.",
    "Answer concisely but informatively.",
    "Answer in a very detailed and technical manner, including advanced concepts.",
    "Answer simply and directly, suitable for beginners.",
    "Answer in a balanced manner, balancing depth and clarity.",
]


def get_response_generation_prompt(question: str, instruction: str) -> str:
    """
    Generate prompt for creating a model response.
    
    Args:
        question: The question to answer
        instruction: Specific instruction for response style
    
    Returns:
        Formatted prompt string
    """
    return f"""{instruction}

Question: {question}

Answer the question clearly and completely."""


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

RUBRIC_EVALUATION_SYSTEM_PROMPT = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant. You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion. Judge only the specified aspect(s), not any other qualities of the answer. Output JSON in the format: {"score": x}."""


def get_rubric_evaluation_prompt(question: str, response: str, rubric_description: str) -> str:
    """
    Generate prompt for evaluating a response against a rubric criterion.
    
    Args:
        question: The original question
        response: The response to evaluate
        rubric_description: Description of the criterion to evaluate
    
    Returns:
        Formatted user prompt string
    """
    return f"""<question>{question}</question>
<response>{response}</response>
<criterion>{rubric_description}</criterion>"""

