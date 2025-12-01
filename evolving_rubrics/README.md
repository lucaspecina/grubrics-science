# DR-Tulu Evolving Rubrics

A modular package for evolving rubrics through iterative refinement using LLMs.

## Structure

```
evolving_rubrics/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration and client initialization
├── prompts.py           # All prompt templates
├── helpers.py           # Utility functions (JSON extraction, LLM calls)
├── rubric_generation.py # Rubric generation functions (original & adaptive)
├── response_generation.py # Model response generation functions
├── evaluation.py        # Response evaluation functions
├── evolution.py         # Main evolution workflow
└── output.py            # Output and history management
```

## Installation

Make sure you have the required dependencies:

```bash
pip install openai python-dotenv
```

## Configuration

Set environment variables (or use a `.env` file):

```bash
# For Azure OpenAI
USE_AZURE_OPENAI=true
AZURE_API_BASE=https://your-endpoint.openai.azure.com/
AZURE_API_KEY=your-key
AZURE_API_VERSION=2024-02-15-preview

# For standard OpenAI
OPENAI_API_KEY=your-key

# Model configuration
RUBRIC_GENERATION_MODEL=gpt-4o-mini
RUBRIC_JUDGE_MODEL=gpt-4o-mini
```

## Usage

### Basic Usage

```python
from evolving_rubrics import evolve_rubrics_for_example

# Evolve rubrics for a single example with multiple iterations
result = await evolve_rubrics_for_example(
    question="Your question here...",
    num_iterations=3,  # Iterate 3 times for the same example
    num_responses_per_iteration=4  # Generate 4 responses per iteration
)

# Access results
print(result['initial_ground_truth'])
print(result['iterations'])  # List with results from each iteration
print(result['final_ground_truth'])
```

### Advanced Usage

You can also use individual functions:

```python
from evolving_rubrics import (
    generate_original_rubrics,
    generate_model_responses,
    generate_adaptive_rubrics,
    evaluate_complete_response
)

# Generate initial rubrics
initial_rubrics = await generate_original_rubrics("Your question...")

# Generate model responses
responses = await generate_model_responses("Your question...", num_responses=4)

# Generate adaptive rubrics
adaptive = await generate_adaptive_rubrics(
    "Your question...",
    responses,
    existing_rubrics=initial_rubrics['rubrics']
)

# Evaluate responses
evaluation = await evaluate_complete_response(
    responses[0],
    initial_rubrics
)
```

## Output and History

The evolution process automatically saves a complete history to a JSON file in the `outputs/` directory. This file contains:

- **Initial rubrics**: Starting rubrics for the question
- **Each iteration**:
  - All responses generated
  - Scores from Judge evaluation
  - Good vs bad response classification
  - Adaptive rubrics generated
  - Rubrics before and after the iteration
- **Final rubrics**: Complete rubric set after all iterations
- **Rubrics evolution**: Easy-to-compare rubric sets across iterations

### Loading and Analyzing Output

```python
from evolving_rubrics import load_evolution_history, compare_rubrics_across_iterations

# Load a saved history file
history = load_evolution_history("outputs/evolution_history_20250112_101530.json")

# Compare rubrics across iterations
comparison = compare_rubrics_across_iterations(history)

# Access specific data
print(f"Initial rubrics: {history['initial_rubrics']['total_count']}")
print(f"Final rubrics: {history['final_rubrics']['total_count']}")

# See rubrics evolution
for iter_key, iter_data in history['rubrics_evolution'].items():
    print(f"{iter_key}: {iter_data['rubrics']['total_count']} rubrics")
```

See `example_analyze_output.py` for a complete example of analyzing evolution history files.

## Module Descriptions

### `config.py`
Handles environment variables, API configuration, and client initialization for OpenAI/Azure OpenAI.

### `prompts.py`
Contains all prompt templates used throughout the package:
- Original rubric generation prompts
- Adaptive rubric generation prompts
- Response generation prompts
- Evaluation prompts

### `helpers.py`
Utility functions for JSON extraction from LLM responses and asynchronous LLM calls.

### `rubric_generation.py`
Functions for:
- Generating initial rubrics from questions
- Generating adaptive rubrics based on response differences
- Updating ground truth with new rubrics

### `response_generation.py`
Functions for:
- Generating model responses with varied instructions

### `evaluation.py`
Functions for:
- Evaluating responses against individual rubric criteria
- Evaluating complete responses against all rubrics
- Calculating weighted reward scores

### `evolution.py`
Main workflow function that orchestrates the entire evolution process through multiple iterations.

## Example Workflow

1. **Initial Rubrics**: Generate base rubrics for a question
2. **Iteration Loop** (repeats `num_iterations` times):
   - Generate model responses
   - Analyze differences and generate adaptive rubrics
   - Update ground truth with new rubrics
   - Evaluate all responses with updated rubrics
3. **Final Results**: Return complete evolution history

