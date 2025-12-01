"""
DR-Tulu Evolving Rubrics Package

A modular package for evolving rubrics through iterative refinement.
"""

from .evolution import evolve_rubrics_for_example
from .rubric_generation import (
    generate_original_rubrics,
    generate_adaptive_rubrics,
    update_ground_truth
)
from .response_generation import generate_model_responses
from .evaluation import (
    evaluate_rubric,
    evaluate_complete_response
)
from .helpers import (
    extract_json_from_response,
    call_llm
)
from .config import (
    get_client,
    USE_AZURE,
    RUBRIC_GENERATION_MODEL,
    RUBRIC_JUDGE_MODEL
)

__all__ = [
    # Main evolution function
    'evolve_rubrics_for_example',
    
    # Rubric generation
    'generate_original_rubrics',
    'generate_model_responses',
    'generate_adaptive_rubrics',
    'update_ground_truth',
    
    # Evaluation
    'evaluate_rubric',
    'evaluate_complete_response',
    
    # Helpers
    'extract_json_from_response',
    'call_llm',
    
    # Configuration
    'get_client',
    'USE_AZURE',
    'RUBRIC_GENERATION_MODEL',
    'RUBRIC_JUDGE_MODEL',
]

__version__ = '1.0.0'

