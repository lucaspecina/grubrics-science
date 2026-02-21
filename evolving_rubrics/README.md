# DR-Tulu Evolving Rubrics (Legacy)

> **Nota:** Este paquete es un prototipo inicial que exploraba la evolucion iterativa de rubricas con LLMs. El sistema principal del proyecto ahora vive en `grubrics_science/` y usa un enfoque diferente basado en RL (GRPO) con functional alignment. Este paquete se conserva como referencia historica.

## Relacion con GRubrics

Este paquete implementaba un loop de evolucion heuristica:
1. Generar rubricas iniciales con un LLM
2. Generar respuestas de modelos
3. Evaluar respuestas con las rubricas
4. Generar rubricas adaptativas basadas en las diferencias entre buenas/malas respuestas
5. Repetir

El sistema GRubrics (`grubrics_science/`) reemplaza este enfoque con entrenamiento RL:
- En vez de iterar heuristicamente, entrena un modelo (Qwen3-8B) con GRPO para generar rubricas
- La señal de reward es **functional alignment** (Spearman correlation con gold scores)
- Soporta curriculum learning (verificable → abierto)
- Escala a miles de preguntas via precompute + cache

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

```bash
pip install openai python-dotenv
```

## Configuration

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

```python
from evolving_rubrics import evolve_rubrics_for_example

result = await evolve_rubrics_for_example(
    question="Your question here...",
    num_iterations=3,
    num_responses_per_iteration=4
)
```

## Module Descriptions

- **`config.py`** — Environment variables, API configuration, client initialization
- **`prompts.py`** — All prompt templates (rubric generation, evaluation, response generation)
- **`helpers.py`** — JSON extraction from LLM responses, async LLM calls
- **`rubric_generation.py`** — Initial and adaptive rubric generation
- **`response_generation.py`** — Model response generation with varied instructions
- **`evaluation.py`** — Response evaluation against rubric criteria
- **`evolution.py`** — Main evolution workflow orchestration
- **`output.py`** — Output and history management
