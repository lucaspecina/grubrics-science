# GRubrics Science

Entrenar (con RL) un modelo llamado **GRubrics** cuyo UNICO trabajo es **generar rúbricas de evaluación** para preguntas abiertas de investigación científica.

## La idea central

Dado un examen de ciencia con preguntas abiertas, queremos un modelo que genere automáticamente las rúbricas de corrección. ¿Cómo sabemos si una rúbrica generada es buena? Usamos **functional alignment**: una rúbrica es buena si, al usarla para evaluar múltiples respuestas, produce un **ranking similar** al que produce la rúbrica golden (la de referencia humana).

En otras palabras: no comparamos el texto de la rúbrica generada vs la golden. Comparamos los **scores que producen** al aplicarlas sobre las mismas respuestas.

## Arquitectura (3 actores)

```
                    ┌─────────────────┐
                    │   GRubrics       │  <-- UNICO modelo que se entrena (Qwen)
                    │   (genera        │
                    │    rúbricas)     │
                    └────────┬────────┘
                             │ rúbrica generada
                             ▼
┌──────────────┐     ┌──────────────┐
│ Answer Policy│     │    Judge     │  <-- Ambos FIJOS (Azure OpenAI, nunca se entrenan)
│ (genera      │────>│ (evalúa      │
│  respuestas) │     │  respuestas  │
└──────────────┘     │  con rúbrica)│
                     └──────┬───────┘
                            │ scores
                            ▼
                     ┌──────────────┐
                     │   Reward     │  = alignment(scores_rúbrica, scores_golden) - λ·length
                     └──────────────┘
```

- **Answer Policy**: Modelo fijo (GPT) que genera respuestas diversas a cada pregunta.
- **Judge**: Modelo fijo (GPT) que toma (pregunta, respuesta, rúbrica) y devuelve un score item-by-item.
- **GRubrics**: Modelo entrenable (Qwen) que genera rúbricas. Se entrena con REINFORCE.

## Las 2 fases: Precompute y Train

### Precompute — "Preparar los datos"

El entrenamiento necesita, para cada pregunta, un conjunto de respuestas ya evaluadas con la rúbrica golden. Como generar respuestas y evaluarlas con el Judge es caro (llamadas a API), esto se hace **una sola vez** y se cachea.

**¿Qué hace?**
1. Para cada pregunta del dataset, genera **K respuestas diversas** usando el Answer Policy (variando temperatura e instrucciones).
2. Evalúa cada respuesta con la **rúbrica golden** usando el Judge.
3. Guarda todo en `grubrics_science/data/cache/precompute_cache.jsonl`.

**¿Por qué separarlo?** Porque es costoso y no cambia entre runs de entrenamiento. Lo corrés una vez y listo.

### Train — "Entrenar GRubrics con RL"

Usa los datos cacheados para entrenar el modelo con REINFORCE (policy gradient).

**¿Qué hace en cada step?**
1. Toma una pregunta y sus K respuestas+gold_scores del cache.
2. GRubrics **genera M rúbricas** (sampling con temperatura).
3. El Judge **evalúa las K respuestas** con cada rúbrica generada → `score_matrix[K][M]`.
4. Para cada rúbrica, calcula el **reward**:
   - `alignment` = correlación de Spearman entre los scores que da esta rúbrica y los gold_scores.
   - `length_penalty` = penalización por rúbricas muy largas.
   - `reward = alignment - λ · length_penalty`
5. Calcula **advantages** (reward de cada rúbrica - promedio) para saber cuáles fueron mejores.
6. **REINFORCE**: actualiza los pesos de GRubrics para que genere más rúbricas parecidas a las que tuvieron alto reward.

## Project Structure

```
grubrics_science/
  configs/          # Configuración (YAML)
  llm/              # Clientes LLM (Azure OpenAI, Qwen) y prompts
  tasks/            # Loaders de datasets (FrontierScience)
  judge/            # Judge wrapper (evalúa respuestas contra rúbricas)
  rewards/          # Cálculo de reward (alignment metrics + length penalty)
  rl/               # Training loop y model wrapper
  utils/            # Utilidades (IO, logging, seeding)
  data/             # Cache de precompute
```

## Quick Start

### 1. Instalar dependencias

```bash
# PyTorch con CUDA (primero)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Resto de dependencias
pip install -r requirements.txt
```

### 2. Configurar API keys

Crear `.env` en la raíz del repo:
```
USE_AZURE_OPENAI=true
AZURE_API_KEY=tu-key
AZURE_API_BASE=https://tu-endpoint.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview
RUBRIC_JUDGE_MODEL=tu-deployment-name
RUBRIC_GENERATION_MODEL=tu-deployment-name
```

### 3. Ejecutar

La forma más simple es usar `test_grubrics.py`, donde controlás todo con flags al inicio del archivo:

```python
RUN_PRECOMPUTE = True   # Generar respuestas y gold scores
RUN_TRAIN = True        # Entrenar GRubrics
```

```bash
python test_grubrics.py
```

O con debug en Cursor/VS Code: seleccionar **"GRubrics: Test"** y F5.

## Configuración

Parámetros clave (editables en `test_grubrics.py` o `configs/default.yaml`):

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `K` | 8 | Respuestas generadas por pregunta (precompute) |
| `k_train` | 4 | Subset de respuestas usadas en cada step de training |
| `M` | 6 | Rúbricas generadas por GRubrics en cada step |
| `alignment_metric` | spearman | Métrica de alignment (spearman, pearson, pairwise) |
| `lambda_len` | 0.01 | Coeficiente de penalización por largo de rúbrica |

## Dataset

FrontierScience-Research en `data/frontierscience-research/test.jsonl`. Cada registro tiene:
- `problem`: pregunta de investigación científica
- `answer`: rúbrica golden (referencia humana)
