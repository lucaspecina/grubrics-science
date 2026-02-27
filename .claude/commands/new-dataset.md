Guía para agregar un nuevo dataset al sistema. Usar cuando se quiere incorporar una nueva fuente de datos para training o evaluación.

## Decidir el tipo de dataset

**¿El dataset tiene respuestas verificables (MCQ, True/False, código, matemática)?**
→ **Verifiable**: gold_scores son programáticos [1.0, 0.0, ..., 0.0]

**¿El dataset tiene rúbricas escritas por humanos?**
→ **Open**: gold_scores vienen del Judge evaluando respuestas pre-generadas con la rúbrica humana

**¿El dataset tiene preguntas abiertas sin rúbricas?**
→ No compatible directamente. Necesitaría un proceso separado para crear rúbricas de referencia.

## Paso 1 — Crear el adapter

Crear `grubrics_science/data/adapters/nombre_dataset.py`:

```python
from grubrics_science.data.base import DatasetAdapter
from datasets import load_dataset

class NombreDatasetAdapter(DatasetAdapter):
    """
    Descripción del dataset: qué es, fuente, tamaño.
    """

    # CAMPOS REQUERIDOS en cada row del output:
    # - question: str — el prompt/pregunta
    # - golden_rubric: str — rúbrica de referencia (para open) o None (para verifiable)
    # - data_source: str — nombre del dataset (ej: "nombre_dataset")
    # - answers: list[str] — respuestas pre-generadas (si las hay)
    # - gold_scores: list[float] | None — si están precomputadas

    def load(self, split="train", limit=None):
        dataset = load_dataset("org/nombre-dataset", split=split)
        if limit:
            dataset = dataset.select(range(limit))
        return dataset

    def to_verl_format(self, example):
        """Convierte un ejemplo al formato estándar del sistema."""
        return {
            "question": example["question"],
            "golden_rubric": example.get("rubric", None),
            "data_source": "nombre_dataset",
            "answers": example.get("answers", []),
            "gold_scores": None,  # se llena en precompute
        }

    def build_prompt(self, question, rubric=None):
        """Construye el prompt para el generador de rúbricas."""
        # Usar el prompt base de la clase padre o personalizar
        return super().build_prompt(question, rubric)
```

Convenciones de nombres:
- Archivo: `nombre_dataset.py` (snake_case)
- Clase: `NombreDatasetAdapter` (PascalCase + Adapter)
- `data_source`: exactamente el mismo string que se usa en el registry y en los configs

## Paso 2 — Registrar en el registry

Editar `grubrics_science/data/adapters/__init__.py`:

```python
from .nombre_dataset import NombreDatasetAdapter

ADAPTER_REGISTRY = {
    # ... adapters existentes ...
    "nombre_dataset": NombreDatasetAdapter,
}
```

## Paso 3 — Agregar preset en configs

Editar `configs/training_presets.yaml` para agregar el dataset a presets existentes o crear uno nuevo:

```yaml
presets:
  # ... presets existentes ...
  nombre_dataset_only:
    description: "Solo nombre_dataset"
    datasets:
      - name: nombre_dataset
        weight: 1.0
```

## Paso 4 — Implementar precompute

**Para verifiable**: agregar el dataset a `grubrics_science/data/precompute_verifiable.py`:

```python
# En el dict de loaders:
"nombre_dataset": lambda: NombreDatasetAdapter().load()
```

**Para open**: crear `grubrics_science/data/precompute_nombre.py` siguiendo el patrón de `precompute_healthbench.py`. Los pasos son:
1. Cargar el dataset
2. Para cada pregunta: llamar al Judge con las respuestas pre-generadas + rúbrica golden
3. Guardar en `data/cache/nombre_dataset_precompute.jsonl`

## Paso 5 — Escribir tests

Crear `tests/test_nombre_dataset.py` siguiendo el patrón de `tests/test_healthbench.py`:
- Test que el adapter carga y devuelve el formato correcto
- Test que los campos requeridos están presentes
- Test que el holdout split funciona
- Test del formato veRL (columnas JSON serializadas correctamente)

```bash
pytest tests/test_nombre_dataset.py -v
```

## Paso 6 — Verificar integración

```bash
# Descargar el dataset
python scripts/download_datasets.py --only nombre_dataset

# Mini precompute
python -m grubrics_science.data.precompute_nombre --limit 5

# Verificar que el parquet se genera
python -m grubrics_science.data.prepare single \
    --dataset nombre_dataset --output_dir data/processed

# Correr todos los tests
pytest tests/ -v
```

## Checklist final

- [ ] Adapter implementado con todos los campos requeridos
- [ ] Registrado en `adapters/__init__.py`
- [ ] Preset en `configs/training_presets.yaml`
- [ ] Precompute implementado
- [ ] Tests escritos y pasando
- [ ] Parquet generado correctamente
- [ ] Documentar en `docs/experiment-log.md`: nuevo dataset, su tamaño, tipo de señal
