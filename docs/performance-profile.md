# GRubrics — Performance Profile

Referencia viva de profiling, bottlenecks y optimizaciones del pipeline de training.
**Actualizar cada vez que se obtienen datos nuevos de rendimiento.**

Última actualización: 2026-03-19

---

## 1. Resumen ejecutivo

El pipeline GRPO tiene 3 fases por step: **rollout** (vLLM genera respuestas), **reward** (Judge API evalúa rúbricas), **update** (policy gradient + sync weights). El profiling a batch=8 en 1×H100 NVL muestra:

- **GPU domina sobre Judge**: el compute GPU es ~75% del step time; el reward (Judge API) se computa en paralelo y NO está en el critical path.
- **VRAM subutilizada**: 33/96 GB (35%). Hay headroom para subir batch size y micro-batch sizes.
- **Checkpoint save es costoso**: 122s por save (3.7x el step time). Mantener save_freq alto.
- **Startup es costo fijo**: 275s (~4.6 min). Se amortiza en runs largos.

---

## 2. Anatomía de un step GRPO

```
┌─────────── STEP (steady state: ~32.5s @ batch=8) ──────────┐
│                                                              │
│  gen (vLLM rollout)     ████████████████  11.4s  (35%)      │
│  update_actor (grad)    █████████████     10.4s  (32%)      │
│  update_weights (sync)  ██████████         8.0s  (25%)      │
│  old_log_prob           ███                2.7s   (8%)      │
│                                                              │
│  reward (Judge API)     ░░░░░░░░░          8-10s (PARALELO) │
│  (no bloquea — Ray workers async, termina antes que GPU)    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Componentes detallados

| Componente | Qué hace | Steady-state (batch=8) | Warmup (step 1) |
|-----------|----------|----------------------|-----------------|
| `gen` | vLLM genera K=6 respuestas por prompt | 11.4s | 21.6s (CUDA compile) |
| `update_actor` | Backprop + policy gradient update (FSDP) | 10.4s | 14.4s |
| `update_weights` | Sync pesos FSDP → vLLM (I/O GPU↔CPU) | 8.0s | 8.4s |
| `old_log_prob` | Calcula log probs de referencia | 2.7s | 6.0s |
| `reward` | Judge API evalúa rúbricas (async, no bloquea) | 8-10s wall | 9.8s |
| `save_checkpoint` | 3 formatos: FSDP + HF + LoRA adapter | 122s | — |

---

## 3. Timeline completo de un run

### Startup (275s = costo fijo)

```
t=0s     Python + imports
t=5s     Ray init
t=5-178s Config parsing + model setup (FSDP, WorkerDict)
         └─ Loading checkpoint shards: ~1.2s (5 shards, Qwen3-8B)
t=178s   vLLM spawn + init (~11s)
t=189s   CUDA graphs capture: ~18s (102 mixed + 102 decode)
t=207s   LoRA + AgentLoopWorkers + wandb init
t=275s   Primera inferencia → Step 1 comienza
```

**Observación**: la mayor parte del startup (~173s) es config parsing + model setup. Costo fijo ineludible.

### Training (steady state)

```
step 1: 50.4s  (warmup — gen tarda 2x por CUDA compilation)
step 2: 32.7s  ← steady state
step 3: 32.3s
step 4: 32.6s
step 5: 154.6s (incluye checkpoint save de 121.9s)
```

### Proyección por duración de run

| Steps | Training | Startup | Saves (freq=50) | Total | GPU cost |
|-------|----------|---------|-----------------|-------|----------|
| 50 | 27 min | 4.6 min | 1 × 2 min | ~34 min | ~$4 |
| 100 | 54 min | 4.6 min | 2 × 2 min | ~63 min | ~$7 |
| 200 | 108 min | 4.6 min | 4 × 2 min | ~121 min | ~$14 |

*Nota: proyección con batch=8. Batch=24 escalará tiempos de step.*

---

## 4. Recursos

### VRAM (GPU Memory)

| Métrica | Valor | % de 95.8 GB |
|---------|-------|-------------|
| Allocated (pico) | 33.2 GB | 35% |
| Reserved (pico) | 33.3 GB | 35% |
| **Libre** | **62.5 GB** | **65%** |

**Conclusión**: headroom enorme. Se puede subir `gpu_memory_utilization` (actualmente 0.5) y micro-batch sizes.

### CPU RAM

| Métrica | Valor | % de 320 GB |
|---------|-------|-------------|
| Usado (pico) | 56.4 GB | 18% |
| **Libre** | **263.6 GB** | **82%** |

### Throughput

| Métrica | Step 1 (warmup) | Steady state |
|---------|----------------|-------------|
| tokens/s | 10.2 | 15.5 |
| ms/token (gen) | 58.6 | 31.0 |
| ms/token (update) | 28.1 | 20.8 |

---

## 5. Reward (Judge API)

### Métricas por step (batch=8, concurrent=10)

| Step | wall | calls/worker | api_avg | api_max | sem_wait |
|------|------|-------------|---------|---------|----------|
| 1 | 9.8s | 11 | 6.2s | 8.1s | 0.37s |
| 2 | 8.0s | 9 | 5.9s | 7.8s | 0.00s |
| 3 | 10.2s | 8 | 5.9s | 9.9s | 0.00s |
| 4 | 8.0s | 6 | 6.4s | 7.8s | 0.00s |

### Hallazgos

- **Batch=8**: Semáforo NO es bottleneck (sem_wait ≈ 0s). 48 calls/step con concurrent=10 tiene capacidad de sobra.
- **Batch=24**: **Reward ES bottleneck** — pero por **rate limiting de Azure (429 errors), no por el semáforo**. reward_wall=127s vs gpu_phase=37-49s. sem_wait sube a 12s avg (63s max) por backpressure de retries.
- **Azure S0 tier**: token rate limit causa 429 errors con 144 calls/step + concurrent=10. El retry logic (espera 1s + retry) amplifica la latencia: api_avg sube de 6s (batch=8) a 20.6s (batch=24).
- **Reward es async**: los Ray RewardLoopWorkers computan rewards en paralelo con el GPU phase. A batch=8 terminan antes que la GPU; a batch=24, la GPU termina antes que el reward.

### Selección de modelo Judge (2026-03-19)

Validación de 5 modelos como Judge contra physician binary labels (50 entries HealthBench):

| Modelo | Kappa | Accuracy | Recurso | Latencia/call | Veredicto |
|--------|-------|----------|---------|-------------|-----------|
| gpt-5.2-chat | ~0.43 | ~0.68 | development-cursor-models (S0) | ~6s | ✅ Bueno pero rate limited |
| **gpt-5-mini** | **0.440** | **0.720** | development-cursor-models | ~10s | ✅ **Seleccionado** |
| gpt-5 | 0.400 | 0.700 | amalia-resource (4.8K RPM) | ~20s | ✅ Viable como backup |
| gpt-4o | 0.000 | 0.500 | amalia-resource | ~3s | ❌ No discrimina |
| gpt-4.1 | 0.000 | 0.000 | development-cursor-models | ~3s | ❌ No discrimina |

**Hallazgos**:
- Los modelos GPT-4.x **no sirven como Judge** — dan scores altos a todo (Mean>0.87), kappa=0.
- Solo los modelos GPT-5.x discriminan correctamente entre respuestas buenas y malas.
- **gpt-5-mini** es el mejor: mayor kappa, mayor accuracy, y al ser mini tiene rate limits más altos y menor costo.
- **Decisión**: cambiar de gpt-5.2-chat a gpt-5-mini como Judge para training.

### Costo API por step

| Batch | Calls/step | Cost/step (est.) | Cost/100 steps |
|-------|-----------|-------------------|----------------|
| 8 | ~48 | ~$0.48 | ~$48 |
| 24 | ~144 | ~$1.44 | ~$144 |

---

## 6. I/O Bottlenecks

### Checkpoint Save (122s)

El mayor overhead de I/O. veRL guarda 3 formatos por save:

```
global_step_N/actor/
  model_world_size_1_rank_0.pt     ← FSDP shard (modelo completo con LoRA)
  optim_world_size_1_rank_0.pt     ← Optimizador
  extra_state_world_size_1_rank_0.pt ← Scheduler, etc.
  fsdp_config.json
  huggingface/                      ← config.json + tokenizer (sin weights)
  lora_adapter/                     ← adapter_config.json + adapter_model.safetensors
```

**Impacto de save_freq**:

| save_freq | Saves en 200 steps | Overhead total |
|-----------|--------------------|--------------------|
| 10 | 20 | 40.7 min |
| 50 | 4 | 8.1 min |
| 100 | 2 | 4.1 min |
| 200 | 1 | 2.0 min |

**Recomendación**: save_freq=50 mínimo para producción.

### Weight Sync — update_weights (8s/step)

Sync de pesos entre FSDP (training) y vLLM (rollout). Es el costo del hybrid engine: ambos comparten la GPU pero tienen copias separadas de los pesos.

- 8s/step × 200 steps = 26.7 min (22% del training time)
- No se puede eliminar (es fundamental para el hybrid engine)
- Config `update_weights_bucket_megabytes: 2048` — ya en valor razonable

### Model Loading (startup)

- Loading checkpoint shards: ~1.2s (5 shards from HF cache, rápido)
- Setup total del modelo (FSDP wrapping, LoRA, vLLM init): ~173s
- CUDA graphs capture: ~18s (102 prefill + 102 decode)

---

## 7. Optimizaciones identificadas

### Ya aplicadas

| Cambio | Estado |
|--------|--------|
| `ppo_mini_batch_size: 64 → 24` | ✅ Fix (era bug, no corría) |

### Tier 1 — Alto impacto, riesgo cero

| Cambio | De → A | Impacto esperado | Estado |
|--------|--------|-----------------|--------|
| `gpu_memory_utilization` | 0.5 → 0.6+ | Más VRAM para vLLM → gen más rápido | 🟡 Pendiente |
| `ppo_micro_batch_size_per_gpu` | 4 → 8 | Menos micro-batches → update más rápido | 🟡 Pendiente |
| `log_prob_micro_batch_size_per_gpu` | 4 → 8 | Menos micro-batches → old_log_prob más rápido | 🟡 Pendiente |
| ~~`PYTORCH_CUDA_ALLOC_CONF`~~ | ~~`expandable_segments:True`~~ | **Incompatible con vLLM 0.17** (CuMemAllocator assert) | ❌ Descartado |
| `save_freq` | 200 (ya) | OK para producción | ✅ |

### Tier 2 — Requiere validar con datos

| Cambio | Condición | Impacto |
|--------|-----------|---------|
| `gpu_memory_utilization` → 0.7 | Solo si 0.6 no causa OOM | gen más rápido |
| `enable_chunked_prefill: true` | Si gen sigue dominando | Mejor prefill |
| Reducir `n: 6 → 4` | Si rate limit sigue siendo problema | -33% API calls (96 vs 144/step) |
| Upgrade Azure tier S0 → S1+ | Contactar Azure | Elimina 429 errors |

### Descartadas por profiling

| Optimización original (CHG-011) | Por qué se descarta |
|----------------------------------|---------------------|
| `JUDGE_MAX_CONCURRENT=24` (batch=8) | sem_wait=0, reward no es bottleneck |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **Incompatible con vLLM 0.17** — CuMemAllocator assertion (pytorch/pytorch#147851) |
| `free_cache_engine: false` | Riesgo de OOM, ganancia incierta |
| `use_dynamic_bsz: true` | Complejidad, batch fijo es más predecible |

---

## 8. Datos disponibles para training

| Dataset | Filas precomputadas | Batch máximo viable |
|---------|--------------------|--------------------|
| HealthBench | **88** (actualizado 2026-03-19) | batch=24+ |
| MedQA | 0 (sin cache) | — |
| MedMCQA | 0 (sin cache) | — |

**Resuelto**: 88 entries precomputadas localmente (~$5 API, ~15 min). Cache en `data/cache/healthbench_precompute.jsonl`.

---

## 9. Historial de profiling runs

| ID | Fecha | Config | Steps | Batch | Key finding |
|----|-------|--------|-------|-------|-------------|
| EXP-PROF-1A | 2026-03-19 | baseline, concurrent=10 | 5 | 8 | GPU domina (75%), reward async no bloquea, VRAM 35% |
| EXP-PROF-2 (fail) | 2026-03-19 | +expandable_segments | 0 | 24 | `expandable_segments` incompatible con vLLM 0.17 |
| EXP-PROF-2b | 2026-03-19 | micro=8, gpu_mem=0.6 | 5 | 24 | Reward bottleneck por Azure 429. GPU sub-lineal. VRAM=33GB (igual que batch=8!) |

---

## 10. Scaling: batch=8 vs batch=24

| Métrica | Batch=8 (Run 1A) | Batch=24 (Run 2b) | Factor |
|---------|-----------------|-------------------|--------|
| gen (steady) | 11.4s | 77-138s | 7-12x (inflado por reward wait) |
| update_actor | 10.4s | 16.0s | 1.5x (sub-lineal, micro=8 ayudó) |
| old_log_prob | 2.7s | 4.1s | 1.5x (sub-lineal) |
| update_weights | 8.0s | 8.0s | 1.0x (constante, solo depende de model size) |
| reward_wall | 8-10s | **71-136s** | 7-14x (rate limit 429!) |
| sem_wait | 0s | 1.5-12s avg | Backpressure por retries |
| api_avg | 6.2s | 17-22s | 3x (retries 429) |
| API calls/step | ~48 | ~144 | 3x |
| checkpoint save | 122s | 130s | ~igual |
| VRAM | 33.2 GB | **33.2 GB** | **igual** (!) |
| step (steady) | 32.5s | **105-167s** | 3-5x |

**Conclusiones**:
1. **GPU escala sub-linealmente**: update_actor 1.5x, update_weights constante. Eficiente.
2. **Bottleneck a batch=24 es Azure S0 rate limit** (429 errors). Sin rate limit, step sería ~105s.
3. **VRAM no cambia** entre batch=8 y batch=24 (33.2 GB). El headroom es del modelo, no del batch.
4. **gen time se infla** porque veRL espera rewards dentro del gen phase. Con rate limit, gen absorbe la espera.

### Proyección a producción (200 steps, batch=24)

| Escenario | Step time | Total | GPU cost | API cost |
|-----------|-----------|-------|----------|----------|
| Con rate limit S0 (actual) | ~135s | 7.5h | ~$52 | ~$144 |
| Sin rate limit (upgrade tier) | ~105s | 5.8h | ~$41 | ~$144 |
| Sin rate limit + n=4 | ~85s (est.) | 4.7h | ~$33 | ~$96 |

---

## 11. Preguntas abiertas

- [x] ¿Cómo escala el step time con batch=24? → GPU sub-lineal (~1.7x), reward explota por rate limit
- [ ] ¿Upgrade de Azure tier (S0→S1+) elimina los 429 errors?
- [ ] ¿Reducir n de 6 a 4 basta para evitar rate limit con S0?
- [ ] ¿update_weights se puede reducir con `update_weights_bucket_megabytes` más alto?
- [ ] ¿El checkpoint save se puede optimizar guardando solo LoRA adapter (sin FSDP full)?
- [ ] ¿`enforce_eager: true` (skip CUDA graphs) reduce startup significativamente? (trade-off: gen más lento)
- [x] ¿`gpu_memory_utilization: 0.6` funciona? → Sí, vLLM inicia OK (Run 2b)
- [x] ¿`expandable_segments:True` es compatible? → **NO**, incompatible con vLLM 0.17
