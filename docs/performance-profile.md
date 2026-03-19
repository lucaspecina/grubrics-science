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

- **Semáforo NO es bottleneck**: sem_wait ≈ 0s en estado estable. Con batch=8 (48 API calls/step) y concurrent=10, hay capacidad de sobra.
- **API latency**: ~6s promedio por call (Azure OpenAI). Es la latencia del modelo, no se puede optimizar desde nuestro lado.
- **Reward es async**: los Ray RewardLoopWorkers computan rewards en paralelo con el GPU phase. El reward termina antes que la GPU (~10s reward vs ~22-25s gpu_phase).
- **A batch=24**: 144 calls/step. Con concurrent=10, reward_wall subiría a ~25-30s. Si gpu_phase también escala, probablemente siguen balanceados. Monitorear.

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
| `PYTORCH_CUDA_ALLOC_CONF` | (nada) → `expandable_segments:True` | Menos fragmentación VRAM | 🟡 Pendiente |
| `save_freq` | 200 (ya) | OK para producción | ✅ |

### Tier 2 — Requiere validar con datos

| Cambio | Condición | Impacto |
|--------|-----------|---------|
| `gpu_memory_utilization` → 0.7 | Solo si 0.6 no causa OOM | gen más rápido |
| `enable_chunked_prefill: true` | Si gen sigue dominando | Mejor prefill |
| `JUDGE_MAX_CONCURRENT` → 24 | Solo si batch=24 muestra sem_wait > 0 | Reward más rápido |

### Descartadas por profiling

| Optimización original (CHG-011) | Por qué se descarta |
|----------------------------------|---------------------|
| `JUDGE_MAX_CONCURRENT=24` (batch=8) | sem_wait=0, reward no es bottleneck |
| `free_cache_engine: false` | Riesgo de OOM, ganancia incierta |
| `use_dynamic_bsz: true` | Complejidad, batch fijo es más predecible |

---

## 8. Datos disponibles para training

| Dataset | Filas precomputadas | Batch máximo viable |
|---------|--------------------|--------------------|
| HealthBench | 9 | batch=8 |
| MedQA | 0 (sin cache) | — |
| MedMCQA | 0 (sin cache) | — |

**Blocker para batch=24**: necesitamos ≥24 filas precomputadas (idealmente 100+). Ver TODO-006.

---

## 9. Historial de profiling runs

| ID | Fecha | Config | Steps | Batch | Key finding |
|----|-------|--------|-------|-------|-------------|
| EXP-PROF-1A | 2026-03-19 | baseline, concurrent=10 | 5 | 8 | GPU domina (75%), reward async no bloquea, VRAM 35% |

---

## 10. Preguntas abiertas

- [ ] ¿Cómo escala el step time con batch=24? (necesita más datos precomputados)
- [ ] ¿update_weights se puede reducir con `update_weights_bucket_megabytes` más alto?
- [ ] ¿El checkpoint save se puede optimizar guardando solo LoRA adapter (sin FSDP full)?
- [ ] ¿`enforce_eager: true` (skip CUDA graphs) reduce startup significativamente? (trade-off: gen más lento)
- [ ] ¿Cuál es el impacto real de `gpu_memory_utilization: 0.6` en gen time?
