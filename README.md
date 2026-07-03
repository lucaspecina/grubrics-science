# grubrics-science · 📦 ARCHIVADO

> **Este repositorio está archivado (read-only) y se conserva como registro histórico.**
> El proyecto evolucionó y continúa en:
>
> ### → **[goodhart-rubrics](https://github.com/lucaspecina/goodhart-rubrics)**
>
> Todo el desarrollo activo, el código vigente y los datos migraron allá. Este repo
> queda congelado como la memoria intelectual de cómo se llegó hasta acá.

---

## Qué fue este proyecto

Nació (feb 2026) como **GRubrics**: entrenar un modelo chico (Qwen3-8B + LoRA, GRPO sobre
veRL) para **generar rúbricas de evaluación médica**, con la señal de *functional alignment*
(una rúbrica es buena si reproduce los rankings de expertos humanos).

A lo largo de 2026 mutó dos veces, cada mutación documentada con su evidencia:

1. **Pivote (jun 2026, CHG-022)**: de "imitar rúbricas humanas" a **"rubricator adaptativo
   anti-hacking como capa de calibración del RL"** — al descubrir que los judges reconocen
   calidad pero no saben inducir criterios, y que las rúbricas estáticas se hackean durante
   el entrenamiento.
2. **Reframing adversarial (jul 2026, CHG-024/025)**: de un artefacto a un **fenómeno medible**
   — la carrera entre la presión de optimización y la evaluación que se defiende. Verificado
   contra la literatura (103 agentes, 21 fuentes) y con un diseño experimental atacado por
   3 revisores adversariales. Ese es el proyecto que continúa en **goodhart-rubrics**.

## Qué queda de valor acá (y qué migró)

- **La historia de decisiones completa**: `CHANGELOG.md` (CHG-001..026) — por qué cada
  cambio de rumbo, qué se descartó y con qué evidencia.
- **Los fundamentos y el mapa de literatura**: `docs/theoretical-foundations.md`,
  `docs/adversarial-evaluation-reframing.md`, `docs/related-work.md`.
- **El diseño experimental v2.1**: `docs/adversarial-evaluation-design.md`.
- **La wiki del proyecto** (su forma final): `WIKI.md`.
- **Resultados de la Fase 0**: `docs/experiment-log.md` (EXP-PHASE0-B4 y otros).

El **código y los datos vigentes** (judge binario, panel sin rúbrica, paquete phase0
completo, pares de entrenamiento, rollout-sets) migraron a goodhart-rubrics como base.

## Punto final de esta era

Tag: `final-grubrics-science`. Todo lo posterior vive en el repo sucesor.
