- Version simple:
    
    ```markdown
    ================================================================================
    RLCER-Transfer (versión mínima viable)
    ================================================================================
    
    UN MODELO πθ, DOS ROLES: Reasoner + Rubricator
    UN GRADER EXTERNO CONGELADO (LLM judge, e.g., GPT-4o)
    
    DOS FUENTES DE DATOS (mezcladas en cada batch):
      - D_verif: (pregunta, respuesta correcta)        → abundante, gratis
      - D_open:  (pregunta, rúbricas humanas)           → pocas, caras
    
    CURRICULUM:
      inicio → mayoría D_verif    |    final → 50/50
    
    ────────────────────────────────────────────
    PARA CADA BATCH:
    ────────────────────────────────────────────
    
    Si tarea es VERIFICABLE:
      1. Reasoner genera N respuestas con CoT
      2. Rubricator genera rúbricas
      3. Grader evalúa CoTs contra rúbricas → satisfacción vₖ
      4. Se calcula correctness z de las respuestas
      5. Rúbrica es válida si corr(vₖ, z) > α y std(vₖ) > 0
      6. Rewards:
         - Reasoner:   r = outcome(±1) + CoT_quality(rúbricas válidas)
         - Rubricator:  r = fracción_rúbricas_válidas + formato
    
      → Esto es RLCER estándar. Señal gratis y masiva.
    
    Si tarea es NO VERIFICABLE:
      1. Reasoner genera N respuestas con CoT
      2. Rubricator genera rúbricas
      3. Grader puntúa los N rollouts con rúbricas generadas → ranking_gen
      4. Grader puntúa los N rollouts con rúbricas humanas   → ranking_hum
      5. Rewards:
         - Reasoner:   r = CoT_quality(rúbricas generadas)
         - Rubricator:  r = ranking_consistency(ranking_gen, ranking_hum)
                          + discriminatividad(std > 0)
                          + formato
    
      → Las rúbricas generadas se alinean con las humanas por RANKING,
        no por score absoluto. Más robusto.
    
    OPTIMIZACIÓN: PPO con advantages por rol. Mismos θ para ambos roles.
    
    ────────────────────────────────────────────
    ESO ES TODO.
    ────────────────────────────────────────────
    
    Lo demás (active learning, grader interno, loop auto-mejorante)
    son extensiones opcionales para el paper completo.
    ```
    
- Version extendida
    
    ```markdown
    ================================================================================
    RLCER-Transfer: Unified Rubric Learning across Verifiable and Open Domains
    ================================================================================
    
    ARQUITECTURA: Un solo policy model πθ con 3 roles (mismo modelo, distintos prompts)
    ─────────────────────────────────────────────────────────────────────────────────
      - Reasoner   πᴿᵉᵃ(·) = πθ(· | Pᴿᵉᵃ)    → genera CoT + respuesta
      - Rubricator πᴿᵘᵇ(·) = πθ(· | Pᴿᵘᵇ)    → genera rúbricas para evaluar CoTs
      - Grader     πᴳʳᵈ(·) = πθ(· | Pᴳʳᵈ)    → evalúa CoTs contra rúbricas
    
      ¿Por qué 3 roles y no 2 como RLCER?
      → RLCER usa un verifier externo congelado (πϕ) para evaluar rúbricas.
        Eso tiene un techo: el verifier no mejora. Internalizando el Grader 
        en πθ, el evaluador CO-EVOLUCIONA con el rubricator y el reasoner.
        Además, el Grader se calibra en tareas verificables (donde hay ground 
        truth) y transfiere esa calibración a tareas abiertas.
    
    DATOS DE ENTRENAMIENTO
    ─────────────────────────────────────────────────────────────────────────────────
      D_verif:  tareas verificables (math, código, lógica)  → abundantes, gratis
                cada instancia tiene: (Q, A_ground_truth)
      
      D_open:   tareas no verificables (medicina, escritura, argumentación)
                cada instancia tiene: (Q, rúbricas_humanas)
                inicialmente POCAS instancias (budget humano limitado)
    
      ¿Por qué usar las dos simultáneamente y no en fases?
      → Evita catastrophic forgetting entre fases.
      → D_verif da señal GRATIS y MASIVA para aprender "qué es una buena rúbrica".
      → D_open da señal CARA pero ESPECÍFICA del dominio target.
      → Entrenando juntas, el rubricator aprende la habilidad general (de D_verif)
        y la calibración de dominio (de D_open) sin perder ninguna de las dos.
    
    CURRICULUM: proporción de tareas por step
    ─────────────────────────────────────────────────────────────────────────────────
      step 0-500:      w_verif=0.9,  w_open=0.1
      step 500-1000:   w_verif=0.7,  w_open=0.3
      step 1000-1500:  w_verif=0.5,  w_open=0.5
    
      ¿Por qué curriculum y no 50/50 desde el inicio?
      → Al principio el rubricator no sabe nada. Las tareas verificables le dan 
        señal abundante y barata para aprender los fundamentos (correlación entre
        rubric satisfaction y correctness). Si le tirás muchas tareas abiertas al
        inicio, la señal es poca (pocas rúbricas humanas) y ruidosa (el grader 
        aún no está calibrado). Mejor arrancar con lo barato y estable, y 
        gradualmente aumentar la exposición al dominio target.
    
    ================================================================================
    TRAINING LOOP (en cada step t)
    ================================================================================
    
    Para cada batch, samplear tareas de D_verif y D_open según curriculum:
    
    ────────────────────────────────────────────
    FLUJO A: Tarea verificable (Q, A_gt)
    ────────────────────────────────────────────
    
      1. Reasoner genera N rollouts:
         {Ĉₙ, Âₙ}ᴺ ~ πᴿᵉᵃ(· | Q)
    
      2. Rubricator genera K rúbricas por rollout:
         R̂ₙ = {τ̂ₖ}ᴷ ~ πᴿᵘᵇ(· | Q, Ĉₙ)
         donde τ̂ₖ = (ĉₖ, ŝₖ)   // criterio textual + score de importancia
    
      3. Grader evalúa cada rúbrica contra cada CoT:
         vₖₙ = πᴳʳᵈ(ĉₖ, Ĉₙ) ∈ {0, 1}    // ¿CoT n satisface rúbrica k?
    
      4. Correctness vector:
         z = [I(A_gt, Â₁), ..., I(A_gt, Âₙ)]   // 1 si respuesta correcta, 0 si no
    
      5. Validación de rúbricas (igual que RLCER):
         τ̂ₖ es válida  ⟺  corr(vₖ, z) > α  AND  std(vₖ) > 0
    
         ¿Por qué este criterio?
         → corr > α: la rúbrica captura algo que correlaciona con calidad real.
         → std > 0: la rúbrica discrimina (no todos los rollouts la satisfacen).
         → Juntas: la rúbrica es informativa como señal de reward.
    
      6. Rewards:
    
         Reasoner:
           rᴿᵉᵃ = rᵒᵘᵗᶜᵒᵐᵉ + rᶜᵒᵗ
           donde:
             rᵒᵘᵗᶜᵒᵐᵉ = +1 si correcta, -1 si incorrecta
             rᶜᵒᵗ     = norm(Σ vₖ · ŝₖ) para τ̂ₖ ∈ R̂_valid    // RLCER estándar
    
         Rubricator:
           rᴿᵘᵇ = K_valid/K + rᶠᵒʳᵐᵃᵗ
           // fracción de rúbricas válidas + bonus por formato correcto
    
         Grader:
           rᴳʳᵈ = accuracy(grading vs z)
           // el grader es recompensado si su evaluación de las rúbricas
           // es consistente con la correctness real de las respuestas.
           
           ¿Por qué?
           → En tareas verificables, SABEMOS si la respuesta es correcta.
             Si una rúbrica válida dice "no saltar pasos" y el grader dice
             que un CoT la satisface, podemos verificar indirectamente:
             ¿los CoTs que el grader marca como "satisface" tienden a tener
             respuestas correctas? Si sí, el grader está bien calibrado.
           → Esta calibración se TRANSFIERE al Flujo B donde no hay ground truth.
    
    ────────────────────────────────────────────
    FLUJO B: Tarea no verificable (Q, rúbricas_humanas)
    ────────────────────────────────────────────
    
      1. Reasoner genera N rollouts:
         {Ĉₙ}ᴺ ~ πᴿᵉᵃ(· | Q)
    
      2. Rubricator genera K rúbricas:
         R̂ = {τ̂ₖ}ᴷ ~ πᴿᵘᵇ(· | Q, Ĉₙ)
    
      3. Grader puntúa cada rollout DOS VECES:
         score_gen_n  = πᴳʳᵈ(R̂, Ĉₙ)          // score usando rúbricas GENERADAS
         score_hum_n  = πᴳʳᵈ(R_humanas, Ĉₙ)   // score usando rúbricas HUMANAS
    
      4. Ranking consistency (clave de esta propuesta):
         En vez de comparar scores absolutos, comparamos el RANKING:
    
         Para cada par de rollouts (i, j):
           concordancia(i,j) = 1 si sign(score_gen_i - score_gen_j) 
                                    == sign(score_hum_i - score_hum_j)
    
         ranking_consistency = promedio(concordancia) sobre todos los pares (i,j)
    
         ¿Por qué ranking y no score absoluto?
         → Los scores absolutos dependen de la escala, el sesgo del grader,
           y la granularidad de la rúbrica. Dos rúbricas pueden medir lo mismo
           pero dar scores en rangos distintos (0-5 vs 0-100).
         → El RANKING es lo que importa para RL: necesitamos saber qué rollout
           es MEJOR que cuál, no el score exacto. Si las rúbricas generadas 
           ordenan las respuestas igual que las humanas, capturan la misma 
           noción de calidad.
         → Es equivalente a Kendall tau o Spearman rank correlation, que son
           métricas robustas a transformaciones monótonas del score.
    
      5. Discriminatividad (no necesita ground truth):
         disc = std([score_gen_1, ..., score_gen_N])
    
         ¿Por qué incluir esto?
         → Si las rúbricas generadas dan el MISMO score a todos los rollouts,
           el ranking consistency es trivialmente alto (todo empata).
           Pero eso es inútil para RL — necesitamos que las rúbricas 
           DIFERENCIEN entre rollouts buenos y malos.
         → disc > 0 asegura que las rúbricas son informativas.
    
      6. Rewards:
    
         Reasoner:
           rᴿᵉᵃ = rᶜᵒᵗ_gen
           // sin outcome reward (no hay ground truth), solo CoT reward basado
           // en rúbricas generadas (que se están alineando con las humanas).
           
           ¿Esto funciona?
           → Sí, RLCER ya demostró en Sec 5.2 que recompensar solo con 
             rúbricas (sin outcome) mejora el reasoning. Y acá las rúbricas 
             están siendo alineadas con criterio humano, así que la señal 
             es aún más informativa.
    
         Rubricator:
           rᴿᵘᵇ = α · ranking_consistency(score_gen, score_hum)
                 + β · disc(score_gen)
                 + γ · rᶠᵒʳᵐᵃᵗ
    
           ¿Por qué esta combinación?
           → ranking_consistency: que las rúbricas generadas capturen la misma
             noción de calidad que las humanas.
           → disc: que no colapsen a dar todo el mismo score.
           → format: que sean parseables y estructuradas.
    
           NOTA: NO incluyo diversidad(gen vs humanas) ni anti-redundancia.
           
           ¿Por qué eliminé la diversidad?
           → Era un parche para evitar parafraseo, pero ranking_consistency
             ya resuelve eso de forma más elegante: si las rúbricas generadas
             parafrasean las humanas, el ranking será idéntico (reward alto),
             PERO eso no es malo — significa que capturan lo mismo.
             El modelo NATURALMENTE va a encontrar rúbricas más eficientes
             o complementarias si eso le ayuda a mantener discriminatividad
             en casos difíciles donde las humanas no discriminan bien.
           → Forzar diversidad artificialmente puede producir rúbricas 
             divergentes que no capturan calidad real.
    
         Grader:
           rᴳʳᵈ = consistencia interna entre sus evaluaciones con rúbricas
                   generadas vs humanas sobre los mismos CoTs.
           
           ¿Por qué?
           → El grader debería dar scores consistentes independientemente
             de si la rúbrica fue generada o humana, SI ambas capturan lo 
             mismo. Cuando hay discrepancia, es señal de que o la rúbrica
             generada es mala, o el grader no entiende bien la rúbrica.
             Recompensando consistencia, el grader aprende a interpretar
             rúbricas de forma estable.
    
    ================================================================================
    ACTIVE LEARNING: adquisición eficiente de rúbricas humanas
    ================================================================================
    
      El budget humano es LIMITADO. No queremos gastar rúbricas humanas en
      preguntas donde el rubricator ya es bueno.
    
      Cada M steps, para un pool de preguntas nuevas de D_open:
    
      1. El rubricator genera rúbricas K veces para cada Q (con temperatura alta):
         R̂₁, R̂₂, ..., R̂ₘ ~ πᴿᵘᵇ(· | Q, Ĉ)
    
      2. Calculamos incertidumbre del rubricator:
         incertidumbre(Q) = varianza entre los rankings producidos por 
                            R̂₁, R̂₂, ..., R̂ₘ sobre los mismos rollouts
    
         ¿Qué significa alta incertidumbre?
         → El rubricator genera rúbricas MUY DISTINTAS para la misma pregunta.
           A veces esas rúbricas ponen rollout A > B, a veces B > A.
           No sabe qué evaluar → necesita guía humana.
    
      3. Seleccionamos las top-P preguntas con mayor incertidumbre.
    
      4. Mandamos esas P preguntas a anotadores humanos para que escriban rúbricas.
    
      5. Agregamos esas (Q, rúbricas_humanas) a D_open.
    
      ¿Por qué esto es crucial?
      → Sin active learning, gastás el budget humano uniformemente, incluyendo
        preguntas fáciles donde el rubricator ya genera rúbricas buenas.
      → Con active learning, cada rúbrica humana se gasta donde MÁS IMPACTA:
        en los casos difíciles/ambiguos donde el rubricator está perdido.
      → En la práctica esto puede significar necesitar 3x-5x MENOS rúbricas 
        humanas para el mismo nivel de calidad. Eso es lo que hace el método
        económicamente viable.
    
    ================================================================================
    OPTIMIZACIÓN CONJUNTA
    ================================================================================
    
      Todos los roles comparten πθ. Se optimizan con PPO (no GRPO, misma razón
      que RLCER: los rollouts del rubricator y grader tienen contextos distintos).
    
      J(θ) = E_D_verif [ Σ_roles min(ρ·Â, clip(ρ)·Â) ]    // Flujo A
            + E_D_open  [ Σ_roles min(ρ·Â, clip(ρ)·Â) ]    // Flujo B
    
      donde los advantages Â son role-specific (AᴿᵉᵃÂ, ᴿᵘᵇÂ, ᴳʳᵈ) y se computan
      por separado para cada rol, pero los gradientes actualizan los mismos θ.
    
      El peso relativo de D_verif vs D_open sigue el curriculum definido arriba.
    
    ================================================================================
    POST-ENTRENAMIENTO: loop auto-mejorante
    ================================================================================
    
      Una vez que el rubricator está calibrado:
    
      1. Rubricator genera rúbricas para Qs NUEVAS (sin rúbricas humanas)
             ↓
      2. Grader evalúa rollouts del Reasoner contra esas rúbricas
             ↓
      3. Reasoner mejora con esa señal
             ↓
      4. Mejores rollouts → Rubricator puede afinar más las rúbricas
             ↓
      5. Cada M steps: active learning selecciona Qs difíciles → humano anota
             ↓
      6. Las nuevas rúbricas humanas re-calibran al rubricator
             ↓
      (volver a 1)
    
      ¿Por qué esto es sostenible?
      → El humano interviene CADA VEZ MENOS: a medida que el rubricator mejora,
        hay menos Qs con alta incertidumbre → se necesitan menos anotaciones.
      → El loop converge a un rubricator autónomo con intervención humana
        esporádica solo para corregir drift.
    
    ================================================================================
    RESUMEN DE BENEFICIOS vs ALTERNATIVAS
    ================================================================================
    
      vs RLCER puro:
        + Funciona en dominios no verificables
        + Grader interno que co-evoluciona (no verifier congelado)
        + Rúbricas calibradas con criterio humano
    
      vs Rubric Anchors:
        + Rúbricas evolucionan (no estáticas)
        + Active learning para adquisición eficiente
        + Pre-training gratis en dominios verificables
    
      vs Self-Rewarding Rubric RL:
        + Bootstrapping desde dominios verificables (no arranca de cero)
        + Rúbricas humanas como ancla (no depende solo de self-reward)
        + Active learning para guiar al humano
    
      vs Pipeline secuencial de 3 fases:
        + Sin catastrophic forgetting (entrenamiento continuo)
        + Sin transiciones frágiles (curriculum suave)
        + Señal verificable siempre presente como grounding
        + Más eficiente en uso de rúbricas humanas (active learning)
    
    ================================================================================
    ```
    
- INTRO TIPO PAPER (con referencias)
    
    Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for improving chain-of-thought (CoT) reasoning in large language models (LLMs), achieving remarkable results in mathematics [7, 13, 23], code generation [14], and logical puzzles [5]. A key ingredient of RLVR's success is the availability of cheap, automatic verifiers: given a question and a candidate answer, correctness can be checked without human intervention. This property enables scalable training with millions of reward signals at negligible cost.
    
    However, many real-world reasoning tasks lack such automatic verifiers. Medical diagnosis, legal argumentation, scientific analysis, and persuasive writing all require nuanced, multi-criteria evaluation where no single "correct answer" exists. Extending the power of RLVR to these *non-verifiable* domains remains a central open challenge [11, 35].
    
    **Rubrics as a bridge.** Recent work has shown that *rubrics* — structured, checkable evaluation criteria — can serve as reward signals for RL training in non-verifiable domains [2, 11, 12, 35]. Rubrics as Rewards (RaR) [Gunjal et al., 2025] demonstrated that rubric-based feedback outperforms Likert-based LLM-as-judge rewards, achieving up to 31% relative improvement on HealthBench. Reinforcement Learning with Rubric Anchors [Huang et al., 2025] scaled this approach with over 10,000 rubrics from humans and LLMs, showing strong gains on open-ended tasks. These results establish rubrics as a viable alternative to verifiable rewards.
    
    **The cost bottleneck.** Despite their promise, rubric-based approaches face a critical scalability problem: *who writes the rubrics?* Existing methods rely on large sets of human-authored or LLM-generated rubrics, created per-instance or per-task [Huang et al., 2025]. Human rubrics are expensive to produce at scale. LLM-generated rubrics are cheaper but lack the quality guarantees of human expertise. Neither approach evolves during training: as the policy improves, static rubrics may become saturated or misaligned with the model's current capability [22].
    
    **Self-evolving rubrics — but only for verifiable tasks.** A parallel line of work addresses rubric evolution. RLCER [Sheng et al., 2026] introduced self-evolving rubrics that co-evolve with the policy during RL training, avoiding saturation by rewarding the rubricator based on the correlation between rubric satisfaction and final-answer correctness. DR-Tulu [2025] similarly proposed evolving rubrics for deep research tasks. However, these self-evolving approaches fundamentally depend on outcome verification: rubric validity is assessed via correlation with correctness signals, confining them to verifiable domains.
    
    **The gap.** We identify a gap at the intersection of these two research directions:
    
    - Rubric-based RL works in open domains but requires *expensive, static* human rubrics.
    - Self-evolving rubrics are *cheap and adaptive* but only work in verifiable domains.
    
    No existing method combines the adaptiveness of self-evolving rubrics with the domain coverage of human-authored rubrics to achieve scalable, evolving CoT supervision in non-verifiable domains.
    
    **Our approach.** We propose **RLCER-Transfer**, a unified framework that bridges verifiable and non-verifiable domains through joint rubric learning. Our key insight is that *the skill of generating good rubrics is largely domain-general*: a rubric that captures "logical consistency" or "avoidance of unsupported claims" is valuable whether the task is a math proof or a medical diagnosis. We exploit this by training a single rubricator simultaneously on:
    
    1. **Verifiable tasks**, where rubric quality is validated for free via correlation with answer correctness (following RLCER), providing abundant, cheap learning signal.
    2. **Non-verifiable tasks**, where rubric quality is validated via *ranking consistency* with a small set of human-authored rubrics, providing expensive but domain-specific calibration.
    
    A curriculum schedule gradually shifts the training distribution from mostly-verifiable (cheap bootstrap) to balanced (domain adaptation). The rubricator learns general rubric-generation capabilities from verifiable tasks and transfers them to the target domain, guided by few human rubrics.
    
    Our formulation introduces *ranking consistency* as the alignment signal for non-verifiable tasks: rather than matching absolute scores (which are sensitive to scale and judge bias), we require that rubrics generated by the model induce the same *relative ordering* of responses as human rubrics. This is both more robust and more aligned with what RL actually needs — a reliable preference signal over rollouts.
    
    We make the following contributions:
    
    - We propose the first framework that jointly trains a rubricator across verifiable and non-verifiable domains, enabling transfer of rubric-generation capabilities from cheap verifiable tasks to expensive open-ended ones.
    - We introduce ranking consistency as a reward signal for rubric alignment in non-verifiable domains, which is more robust than score-level comparison.
    - We show that curriculum-based joint training outperforms sequential approaches (pre-train then transfer) by avoiding catastrophic forgetting and maintaining continuous grounding from verifiable signals.
    - We demonstrate that our approach reduces the number of human rubrics needed by [X]x compared to static rubric-based RL, while matching or exceeding its performance.
- MEJORA → agregandole nuevas ideas
    
    Encontré tres ideas de otros campos que creo que pueden mejorar sustancialmente la propuesta. Dejame explicar cada una y después cómo se integran.
    
    ---
    
    ## 3 ideas de fuera que cambian el juego
    
    ### Idea 1: Rubricator CONTRASTIVO (inspirado en OpenRubrics/CRG)
    
    Nuestro rubricator actual genera rúbricas mirando UN CoT aislado. Pero hay un paper reciente — **OpenRubrics (Oct 2025)** — que muestra que las mejores rúbricas salen de **contrastar una respuesta buena con una mala** y pedir: "¿qué las diferencia?"
    
    En dominios verificables, tenemos pares contrastivos **gratis**: de los N rollouts para la misma pregunta, algunos tienen respuesta correcta y otros incorrecta. Entonces:
    
    ```
    ACTUAL:     Rubricator(Q, Ĉ)              → genera rúbricas genéricas
    MEJORADO:   Rubricator(Q, Ĉ_good, Ĉ_bad) → genera rúbricas que EXPLICAN la diferencia
    ```
    
    ¿Por qué esto es mejor?
    
    - Las rúbricas son **inherentemente discriminativas** por construcción. No necesitás filtrar después con `std > 0` porque la rúbrica nació para distinguir entre bueno y malo.
    - Es más natural: un experto humano escribe mejores rúbricas cuando ve un ejemplo bueno y uno malo lado a lado, no mirando uno solo.
    - Y lo más importante: **el skill de "identificar qué distingue buen razonamiento de malo" transfiere entre dominios** mucho mejor que "generar criterios que correlacionen con correctness matemática."
    
    ### Idea 2: Rúbricas JERÁRQUICAS (inspirado en Auto-Rubric)
    
    Hay un paper — **Auto-Rubric (Oct 2025)** — que muestra algo sorprendente: con solo **70 pares de preferencia** (1.5% de los datos), generando rúbricas jerárquicas tipo "Theme-Tips", un Qwen3-8B supera a modelos entrenados con datasets completos.
    
    La estructura jerárquica es:
    
    ```
    Meta-rúbricas (transferibles entre dominios):
      "Consistencia lógica"
      "Uso de evidencia"
      "No contradicción"
          ↓
    Sub-rúbricas (transferibles dentro del dominio):
      "Cada paso sigue del anterior"
      "No hay saltos lógicos injustificados"
          ↓
    Checks instancia-específicos (por pregunta):
      "Aplica correctamente integración por partes en paso 3"
    ```
    
    ¿Por qué esto importa para nosotros? Porque **el transfer entre dominios pasa por las meta-rúbricas**. Las meta-rúbricas aprendidas en matemáticas ("consistencia lógica", "verificación de pasos intermedios") aplican directamente a medicina o argumentación legal. Los checks instancia-específicos no transfieren, pero las meta-rúbricas sí.
    
    ### Idea 3: Dificultad adaptativa de rúbricas (inspirado en RLVE)
    
    **RLVE (Nov 2025)** muestra que en RL, ajustar dinámicamente la dificultad de los problemas al nivel actual del modelo produce ganancias enormes (3.37% vs 0.49% con 3x más compute). Apliquemos esto a rúbricas:
    
    Track la **tasa de satisfacción** de cada rúbrica:
    
    - Si una rúbrica se satisface >90% de las veces → **demasiado fácil**, no da señal
    - Si una rúbrica se satisface <10% de las veces → **demasiado difícil**, no da señal
    - La zona óptima es ~30-70% → máxima información
    
    El rubricator debería recibir reward extra por generar rúbricas en la **zona de máxima información**, no solo válidas. Esto es un refinamiento del mecanismo de evolución de RLCER que es mucho más preciso.
    
    ---
    
    ## La propuesta mejorada: RLCER-Transfer v2
    
    ```
    ================================================================================
    CAMBIOS CLAVE vs v1
    ================================================================================
    
    CAMBIO 1: Rubricator CONTRASTIVO
    ─────────────────────────────────
      Antes:  πᴿᵘᵇ(· | Q, Ĉ)              → rúbricas desde un solo CoT
      Ahora:  πᴿᵘᵇ(· | Q, Ĉ⁺, Ĉ⁻)        → rúbricas desde par contrastivo
    
      En dominios verificables:
        Ĉ⁺ = CoT con respuesta correcta (de los N rollouts)
        Ĉ⁻ = CoT con respuesta incorrecta (de los N rollouts)
        → Pares contrastivos GRATIS
    
      En dominios no verificables:
        Ĉ⁺ = CoT mejor rankeado por rúbricas humanas
        Ĉ⁻ = CoT peor rankeado por rúbricas humanas
        → Pares contrastivos derivados de rúbricas humanas
    
      El rubricator genera rúbricas respondiendo:
        "¿Qué hace Ĉ⁺ mejor que Ĉ⁻? Lista criterios específicos."
    
      BENEFICIO: las rúbricas son discriminativas POR CONSTRUCCIÓN.
      No necesitás filtrar con std > 0 después. Y el skill de
      "identificar diferencias entre razonamiento bueno y malo"
      es más transferible que "generar criterios genéricos".
    
    CAMBIO 2: Rúbricas JERÁRQUICAS
    ───────────────────────────────
      El rubricator genera rúbricas en 2 niveles:
    
      Nivel 1 - Meta-rúbricas (pocas, estables, transferibles):
        Generadas y actualizadas cada M steps.
        Ejemplos: "consistencia lógica", "verificación de intermedios",
        "no asumir lo que se quiere probar".
        → Estas SON las que transfieren entre dominios.
        → Se mantienen en un "banco de meta-rúbricas" compartido.
    
      Nivel 2 - Rúbricas instancia-específicas (muchas, por pregunta):
        Generadas por el rubricator condicionadas en las meta-rúbricas:
        πᴿᵘᵇ(· | Q, Ĉ⁺, Ĉ⁻, meta-rúbricas)
        → Instanciaciones concretas para la pregunta Q.
    
      BENEFICIO para transfer:
        Cuando pasás de dominio verificable a abierto, las meta-rúbricas
        ya están aprendidas. El rubricator solo necesita aprender a
        instanciarlas en el nuevo dominio, no descubrirlas de cero.
        Esto hace el transfer MUCHO más eficiente.
    
    CAMBIO 3: Reward de ZONA ÓPTIMA (reemplaza fracción_válidas)
    ─────────────────────────────────────────────────────────────
      Antes:  rᴿᵘᵇ = K_valid / K  (fracción de rúbricas válidas)
      Ahora:  rᴿᵘᵇ = Σₖ info_value(τₖ)
    
      donde:
        satisfaction_rate(τₖ) = promedio de πᴳʳᵈ(ĉₖ, Ĉₙ) sobre N rollouts
        info_value(τₖ) = 4 · satisfaction_rate · (1 - satisfaction_rate)
                       → máximo cuando sat_rate = 0.5 (máxima entropía)
                       → cero cuando sat_rate = 0 o 1 (sin información)
    
      Esto es simplemente la varianza de una Bernoulli, normalizada.
    
      BENEFICIO: no solo recompensás rúbricas "válidas" (corr > α),
      sino que recompensás rúbricas en la ZONA DE MÁXIMA INFORMACIÓN.
      Una rúbrica que se satisface 50% del tiempo da más señal para RL
      que una que se satisface 90%, incluso si ambas correlacionan con
      correctness. Esto previene saturación de forma más elegante que
      RLCER y empuja al rubricator a buscar rúbricas DESAFIANTES.
    
    ================================================================================
    FLUJO COMPLETO v2 (simplificado)
    ================================================================================
    
    UN MODELO πθ, DOS ROLES: Reasoner + Rubricator
    UN GRADER EXTERNO CONGELADO
    UN BANCO DE META-RÚBRICAS (actualizado cada M steps)
    
    DOS FUENTES DE DATOS (curriculum: más verificable al inicio):
      D_verif: (Q, A_gt)            → abundante, gratis
      D_open:  (Q, rúbricas_humanas) → pocas, caras
    
    ────────────────────────────────────────────
    Si tarea VERIFICABLE:
    ────────────────────────────────────────────
      1. Reasoner genera N rollouts: {Ĉₙ, Âₙ}
      2. Separar: Ĉ⁺ (correctas), Ĉ⁻ (incorrectas)
      3. Rubricator genera rúbricas contrastivas:
         πᴿᵘᵇ(· | Q, Ĉ⁺, Ĉ⁻, meta-rúbricas) → {τₖ}
      4. Grader evalúa satisfacción de cada τₖ en cada Ĉₙ
      5. Rewards:
         Reasoner:   outcome(±1) + CoT_quality(rúbricas válidas)
         Rubricator: Σ info_value(τₖ) · I(corr(vₖ,z) > α) + formato
                     ↑ zona óptima      ↑ correlación con correctness
    
    ────────────────────────────────────────────
    Si tarea NO VERIFICABLE:
    ────────────────────────────────────────────
      1. Reasoner genera N rollouts: {Ĉₙ}
      2. Grader rankea con rúbricas humanas → ranking_hum
      3. Separar: Ĉ⁺ (mejor según humanas), Ĉ⁻ (peor según humanas)
      4. Rubricator genera rúbricas contrastivas:
         πᴿᵘᵇ(· | Q, Ĉ⁺, Ĉ⁻, meta-rúbricas) → {τₖ}
      5. Grader rankea con rúbricas generadas → ranking_gen
      6. Rewards:
         Reasoner:   CoT_quality(rúbricas generadas)
         Rubricator: ranking_consistency(ranking_gen, ranking_hum)
                     + Σ info_value(τₖ) + formato
    
    ────────────────────────────────────────────
    Cada M steps: actualizar banco de meta-rúbricas
    ────────────────────────────────────────────
      1. Recopilar todas las rúbricas generadas en los últimos M steps
      2. Clusterizar por similitud semántica
      3. Extraer el "tema" de cada cluster → meta-rúbrica
      4. Filtrar: mantener solo meta-rúbricas que aparecen
         en AMBOS dominios (verificable y abierto)
         → Estas son las genuinamente transferibles
      5. Actualizar banco
    
    ================================================================================
    ```
    
    ---
    
    ## ¿Por qué v2 es sustancialmente mejor que v1?
    
    | Aspecto | v1 | v2 | Por qué importa |
    | --- | --- | --- | --- |
    | **Generación de rúbricas** | Desde un CoT solo | Contrastiva (Ĉ⁺ vs Ĉ⁻) | Rúbricas discriminativas por construcción, no por filtro posterior |
    | **Estructura** | Flat (lista de rúbricas) | Jerárquica (meta + instancia) | Transfer explícito: las meta-rúbricas son el vehículo de transferencia |
    | **Anti-saturación** | Fracción de válidas | Zona de máxima información | Más preciso: no solo "válida o no" sino "cuánta señal da" |
    | **Qué aprende el rubricator** | "Generar criterios que correlacionen con correctness" | "Identificar qué distingue buen razonamiento de malo" | Segundo skill es más fundamental y más transferible |
    | **Uso de rollouts** | Cada rollout por separado | Pares contrastivos del mismo batch | Aprovecha mejor la información ya disponible, sin costo extra |
    
    La mejora más impactante es la **generación contrastiva**. Cambia fundamentalmente *qué aprende* el rubricator: pasa de "generar criterios" (habilidad generativa abstracta) a "diagnosticar diferencias entre razonamiento bueno y malo" (habilidad analítica concreta). Eso segundo es exactamente lo que necesitás para que una rúbrica sea útil como reward en RL.