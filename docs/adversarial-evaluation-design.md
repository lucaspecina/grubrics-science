# Diseño experimental: Evaluación Adversarial (borrador v1, 2026-07-02)

**Estado: BORRADOR bajo revisión adversarial.** El "cómo" del reframing verificado
(`adversarial-evaluation-reframing.md` — RQs en §4, definiciones en §8c-8g, mapa en §9).
Este doc se ataca con revisores adversariales antes de congelarse.

---

## 1. La arquitectura del experimento: DOS instrumentos, no uno

La decisión de diseño más importante. Separamos la medición en dos instrumentos que se
validan mutuamente:

### E-I — Las curvas (el instrumento caro y lento)
Policy entrenándose con GRPO contra rúbrica+judge; medimos proxy(t) y gold(t) y el quiebre,
por (tier × defensor × seed). Pocas celdas, es el resultado headline.

### E-II — El probing con atacante (el instrumento barato y escalable)
SIN policy: el Tramposo ataca directamente una configuración (rúbrica + defensor). Medimos
**attack-cost-to-break**: cuántos pasos/samples de optimización del atacante hasta producir
respuestas que superan el percentil-X de honestas bajo el proxy mientras el gold las rechaza.
Barato por configuración → escala a muchos evaluadores → es el corazón del benchmark vivo.

### El pilar: cross-validación E-II ↔ E-I
¿El attack-cost-to-break (barato) PREDICE el onset de las curvas (caro)? Si correlaciona
alto: **la métrica barata queda validada contra la cara** → cualquier equipo puede medir la
robustez de su evaluador sin correr RL completo. Ese es el producto-métrica del paper. Si no
correlaciona: hallazgo igual (la presión de policy y la de atacante dedicado rompen distinto).

## 2. Setup común (fijo entre brazos — todo lo que no varía)

| Componente | Decisión | Justificación |
|---|---|---|
| Policy | **Qwen3-4B** | CHERRL demuestra hacking-onset a 4B (steps 68-478); mitad de costo que 8B → el presupuesto compra 2× celdas/seeds. 8B solo si 4B no hackea en nuestros pisos |
| RL | GRPO / veRL 0.7.1 | Nuestro stack validado; compatible CHERRL |
| Judge (aplicador) | **LOCAL fijo** (candidato: Qwen3-32B-instruct o el mejor local que valide) — MISMO en todos los brazos y pisos | Curvas requieren cientos de steps × batch × criterios = millones de calls; API inviable. Validación: acuerdo local-vs-GPT-4.1-binario medido offline con nuestra maquinaria ANTES de correr (kill si kappa<0.35). En Tier S el judge se cancela por diseño (proxy y gold usan el MISMO judge) |
| Prompts | **Split train/eval por pregunta**: la policy entrena en P_train; las curvas gold se miden en P_eval held-out; el defensor SOLO ve rollouts de P_train | Anti-leakage (falla #1 cazada) |
| Rúbrica inicial | IDÉNTICA entre brazos por pregunta (S: gold-con-agujeros; V: generada por frontier y congelada como semilla; A: humana o G1) | Los defensores difieren solo en las ACTUALIZACIONES |
| Cadencia del defensor | mismo N (steps) para todos los brazos activos | Fairness; el costo/latencia por regeneración se reporta como eje secundario (ventaja del 8B local) |
| Seeds | ≥2 por celda primaria; ≥3 en la comparación primaria | Prespecificado abajo |

## 3. La matriz y su priorización presupuestaria

Defensores: **D0** nada · **D1** frontier prompteado (receta OnlineRubrics: mira K rollouts
recientes, propone/edita criterios) · **D2** 8B entrenado (señal funcional; mismo input que D1)
· **D4** humana congelada (solo tier A; es D0 con rúbrica experta).

| Orden | Celdas | Por qué primero | Costo est. |
|---|---|---|---|
| 1. **Tier S completo** | S × {D0,D1,D2} × 3 seeds = 9 runs | Mecanismo con atribución perfecta; judge se cancela; corto (onset esperado <500 steps); TODO local (judge local, gold local) | ~9×3-5 GPU-h spot ≈ $80-150 |
| 2. **Tier V núcleo** | V × {D0,D1,D2} × 2 seeds = 6 runs + calibración panel (§8g) | El claim duro (gold real); MedQA/GSM8K adapters listos | ~6×4-6 GPU-h + panel eval |
| 3. **E-II atacante** | Tramposo vs ~6-10 configuraciones | La métrica + cross-validación con onsets ya medidos en 1-2 | GPU attacker training + panel-split evals |
| 4. **Tier A confirmatorio** | A × {D4,D1,D2} × 1-2 seeds | Validez externa; SOLO si 1-3 dieron señal | El más caro por panel |

**La comparación primaria prespecificada**: onset(D2) vs onset(D1) en Tier V (2 colas,
seeds≥3 si el efecto es chico). Todo lo demás: secundario/exploratorio, se declara así.

## 4. Formalizaciones (prespecificadas, anti p-hacking)

- **Curvas**: proxy(t) = media del reward rúbrica en train; gold(t) = métrica gold del tier
  sobre rollouts frescos de P_eval, cada K steps, decoding fijo (temp, seeds de eval fijos).
- **Onset (primario)**: el menor t tal que gold(t) < max_{s≤t} gold(s) − δ sostenido durante
  m evals consecutivos, MIENTRAS la pendiente de proxy en la ventana es > 0. δ y m se fijan
  en el piloto de Tier S ANTES de ver las celdas comparativas (δ candidato: 1 desvío estándar
  de gold entre seeds en régimen estable; m=3).
- **Divergencia integral (secundaria, continua)**: ∫ [proxỹ(t) − gold̃(t)] dt con ambas
  curvas normalizadas a [0,1] por su rango pre-onset — captura magnitud, no solo momento.
- **La pareja (onset, gold-at-plateau)**: OBLIGATORIA en toda tabla. Cazafallas #3: un
  defensor "trivialmente duro" (rúbrica imposible → nadie hackea pero nadie aprende) muestra
  onset=∞ con gold-at-plateau bajo → se ve, no se premia.
- **attack-cost-to-break (E-II)**: pasos de GRPO del atacante hasta que ≥ p% de sus outputs
  superan la mediana proxy de honestas con gold bajo umbral. Reportar curva completa
  cost-vs-%break, no solo el punto.

## 5. Los defensores, contrato exacto

Contrato común (de 8d): cada N steps reciben (pregunta, rúbrica vigente, muestra de M
rollouts recientes de P_train) → emiten rúbrica nueva. NUNCA ven gold/panel/etiquetas.

- **D1 (frontier prompteado)**: GPT-4.1 con prompt fijo pre-registrado (receta OnlineRubrics
  adaptada: identificar patrones sospechosos en los rollouts, proponer criterios nuevos y
  penalizaciones, PRESERVANDO discriminación sobre respuestas legítimas). Sin iterar el
  prompt después de ver resultados (anti-garden-of-forking-paths); el prompt se congela tras
  el piloto S.
- **D2 (entrenado)**: el rubricator 8B. Input idéntico a D1. Entrenamiento: DPO con señal
  funcional sobre ESCENAS DE REVISIÓN construidas en S/V (estado: rúbrica con agujero +
  rollouts que lo explotan; candidatas de revisión puntuadas funcionalmente contra el gold
  del tier de entrenamiento — que en S/V es legítimo usar porque es el tier de entrenamiento
  del defensor, no de su evaluación). **D2 se entrena UNA VEZ, se congela, y se evalúa en
  pisos/preguntas que no vio** — la adaptación en despliegue es solo in-context.
- **Restricción anti-trivialidad (ambos)**: el objetivo/prompt incluye preservar
  discriminación entre honestas (info_value de nuestra maquinaria); la pareja de métricas
  de §4 la audita.

## 6. E-II: el protocolo del atacante

- Atacante: Qwen3-4B/8B entrenado con GRPO; reward = score_proxy − λ·score_panel_A (recета
  Adv-RM adaptada a rúbricas), sobre preguntas de P_train del tier.
- **Panel splitting (8e)**: panel_A (señal del atacante) y panel_B (medición) de familias
  distintas; auditoría humana de muestra de outputs "exitosos".
- Cold start: nuestras 4 familias de hacks como SFT/few-shot inicial.
- **Transfer tests (RQ4)**: atacante entrenado contra config X, evaluado contra Y (otra
  rúbrica, otro defensor, otro tier) — ¿los exploits generalizan o son config-específicos?
  (Define si el benchmark necesita re-entrenar el atacante por evaluador o si un atacante
  general sirve.)

## 7. Fallas cazadas en v1 y sus fixes (registro vivo)

| # | Falla | Fix en el diseño |
|---|---|---|
| 1 | Defensor ve rollouts de preguntas de eval → contamina gold | Split P_train/P_eval; defensor solo ve P_train (§2) |
| 2 | Regeneración cambia la escala del proxy → curvas incomparables | Quiebre definido sobre GOLD; proxy solo exige pendiente>0 intra-época de rúbrica; normalización por época |
| 3 | Defensa trivialmente dura (nadie aprueba) parece "robusta" | Pareja (onset, gold-at-plateau) obligatoria + restricción de discriminación (§5) |
| 4 | Inestabilidad RL por reward no-estacionario se confunde con colapso-de-alineación | Diagnósticos prespecificados (KL, entropía, longitud) por run; "colapso" solo se declara si gold cae CON policy estable; N grande y cambios graduales |
| 5 | Judge débil local mete sus propios huecos → atribución sucia | Judge IDÉNTICO entre brazos (se cancela en comparaciones); validado offline vs GPT-4.1; en S se cancela por construcción |
| 6 | El atacante aprende los puntos ciegos del panel de medición | Panel splitting + auditoría humana (8e) |
| 7 | Multiple comparisons entre 9-20 celdas | Comparación primaria única prespecificada (§3); resto declarado exploratorio |
| 8 | D1 con prompt tuneado post-hoc le regala ventaja/desventaja | Prompt de D1 congelado tras piloto, pre-registrado en el repo |
| 9 | Onset estimator elegido mirando los datos | δ, m fijados en piloto S antes de las celdas comparativas |
| 10 | "Rúbrica inicial distinta entre brazos" confunde defensor con semilla | Rúbrica inicial idéntica por pregunta en todos los brazos (§2) |

## 8. Fases operativas

- **α (ya en curso)**: motor Fase 0 (bloque 2 GPU pendiente) → ¿la señal funcional mueve al
  8B? Insumo directo de D2.
- **β**: validar judge local vs GPT-4.1 (offline, con maquinaria existente) + adaptar/absorber
  testbed CHERRL + piloto S (1 celda D0) para fijar δ, m, N, M → **congelar el diseño v2**.
- **γ**: Tier S completo (9 runs) → primer resultado publicable (mecanismo).
- **δ**: Tier V núcleo + calibración del panel (8g).
- **ε**: E-II atacante + cross-validación métrica.
- **ζ**: Tier A confirmatorio + paper + benchmark release.

Kill criteria por fase: β — si ningún judge local alcanza kappa≥0.35 vs GPT-4.1, replantear
(¿API con eval espaciado? ¿tier S only?). γ — si D0 no muestra hacking en S (sin quiebre),
los agujeros plantados son demasiado benignos → rediseñar agujeros antes de seguir. δ — si
el orden de defensores en V contradice S, investigar antes de A (es hallazgo, no fracaso).

## 9. Presupuesto global estimado (draft)

GPU (spot 2×H100): S ~$80-150 · V ~$100-180 · E-II ~$60-120 · A ~$100-200 → **~$350-650 GPU**.
API: panel de A + calibración + D1 regeneraciones + validación judge → **~$150-300**.
Total draft: **~$500-950**, escalonado con kill criteria por fase. (La decisión de 4B y el
judge local son lo que baja esto de ~$3K a ~$700.)
