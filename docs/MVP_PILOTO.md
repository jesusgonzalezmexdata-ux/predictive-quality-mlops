# MVP «Dictamen de viabilidad de un modelo de calidad»: protocolo de piloto

**Para quién:** ingeniero de proceso, calidad o equipo de datos que está por invertir en un modelo predictivo de calidad (inyección de plástico, electrónica, farmacéutica, alimentos, semiconductores).
**Qué decide:** si conviene **desplegar** el modelo, **no desplegarlo** o **recolectar más datos**, y cuántos.
**Dónde se usa:** demo de este repositorio → modo «Evalúa tu proceso», o `make dictamen LOTES=lotes.csv`. El archivo se procesa en el navegador; no se envía a ningún servidor.

## 1. Datos

| Columna | Obligatoria | Contenido |
|---|---|---|
| fecha | Sí | Fecha y hora de fabricación del lote |
| resultado | Sí | pasa / falla (acepta OK/NG, conforme/rechazo, 0/1) |
| lote | No | Identificador |
| variables | Sí, al menos 1 | Una columna numérica por variable medida **antes** de conocer el resultado |

Acepta Excel o CSV (coma o punto y coma, coma decimal). Las columnas de texto, constantes o con más de 50 % de vacíos se descartan y se avisan. Mínimo 200 lotes; con menos de 20 fallas no hay nada que evaluar.

## 2. Protocolo (el mismo del caso SECOM)

1. **Partición cronológica:** el último 20 % de los lotes se evalúa una sola vez.
2. **Selección sin ver el futuro:** con ventanas crecientes dentro del pasado se elige entre gradient boosting y regresión logística, y el umbral de menor costo.
3. **Señal:** PR-AUC en lotes futuros con intervalo bootstrap del 95 % contra la tasa de falla (lo que daría el azar).
4. **Dinero:** ahorro contra la mejor alternativa simple (no inspeccionar o inspeccionar todo) con intervalo bootstrap.
5. **Tamaño de muestra:** si el ahorro es positivo pero su intervalo incluye cero, cuántos lotes, fallas y meses harían falta para que lo excluya.
6. **Riesgos:** brecha entre validación al azar y cronológica (el proceso cambia), deriva de variables (PSI), cambio de la tasa de falla (Fisher) y sensibilidad a la razón de costos.

**Dictamen:** DESPLEGAR si hay señal y el intervalo del ahorro está por encima de cero; NO DESPLEGAR si el ahorro es negativo o el modelo no ahorra; RECOLECTAR MÁS DATOS si el ahorro es positivo pero no demostrado y demostrarlo tomaría 24 meses o menos.

## 3. Fases

| Fase | Duración | Actividad | Entregable |
|---|---|---|---|
| 0 · Acuerdo | 1 semana | Exportación de lotes con variables y resultado; costos de scrap, inspección y retrabajo | Tabla en la plantilla |
| 1 · Dictamen | 2 días | Evaluación y revisión de hallazgos con el dueño del proceso | Dictamen de una página firmado |
| 2 · Decisión | 1 semana | El usuario decide y documenta: desplegar en modo sombra, descartar o seguir registrando | Acta de decisión con el dictamen como evidencia |

Si el dictamen es DESPLEGAR, el siguiente paso es modo sombra (4 semanas) y después retener solo los lotes marcados en una línea con otra como control (8 semanas).

## 4. Criterios definidos antes de empezar

**Éxito:** el usuario toma y documenta la decisión con el dictamen.
**Abandono:** menos de 20 fallas en el histórico, o el resultado por lote no se registra de forma confiable.

## 5. Límites

- El puntaje ordena lotes; no es una probabilidad calibrada.
- Una variable medida después de la inspección inflaría todo: el dictamen no puede detectarlo si el usuario la incluye.
- Los costos son supuestos del usuario; la sección de sensibilidad muestra si la conclusión depende de ellos.
