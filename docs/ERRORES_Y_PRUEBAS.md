# Errores encontrados y pruebas de estrés

| Hallazgo | Acción |
|---|---|
| El "ahorro" se medía contra *sin modelo*; con razones de costo altas el modelo parecía ganar solo porque inspeccionar todo es peor | Se compara contra la mejor alternativa simple (`min(sin modelo, inspeccionar todo)`) |
| Un split aleatorio daba PR-AUC 0.154 vs 0.09 temporal | Todo el desarrollo usa ventanas expansivas cronológicas y holdout final; la comparación se publica |
| Imputación, umbral y colinealidad podían ver el futuro | `SensorCleaner` se ajusta solo con entrenamiento de cada pliegue; el umbral se elige con predicciones fuera de muestra |
| Ranking SHAP inestable (solape top-10 de 52 %) | Se reporta la inestabilidad y se retira la narrativa de "aprendió la física" |
| Alias de sensores podrían leerse como reales | Marcados como hipotéticos en YAML, API, dashboard y README |
| Matriz `corr()` de solo lectura en NumPy | `to_numpy(copy=True)` |
| Ventanas de C-MAPSS podían filtrar el futuro | Prueba `test_ventanas_no_usan_el_futuro` altera el futuro y verifica que el pasado no cambia |
| Plazo fijo como baseline de mantenimiento | Se elige mirando el test (favorece al baseline); aun así el modelo cuesta menos |
| PSI con 60 lotes recientes disparó falsa alarma (4 sensores críticos) sin cambio real | El PSI tiene sesgo de muestra pequeña (~bins/n) | `/drift` exige ≥100 predicciones; prueba `test_drift_sin_corrimiento_no_alerta` |
| El umbral elegido en entrenamiento (0.106) costó 77 750 en el holdout; el óptimo a posteriori, 66 000 | Pocas fallas: umbral inestable | Se publica la brecha y la sensibilidad; no se presenta el umbral como "óptimo" |
| El lote de mayor riesgo se debía a un sensor a +111 σ | Valor extremo, probable error de medición | Guarda de calidad de dato en el dashboard: verificar la lectura antes de retener |
| El PR-AUC no cae de forma monótona con ruido o corrimiento | 17 fallas hacen ruidosa la métrica | La deriva se presenta como alerta de operación fuera de lo visto, no como prueba de degradación |

Pruebas: 20 (`pytest`): datos, split cronológico, folds, limpieza sin fuga, costos, umbral, PSI, API (7) y RUL (5) y dashboard (1).
