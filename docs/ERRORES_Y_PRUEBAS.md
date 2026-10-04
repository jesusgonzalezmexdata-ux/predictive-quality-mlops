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

Pruebas: 18 (`pytest`): datos, split cronológico, folds, limpieza sin fuga, costos, umbral, PSI, API (6) y RUL (5).
