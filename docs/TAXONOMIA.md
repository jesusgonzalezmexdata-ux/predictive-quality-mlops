# Taxonomía de decisión y de sensores

SECOM no documenta qué mide cada sensor. La taxonomía de decisión es la contribución de diseño: convierte un puntaje de riesgo en una acción trazable a la norma.

## 1. Jerarquía de decisión (del dato a la acción)
```mermaid
flowchart TD
  L[Lote medido por 590 sensores] --> Q{Puntaje del modelo}
  Q -->|>= umbral de costo| R[ALTO RIESGO]
  Q -->|>= umbral/2| A[VIGILAR]
  Q -->|< umbral/2| V[NORMAL]
  R --> R1[Retener + inspección 100 %<br/>ISO 9001 · 8.7 salida no conforme]
  A --> A1[Muestreo reforzado + registro<br/>ISO 9001 · 9.1.3 análisis y evaluación]
  V --> V1[Liberar flujo habitual]
  R1 --> C[Costo: inspección + retrabajo o scrap evitado]
  subgraph Proceso
    D[Deriva PSI>0.25 en >20 % de sensores] --> D1[Reentrenar + revisar instrumentos<br/>ISO 9001 · 7.1.5]
    T[Tasa mensual > 2x la media] --> T1[Causa raíz + acción correctiva<br/>ISO 9001 · 10.2]
  end
```
**Por qué así:** el umbral de alarma sale de minimizar el costo (no de maximizar F1); la banda "vigilar" evita decisiones binarias donde el modelo es débil; las reglas de proceso protegen frente a la no estacionariedad observada.

## 2. Familias de sensores (hipotéticas)
Los alias de `config/sensor_aliases.yaml` se agrupan así para mostrar cómo se organizaría el diccionario real. **Son ilustrativos, no mediciones reales.**
```mermaid
flowchart LR
  P[Proceso de oblea] --> Vac[Vacío y presión]:::h --> a1[f59 Presion_camara_vacio_2]
  Vac --> a2[f31 Presion_linea_nitrogeno]
  P --> Ter[Térmico]:::h --> b1[f460 Temperatura_zona_horno_2]
  Ter --> b2[f130 Temperatura_agua_enfriamiento]
  P --> Ele[Eléctrico y RF]:::h --> c1[f197 Potencia_RF_electrodo]
  Ele --> c2[f247 Voltaje_electrodo_superior]
  Ele --> c3[f0 Corriente_motor_transferencia]
  P --> Flu[Gases]:::h --> d1[f33 Flujo_gas_proceso_A]
  P --> Amb[Ambiente y mecánica]:::h --> e1[f10 Humedad_sala_limpia]
  Amb --> e2[f19 Vibracion_bomba_vacio]
  classDef h fill:#eef,stroke:#446
```

## 3. Diccionario de datos
| Campo | Descripción |
|---|---|
| `f0..f589` | Sensores anónimos de SECOM (valores continuos, con faltantes) |
| `y` | 1 = lote falla (104), 0 = pasa (1 463) |
| `timestamp` | Fecha del lote (19-jul a 17-oct 2008); define el orden de validación |
| `p` | Puntaje del modelo (LightGBM con `scale_pos_weight`; **no calibrado**) |
| Costos | Supuestos editables, no provienen de los datos |
C-MAPSS: `motor, ciclo, op1-3, s1..s21`; RUL = ciclos hasta la falla (tope 125 en entrenamiento).
