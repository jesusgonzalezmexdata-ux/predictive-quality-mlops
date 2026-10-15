# Model Card · SECOM lot-risk model

| | |
|---|---|
| Task | Rank semiconductor lots by risk of failing final test, and map the score to release / monitor / hold |
| Model | LightGBM with `scale_pos_weight` = 13.4 after a `SensorCleaner` fitted on training data only (constants, >50 % missing, KNN imputation, \|r\| > 0.95) |
| Threshold | Minimises expected cost on temporal out-of-fold training predictions (0.106) |
| Data | UCI SECOM: 1,567 lots, 590 anonymous sensors, 104 failures (6.6 %), Jul–Oct 2008 |
| Validation | Chronological split; expanding-window CV inside training; last 313 lots (17 failures) evaluated once |
| Author | Jesús R. L. González |

## Performance (chronological holdout)

| Metric | Value |
|---|---|
| PR-AUC | 0.089 (95 % CI 0.043–0.197); prevalence 0.054 |
| ROC-AUC | 0.62 |
| Lift in the top 10 % | 2.4× |
| Failures detected / false alarms | 5 of 17 / 56 |
| Saving vs best simple alternative | 7,250 MXN, 95 % CI −22,031 to +19,769; P(saving > 0) = 55 % |

## Intended use

- Prioritising lots for tightened sampling, with a human decision on every hold.
- Teaching leakage-free validation and cost-based thresholds on weak signals.

## Out of scope

- Automatic release or rejection of lots.
- Root-cause claims: sensor names are anonymous and the SHAP ranking is unstable (~52 % of the top 10 survives retraining).
- Monetary savings in a real fab: costs are assumptions in `src/pqm/config.py`.

## Limitations and risks

- **Weak signal, few failures:** the economic benefit is not statistically demonstrated and depends on the cost ratio; at ratios 5, 10 and 80 the model loses to the best simple rule.
- **Non-stationary process:** failure rate fell from 22 % to 3 % over the period; 142 of 273 sensors drift (PSI > 0.25) between training and holdout.
- **Not calibrated:** the score is not a probability; the dashboard shows observed failure rates of similar lots instead.
- **Drift alerts** need at least 100 recent predictions; PSI on small samples raises false alarms.
- **Data-quality guard:** readings beyond 10 σ must be verified before holding a lot.

## Evidence

[README](README.md) · [docs/ERRORES_Y_PRUEBAS.md](docs/ERRORES_Y_PRUEBAS.md) · `make train` and `make evidence` reproduce the figures.
