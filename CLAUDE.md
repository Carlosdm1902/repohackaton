# Bloc2026 Grupo 2 — Grip Demand Forecasting

## Contexto
Hackathon de Grip Shipping (Mayo 2026). Predecir volumen diario de envíos por
`(fecha, origin_state, item_sku)` para enero–abril 2026 con datos históricos de 2025.
Métrica oficial: **RMSLE** (menor es mejor). Competencia en Kaggle.

## Flujo completo
```
python limpieza.py   →  data/grip_orders_clean.csv
python modelo.py     →  data/submission_final.csv   ← subir a Kaggle
                     →  data/clusters.csv           ← análisis de clusters
                     →  data/metricas.json          ← métricas de validación
```

## Archivos clave

| Archivo | Rol |
|---|---|
| `limpieza.py` | Limpieza y explosión de `line_items` JSON → CSV limpio |
| `modelo.py` | Pipeline principal: features + LGB + ARIMA + submission |
| `utils.py` | Funciones compartidas reutilizables |
| `train_cv.py` | Validación cruzada walk-forward (4 folds × 30 días) |
| `tuning_optuna.py` | Tuning de hiperparámetros LGB con Optuna |
| `data/grip_orders_clean.csv` | Datos limpios (output de limpieza.py) |
| `data/submission_final.csv` | Predicciones finales para Kaggle |
| `data/clusters.csv` | Análisis de clusters por serie (origen, SKU, perfil) |
| `data/metricas.json` | RMSLE, MAE, RMSE, MAPE e importancia de features |
| `sample_submission.csv` | Plantilla de submission del concurso |

## Arquitectura del modelo

### Clasificación de series por régimen de demanda
| Clase | Criterio | Estrategia |
|---|---|---|
| `zero` | `nonzero_frac < 5%` | Predice 0 (discontinuadas) |
| `intermittent` | `5% – 40%` | 100% LightGBM |
| `regular` | `> 40%` | LGB + ARIMA (blend adaptativo por cluster) |

### PASO 2b: STL Decomposition + Clustering estadístico

#### STL por serie (paralelo, `period=7`, `robust=True`)
Descompone cada serie activa en `trend + seasonal + residual`:
- `seasonal_strength` = `1 − var(residual) / var(seasonal + residual)`
- `trend_strength`    = `1 − var(residual) / var(trend + residual)`
- `autocorr_lag7`     = autocorrelación al lag 7 (fuerza del patrón semanal)
- `skewness`          = asimetría de la distribución de demanda
- `peak_mean_ratio`   = max / (mean + 1), qué tan extremos son los picos
- `atypical_scores`   = `|residuo|/σ` por fecha (permite umbral adaptativo por cluster)
- `atypical_dates`    = fechas donde el score > 2.0 (threshold base)

#### Selección de k óptimo (k=2..12) con 3 métricas
```
Combined = 0.50 × Silhouette + 0.30 × CH_norm + 0.20 × (1 − DB_norm)
```

| Métrica | Qué mide |
|---|---|
| Silhouette | Cohesión intra-cluster vs separación inter-cluster |
| Calinski-Harabász (CH) | Ratio varianza inter / intra — compacidad |
| Davies-Bouldin (DB) | Similitud entre clusters vecinos — separación |

#### Features de clustering (8 dimensiones)
`seasonal_strength`, `trend_strength`, `nonzero_frac`, `CV`, `atypical_frac`,
`autocorr_lag7`, `skewness`, `peak_mean_ratio`

#### Parámetros adaptativos por cluster
Derivados automáticamente del perfil estadístico de cada cluster:

| Parámetro | Fórmula | Rango | Efecto |
|---|---|---|---|
| `arima_blend` | `clip(0.12 + 0.33×max(seasonal_str, autocorr))` | [0.08, 0.45] | Más ARIMA para series periódicas |
| `dow_alpha` | `clip(0.06 + 0.36×seasonal_str)` | [0.04, 0.42] | Empuje DOW más fuerte donde hay patrón semanal |
| `stl_threshold` | `clip(1.5 + 2.0×CV_median)` | [1.5, 4.5] | Menos falsos positivos en series volátiles |
| `clip_mult` | `clip(1.0 + 1.5×CV_median)` | [1.0, 3.5] | Rango de clip más amplio para alta varianza |

### Features de LightGBM (versión actual)

**Calendar básico (9):**
- `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, `day_of_year`
- `is_weekend`, `is_month_end`, `is_month_start`

**Festivos federales EEUU (2):**
- `is_holiday`, `days_to_holiday`

**Calendario comercial completo EEUU (16):**

| Feature | Evento |
|---|---|
| `is_black_friday` / `is_black_friday_week` | Black Friday + semana |
| `is_cyber_monday` / `is_cyber_week` | Cyber Monday + semana |
| `is_valentines_season` | Feb 10-14 |
| `is_mothers_day_season` | Semana previa al 2do domingo de mayo |
| `is_super_bowl_week` | Vie-Dom del Super Bowl |
| `is_easter_week` | Semana Santa |
| `is_xmas_rush` | Dic 10-23 |
| `is_post_holiday_slump` | Ene 1-10 |
| `is_peak_season` | Oct 15 – Dic 31 |
| `is_pre_holiday` / `is_post_holiday_day` | Ventanas alrededor de festivos |
| `days_to_valentines` / `days_to_black_friday` / `days_to_christmas` | Distancias numéricas |

**Fourier semanal (6):**
- `sin_w1`, `cos_w1`, `sin_w2`, `cos_w2`, `sin_w3`, `cos_w3`

**Estadísticas históricas 2025 por serie (8):**

| Feature | Qué representa |
|---|---|
| `month_hist_mean` | Media mensual de 2025 para este mes |
| `month_hist_median` | Mediana mensual 2025 |
| `month_hist_max` | Pico mensual 2025 |
| `month_hist_std` | Desviación estándar mensual 2025 |
| `month_hist_nonzero` | Fracción de días activos por mes |
| `q1_hist_mean` | Media Q1 (Jan-Abr 2025) |
| `q1_hist_nonzero` | Fracción activa en Q1 2025 |
| `q1_hist_total` | Volumen total Q1 2025 |

**Series temporales por grupo (origin_state, sku):**
- Lags: `[1, 2, 3, 7, 14, 21, 28, 56, 91, 182]` días
- Rolling mean, std, max, min en ventanas `[3, 7, 14, 28]` días
- `dow_mean`, `month_mean` — medias históricas sin leakage
- `trend_7`, `volatility_28`

**Features de rango estadístico, varianza y atípicos (13):**

| Feature | Qué captura |
|---|---|
| `roll_range_{3,7,14,28}` | Amplitud max−min por ventana |
| `cv_{3,7,14,28}` | Coeficiente de variación (std/mean) |
| `zscore_lag1_{7,28}` | Qué tan atípico fue el último día |
| `growth_7_28` | Aceleración media 7d vs 28d |
| `growth_14_28` | Aceleración media 14d vs 28d |

**Features de perfil de serie STL/cluster (7):**
- `cluster_enc` — ID del cluster (0..k-1)
- `stl_seasonal_str` — fuerza estacional de la serie
- `stl_trend_str` — fuerza de tendencia
- `stl_autocorr_lag7` — autocorrelación semanal
- `stl_atypical_frac` — fracción de días atípicos históricos
- `stl_skewness` — asimetría de la distribución
- `stl_peak_mean_ratio` — ratio pico/media

**Exógenas del negocio (3):**
- `is_campaign`, `has_discount`, `discount_amount`

### ARIMA
- Modelo: `ARIMA(2,1,1)(1,0,1)[7]`
- Solo para series `regular`
- Blend: **adaptativo por cluster** (rango 0.08–0.45), derivado de `autocorr_lag7` y `seasonal_strength`
- Ajuste paralelo: `ThreadPoolExecutor(max_workers=8)`

### Post-procesamiento estadístico (4 checks en cascada)

Aplicado **después** del ensemble, sin reentrenar el modelo:

```
Check 3  → Evento conocido (holiday/comercial)     → histórico 2025 × growth
Check STL→ |residuo|/σ ≥ stl_threshold (cluster)  → mes limpio 2025 × growth
Check 1  → Clip [Q10, Q90 × clip_mult] MAD-robusto
Check 2  → Empuje DOW con dow_alpha (cluster)
```

**Estadísticas de referencia: nov-dic 2025, MAD-robusto:**
- `μ = mediana` (robusta a outliers)
- `σ = MAD × 1.4826` (equivalente a std bajo distribución normal)
- Bounds: `lo = Q10`, `hi = Q90 × clip_mult`

**Mapeo de atípicos conocidos (Check 3):**
- New Year, MLK, Presidents Day
- Valentine's (Feb 10-14), Super Bowl (Feb 6-8), Easter (Mar 29-Apr 5)
- Post-holiday slump (Ene 1-10)
- YoY growth: `Q4_2025 / Q1_2025` clippeado a [0.5, 2.0]

### Hiperparámetros LGB (Optuna, 20 trials)
```python
learning_rate     = 0.04299
num_leaves        = 194
min_child_samples = 100
feature_fraction  = 0.9962
bagging_fraction  = 0.4055
lambda_l1         = 2.075
lambda_l2         = 0.143
n_estimators      = 2000   (con early stopping, val_days=30)
```

## Historial de scores y experimentos

| Versión | RMSLE local (CV) | Resultado |
|---|---|---|
| LGB baseline + Optuna | **~0.33** | Mejor score local registrado |
| LGB + ARIMA simple | ~0.33 | Sin degradación |
| + 16 eventos comerciales | por medir | — |
| + 8 stats hist. mensuales/Q1 + month_hist_std | por medir | — |
| + Fourier semanal (k=1,2,3) + lag_91 + lag_182 | por medir | — |
| + 13 features estadísticas (rango, CV, z-score, growth) | por medir | — |
| + Post-proc estadístico (checks 1-2-3-STL) | por medir | — |
| + STL clustering multi-métrico (k=2..12) + parámetros adaptativos | por medir | Versión actual |
| LGB dos etapas + SARIMAX + EWM + cross-series | **EMPEORÓ** | Revertido |

## Reglas de trabajo (lecciones aprendidas)

### NO hacer
- **No apilar múltiples cambios simultáneamente** — validar de a uno con CV
- **No usar modelo de dos etapas** (clasificador binario × regresor)
- **No usar SARIMAX** — más lento, no demostró mejora
- **No cambiar hiperparámetros Optuna** sin nueva ronda completa (50-100 trials)

### SÍ hacer
- Agregar **una** feature/cambio → correr → medir RMSLE → decidir
- El ARIMA simple (2,1,1)(1,0,1)[7] funciona, no complicarlo
- Revisar `data/clusters.csv` después de cada corrida para entender los grupos

## Próximas mejoras a validar (de a una)
1. Re-tuning Optuna con el feature set completo (50-100 trials)
2. `lag_365` — captura exactamente el patrón Jan-Abr 2025 → Jan-Abr 2026
3. YoY growth rate como feature estática del LGB (no solo en post-proc)
4. STL residual como feature binaria `is_atypical_day` en entrenamiento

## Git
- Repo: `https://github.com/AMSG-lab/Bloc2026-grupo2`
- Branch: `main`
- Email: `quinteronogueraj@gmail.com`

## Dependencias
```bash
pip install -r requirements.txt
# pandas, numpy, lightgbm, statsmodels, optuna, scikit-learn, gdown
```
