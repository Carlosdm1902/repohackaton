"""
Pipeline de forecasting: LightGBM + ARIMA ensemble.

Base: version LGB + ARIMA que dio RMSLE ~0.33 en CV (hiperparametros Optuna).
Mejora unica de esta version: calendario comercial completo de EEUU.

Eventos nuevos (16 features):
  - Black Friday day / semana Thanksgiving
  - Cyber Monday / semana Cyber
  - Valentine's season (Feb 10-14)      → DTC perishables spike
  - Mother's Day season (semana previa)  → flores, perishables
  - Super Bowl week                      → comida a domicilio
  - Easter week                          → perishables, primavera
  - Christmas rush (Dec 10-23)          → peak shipping
  - Post-holiday slump (Ene 1-10)       → caida post-fiestas
  - Peak season (Oct 15-Dic 31)         → temporada alta
  - Pre-holiday window (1-3 dias antes de cualquier festivo)
  - Post-holiday window (1-2 dias despues de festivo)
  - days_to_valentines / days_to_black_friday / days_to_christmas
"""

import warnings
warnings.filterwarnings("ignore")

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA

from utils import rmsle as compute_rmsle

# ── Festivos federales EEUU 2025-2026 ─────────────────────────────────────────
US_HOLIDAYS = pd.to_datetime([
    "2025-01-01","2025-01-20","2025-02-17","2025-05-26","2025-06-19",
    "2025-07-04","2025-09-01","2025-10-13","2025-11-11","2025-11-27",
    "2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-05-25","2026-06-19",
    "2026-07-04","2026-09-07","2026-10-12","2026-11-11","2026-11-26",
    "2026-12-25",
])
US_HOLIDAYS_SET = set(US_HOLIDAYS)

# ── Calendario comercial completo EEUU ────────────────────────────────────────
#
# Fechas clave que mueven la demanda en eCommerce DTC / cold-chain logistics
#
# Black Friday (viernes despues del 4to jueves de noviembre)
BLACK_FRIDAY_DATES  = pd.to_datetime(["2025-11-28", "2026-11-27"])
BLACK_FRIDAY_SET    = set(BLACK_FRIDAY_DATES)

# Semana de Thanksgiving (lunes-viernes = semana de Black Friday)
# 2025: Nov 24-28 | 2026: Nov 23-27
BLACK_FRIDAY_WEEK_SET = set(pd.DatetimeIndex([
    "2025-11-24","2025-11-25","2025-11-26","2025-11-27","2025-11-28",
    "2026-11-23","2026-11-24","2026-11-25","2026-11-26","2026-11-27",
]))

# Cyber Monday (lunes post-Black Friday)
CYBER_MONDAY_DATES = pd.to_datetime(["2025-12-01", "2026-11-30"])
CYBER_MONDAY_SET   = set(CYBER_MONDAY_DATES)

# Semana Cyber (lunes-miercoles post-Black Friday)
CYBER_WEEK_SET = set(pd.DatetimeIndex([
    "2025-12-01","2025-12-02","2025-12-03",
    "2026-11-30","2026-12-01","2026-12-02",
]))

# Valentine's season: Feb 10-14 (pedidos previos al 14 disparan envios)
VALENTINES_DATES = pd.to_datetime(["2025-02-14", "2026-02-14"])
VALENTINES_SEASON_SET = set(pd.DatetimeIndex(
    [f"2025-02-{d:02d}" for d in range(10, 15)] +
    [f"2026-02-{d:02d}" for d in range(10, 15)]
))

# Mother's Day: 2do domingo de mayo (semana previa = peak pedidos)
# 2025: May 11 → semana May 5-11 | 2026: May 10 → semana May 4-10
MOTHERS_DAY_SEASON_SET = set(pd.DatetimeIndex(
    [f"2025-05-{d:02d}" for d in range(5, 12)] +
    [f"2026-05-{d:02d}" for d in range(4, 11)]
))

# Super Bowl Sunday (1er domingo de febrero aprox.)
# Super Bowl LIX: Feb 9, 2025 | Super Bowl LX: Feb 8, 2026
# Semana previa: Vie-Dom
SUPER_BOWL_WEEK_SET = set(pd.DatetimeIndex([
    "2025-02-07","2025-02-08","2025-02-09",
    "2026-02-06","2026-02-07","2026-02-08",
]))

# Easter: domingos de Pascua + semana previa (Semana Santa)
# 2025: Easter Apr 20 → semana Apr 13-20
# 2026: Easter Apr 5  → semana Mar 29-Apr 5
EASTER_WEEK_SET = set(
    list(pd.date_range("2025-04-13", "2025-04-20", freq="D")) +
    list(pd.date_range("2026-03-29", "2026-04-05", freq="D"))
)

# Christmas rush: Dec 10-23 (ventana critica de envios navidenos)
# Post-holiday slump: Ene 1-10 (caida brusca post-fiestas)
# Peak season: Oct 15 - Dic 31

# Pre-holiday window: 1-3 dias antes de cualquier festivo federal
PRE_HOLIDAY_SET = set()
for h in US_HOLIDAYS:
    for d in range(1, 4):
        PRE_HOLIDAY_SET.add(h - pd.Timedelta(days=d))

# Post-holiday window: 1-2 dias despues de festivo federal
POST_HOLIDAY_SET = set()
for h in US_HOLIDAYS:
    for d in range(1, 3):
        POST_HOLIDAY_SET.add(h + pd.Timedelta(days=d))

# Navidad y referencias para distancias
CHRISTMAS_DATES = pd.to_datetime(["2025-12-25", "2026-12-25"])

# ── Config ────────────────────────────────────────────────────────────────────
CLEAN_CSV      = "data/grip_orders_clean.csv"
SUBMISSION_CSV = "sample_submission.csv"
OUTPUT_CSV     = "data/submission_final.csv"

LAG_DAYS     = [1, 2, 3, 7, 14, 21, 28, 56]
ROLLING_WINS = [3, 7, 14, 28]
VAL_DAYS     = 90   # horizonte largo para detectar colapso de lags recursivos

REGULAR_THRESHOLD = 0.40
ZERO_THRESHOLD    = 0.05
N_CLUSTERS        = 12

ARIMA_BLEND       = 0.30
ARIMA_MAX_WORKERS = 8

FLOOR_RATIO = 0.25   # piso: 25% de la media historica mensual
CAP_RATIO   = 5.0    # techo: 5x el maximo historico mensual

LGB_PARAMS = {
    "objective"        : "regression",
    "metric"           : "rmse",
    "learning_rate"    : 0.042987099962655866,
    "num_leaves"       : 194,
    "min_child_samples": 100,
    "feature_fraction" : 0.9962411422735371,
    "bagging_fraction" : 0.4055366014266707,
    "bagging_freq"     : 1,
    "lambda_l1"        : 2.074955758568211,
    "lambda_l2"        : 0.14281808621478395,
    "n_estimators"     : 2000,
    "verbose"          : -1,
    "n_jobs"           : -1,
    "random_state"     : 42,
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def commercial_features_for_date(d):
    """Devuelve dict con todas las features comerciales para una fecha dada."""
    month, day, dow = d.month, d.day, d.dayofweek
    return {
        # ── Eventos puntuales ──────────────────────────────────────────────────
        "is_black_friday"      : int(d in BLACK_FRIDAY_SET),
        "is_black_friday_week" : int(d in BLACK_FRIDAY_WEEK_SET),
        "is_cyber_monday"      : int(d in CYBER_MONDAY_SET),
        "is_cyber_week"        : int(d in CYBER_WEEK_SET),
        "is_valentines_season" : int(d in VALENTINES_SEASON_SET),
        "is_mothers_day_season": int(d in MOTHERS_DAY_SEASON_SET),
        "is_super_bowl_week"   : int(d in SUPER_BOWL_WEEK_SET),
        "is_easter_week"       : int(d in EASTER_WEEK_SET),
        # ── Ventanas de temporada ──────────────────────────────────────────────
        "is_xmas_rush"         : int(month == 12 and 10 <= day <= 23),
        "is_post_holiday_slump": int(month == 1  and day <= 10),
        "is_peak_season"       : int((month == 10 and day >= 15) or month in [11, 12]),
        # ── Ventanas alrededor de festivos federales ───────────────────────────
        "is_pre_holiday"       : int(d in PRE_HOLIDAY_SET),
        "is_post_holiday_day"  : int(d in POST_HOLIDAY_SET),
        # ── Distancias numericas a eventos clave ──────────────────────────────
        "days_to_valentines"   : min(abs((d - v).days) for v in VALENTINES_DATES),
        "days_to_black_friday" : min(abs((d - bf).days) for bf in BLACK_FRIDAY_DATES),
        "days_to_christmas"    : min(abs((d - x).days) for x in CHRISTMAS_DATES),
    }

COMMERCIAL_COLS = list(commercial_features_for_date(pd.Timestamp("2025-01-01")).keys())

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y AGREGACION
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PASO 1: Cargando y agregando datos...")
print("=" * 65)


# Leer en chunks para no superar la RAM disponible
_cols = ["date","origin_state","sku","quantity","is_campaign","has_discount","discount_amount"]
_available = pd.read_csv(CLEAN_CSV, nrows=0).columns.tolist()
_read_cols = [c for c in _cols if c in _available]

_parts = []
for _chunk in pd.read_csv(CLEAN_CSV, usecols=_read_cols, chunksize=100_000, low_memory=False):
    for c in _cols:
        if c not in _chunk.columns:
            _chunk[c] = 0
    _chunk["date"] = pd.to_datetime(_chunk["date"], utc=True).dt.tz_localize(None).dt.normalize()
    _parts.append(
        _chunk.groupby(["date","origin_state","sku"]).agg(
            total_quantity  = ("quantity",        "sum"),
            is_campaign     = ("is_campaign",     "max"),
            has_discount    = ("has_discount",    "max"),
            discount_sum    = ("discount_amount", "sum"),
            discount_cnt    = ("discount_amount", "count"),
        ).reset_index()
    )

agg = pd.concat(_parts, ignore_index=True)
del _parts
agg = agg.groupby(["date","origin_state","sku"]).agg(
    total_quantity  = ("total_quantity", "sum"),
    is_campaign     = ("is_campaign",   "max"),
    has_discount    = ("has_discount",  "max"),
    discount_sum    = ("discount_sum",  "sum"),
    discount_cnt    = ("discount_cnt",  "sum"),
).reset_index()
agg["discount_amount"] = agg["discount_sum"] / agg["discount_cnt"].clip(lower=1)
agg = agg.drop(columns=["discount_sum","discount_cnt"])

print(f"  Series unicas: {agg[['origin_state','sku']].drop_duplicates().shape[0]:,}")
print(f"  Rango: {agg['date'].min().date()} -> {agg['date'].max().date()}")

# Grid completo (dias sin ventas = 0)
pairs = agg[["origin_state","sku"]].drop_duplicates()
dates = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
grid  = pairs.merge(pd.DataFrame({"date": dates}), how="cross")
agg   = grid.merge(agg, on=["date","origin_state","sku"], how="left")
del grid  # libera RAM
agg["total_quantity"]  = agg["total_quantity"].fillna(0).astype(int)
agg["is_campaign"]     = agg["is_campaign"].fillna(0).astype(int)
agg["has_discount"]    = agg["has_discount"].fillna(0).astype(int)
agg["discount_amount"] = agg["discount_amount"].fillna(0.0)

agg = agg.sort_values(["origin_state","sku","date"]).reset_index(drop=True)
print(f"  Grid completo: {len(agg):,} filas")

# Agregar columna 'month' para estadísticas históricas
agg["month"] = agg["date"].dt.month

# ── Estadisticas historicas por mes (todos los meses de 2025) ─────────────────
# Le dan al modelo la "memoria anual": cuanto vende cada serie en enero, febrero, etc.
# Para prediccion de Jan-Abr 2026, estos valores reflejan Jan-Abr 2025.
print("  Calculando estadisticas historicas por mes (2025)...")
monthly_stats = (
    agg.groupby(["origin_state","sku","month"])["total_quantity"]
    .agg(
        month_hist_mean   = "mean",
        month_hist_median = "median",
        month_hist_max    = "max",
        month_hist_nonzero= lambda x: (x > 0).mean(),
    )
    .reset_index()
)
agg = agg.merge(monthly_stats, on=["origin_state","sku","month"], how="left")
for c in ["month_hist_mean","month_hist_median","month_hist_max","month_hist_nonzero"]:
    agg[c] = agg[c].fillna(0)

# Estadisticas de Q1 (Jan-Abr 2025) — contexto directo del periodo a predecir
q1_stats = (
    agg[agg["month"].isin([1,2,3,4])]
    .groupby(["origin_state","sku"])["total_quantity"]
    .agg(
        q1_hist_mean   = "mean",
        q1_hist_nonzero= lambda x: (x > 0).mean(),
        q1_hist_total  = "sum",
    )
    .reset_index()
)
agg = agg.merge(q1_stats, on=["origin_state","sku"], how="left")
for c in ["q1_hist_mean","q1_hist_nonzero","q1_hist_total"]:
    agg[c] = agg[c].fillna(0)

HIST_MONTH_COLS = ["month_hist_mean","month_hist_median","month_hist_max","month_hist_nonzero"]
HIST_Q1_COLS   = ["q1_hist_mean","q1_hist_nonzero","q1_hist_total"]
print(f"    Nuevas features historicas: {HIST_MONTH_COLS + HIST_Q1_COLS}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLASIFICACION DE SERIES
# ═══════════════════════════════════════════════════════════════════════════════
print("\nClasificando series (zero / intermittent / regular)...")
series_stats = (
    agg.groupby(["origin_state","sku"])["total_quantity"]
    .agg(nonzero_frac=lambda x: (x > 0).mean(), total_vol="sum")
    .reset_index()
)

def _classify(row):
    if row["nonzero_frac"] < ZERO_THRESHOLD:    return "zero"
    if row["nonzero_frac"] < REGULAR_THRESHOLD: return "intermittent"
    return "regular"

series_stats["demand_type"] = series_stats.apply(_classify, axis=1)
demand_type_map = series_stats.set_index(["origin_state","sku"])["demand_type"].to_dict()
print(series_stats["demand_type"].value_counts().to_string())

agg = agg.merge(series_stats[["origin_state","sku","demand_type"]], on=["origin_state","sku"], how="left")
DEMAND_ENC = {"zero": 0, "intermittent": 1, "regular": 2}

# ── CLUSTERING DE SERIES (K-Means) ────────────────────────────────────────────
print(f"\nClusterizando series en {N_CLUSTERS} grupos por perfil de demanda...")
_clust = (
    agg.groupby(["origin_state","sku"])["total_quantity"]
    .agg(clust_mean="mean", clust_std="std", clust_nonzero=lambda x: (x > 0).mean())
    .reset_index()
)
_clust["clust_cv"] = (_clust["clust_std"] / _clust["clust_mean"].clip(lower=1e-3)).fillna(0)

_dow = (
    agg.assign(is_wknd=(agg["date"].dt.dayofweek >= 5).astype(int))
    .groupby(["origin_state","sku","is_wknd"])["total_quantity"]
    .mean().unstack(fill_value=0)
)
_dow.columns = ["weekday_mean","weekend_mean"]
_dow = _dow.reset_index()
_dow["wknd_ratio"] = (
    _dow["weekend_mean"] / _dow[["weekday_mean","weekend_mean"]].max(axis=1).clip(lower=1e-3)
).fillna(0)
_clust = _clust.merge(_dow[["origin_state","sku","wknd_ratio"]], on=["origin_state","sku"], how="left").fillna(0)

_feat_k = ["clust_mean","clust_std","clust_cv","clust_nonzero","wknd_ratio"]
_X_k = StandardScaler().fit_transform(_clust[_feat_k].fillna(0))
_clust["cluster_id"] = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(_X_k)
cluster_map = _clust.set_index(["origin_state","sku"])["cluster_id"].to_dict()
agg["cluster_id"] = [cluster_map.get((s, sk), 0) for s, sk in zip(agg["origin_state"], agg["sku"])]
print(_clust["cluster_id"].value_counts().sort_index().rename("series_por_cluster").to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. INGENIERIA DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 2: Ingenieria de features...")
print("=" * 65)

# Encodings
sub_tmp = pd.read_csv(SUBMISSION_CSV)
sub_ext = sub_tmp["id"].str.extract(r"^(\d{4}-\d{2}-\d{2})_([A-Za-z]{2,})_(SKU[-_][\w-]+)$")
all_states = pd.concat([agg["origin_state"], sub_ext[1].dropna()]).unique()
all_skus   = pd.concat([agg["sku"],          sub_ext[2].dropna()]).unique()

le_state = LabelEncoder().fit(all_states)
le_sku   = LabelEncoder().fit(all_skus)
agg["state_enc"]       = le_state.transform(agg["origin_state"])
agg["sku_enc"]         = le_sku.transform(agg["sku"])
agg["demand_type_enc"] = agg["demand_type"].map(DEMAND_ENC)

# Calendar basico
agg["day_of_week"]    = agg["date"].dt.dayofweek
agg["day_of_month"]   = agg["date"].dt.day
agg["week_of_year"]   = agg["date"].dt.isocalendar().week.astype(int)
agg["month"]          = agg["date"].dt.month
agg["quarter"]        = agg["date"].dt.quarter
agg["day_of_year"]    = agg["date"].dt.dayofyear
agg["is_weekend"]     = (agg["day_of_week"] >= 5).astype(int)
agg["is_month_end"]   = agg["date"].dt.is_month_end.astype(int)
agg["is_month_start"] = agg["date"].dt.is_month_start.astype(int)

# Festivos federales
agg["is_holiday"]     = agg["date"].isin(US_HOLIDAYS_SET).astype(int)
agg["days_to_holiday"]= agg["date"].apply(
    lambda d: min(abs((d - h).days) for h in US_HOLIDAYS)
)

# ── CALENDARIO COMERCIAL ───────────────────────────────────────────────────────
print("  Agregando calendario comercial EEUU (16 features)...")
comm_df = pd.DataFrame(
    [commercial_features_for_date(d) for d in agg["date"]],
    index=agg.index
)
agg = pd.concat([agg, comm_df], axis=1)
print(f"    Eventos cubiertos: {COMMERCIAL_COLS}")


# Lags y rolling
print(f"  Lags: {LAG_DAYS} + lag_365 + lag_182")
grp = agg.groupby(["origin_state","sku"])["total_quantity"]
for lag in LAG_DAYS:
    agg[f"lag_{lag}"] = grp.shift(lag)
# Lags anuales y semestrales
agg["lag_365"] = grp.shift(365)
agg["lag_182"] = grp.shift(182)

print(f"  Rolling windows: {ROLLING_WINS}")
for win in ROLLING_WINS:
    shifted = grp.shift(1)
    agg[f"roll_mean_{win}"] = shifted.transform(lambda x: x.rolling(win, min_periods=1).mean())
    agg[f"roll_std_{win}"]  = shifted.transform(lambda x: x.rolling(win, min_periods=1).std().fillna(0))
    agg[f"roll_max_{win}"]  = shifted.transform(lambda x: x.rolling(win, min_periods=1).max())

# Medias historicas sin leakage
agg["dow_mean"] = agg.groupby(["origin_state","sku","day_of_week"])["total_quantity"].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean().fillna(0)
)
agg["month_mean"] = agg.groupby(["origin_state","sku","month"])["total_quantity"].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean().fillna(0)
)
agg["trend_7"] = grp.shift(1).transform(
    lambda x: x.rolling(7, min_periods=2).mean().diff().fillna(0)
)
agg["volatility_28"] = grp.shift(1).transform(
    lambda x: x.rolling(28, min_periods=1).std().fillna(0)
)


FEATURE_COLS = (
    ["state_enc","sku_enc","demand_type_enc","cluster_id",
     "day_of_week","day_of_month","week_of_year","month","quarter","day_of_year",
     "is_weekend","is_month_end","is_month_start",
     "is_holiday","days_to_holiday",
     "is_campaign","has_discount","discount_amount",
     "dow_mean","month_mean","trend_7","volatility_28"]
    + COMMERCIAL_COLS
    + HIST_MONTH_COLS
    + HIST_Q1_COLS
    + [f"lag_{l}" for l in LAG_DAYS]
    + ["lag_365","lag_182"]
    + [f"roll_mean_{w}"  for w in ROLLING_WINS]
    + [f"roll_std_{w}"   for w in ROLLING_WINS]
    + [f"roll_max_{w}"   for w in ROLLING_WINS]
)

agg[FEATURE_COLS] = agg[FEATURE_COLS].fillna(0)
print(f"  Total features: {len(FEATURE_COLS)}  (base={len(FEATURE_COLS)-len(COMMERCIAL_COLS)}, comerciales={len(COMMERCIAL_COLS)})")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 3: Entrenamiento LightGBM...")
print("=" * 65)

cut_date = agg["date"].max() - pd.Timedelta(days=VAL_DAYS)
train    = agg[agg["date"] <= cut_date].dropna(subset=FEATURE_COLS)
val      = agg[agg["date"] >  cut_date].dropna(subset=FEATURE_COLS)

print(f"  Train: {train['date'].min().date()} -> {train['date'].max().date()} ({len(train):,})")
print(f"  Val  : {val['date'].min().date()} -> {val['date'].max().date()} ({len(val):,})")

X_train, y_train = train[FEATURE_COLS], train["total_quantity"]
X_val,   y_val   = val[FEATURE_COLS],   val["total_quantity"]

model = lgb.LGBMRegressor(**LGB_PARAMS)
model.fit(
    X_train, np.log1p(y_train),
    eval_set=[(X_val, np.log1p(y_val))],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

val_pred = np.maximum(0.0, np.expm1(model.predict(X_val)))
rmsle_v  = float(compute_rmsle(y_val, val_pred))
mae_v    = float(np.mean(np.abs(val_pred - y_val)))
rmse_v   = float(np.sqrt(np.mean((val_pred - y_val)**2)))
mask     = y_val > 0
mape_v   = float(np.mean(np.abs((val_pred[mask] - y_val.values[mask]) / y_val.values[mask])) * 100)

print(f"\n  ━━━ Metricas validacion ({VAL_DAYS} dias) ━━━")
print(f"  RMSLE: {rmsle_v:.4f}  ← metrica del concurso")
print(f"  MAE  : {mae_v:.3f}")
print(f"  RMSE : {rmse_v:.3f}")
print(f"  MAPE : {mape_v:.1f}%")

fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\n  Top 20 features:")
print(fi.head(20).to_string())

json.dump(
    {"rmsle": round(rmsle_v,4), "mae": round(mae_v,3), "rmse": round(rmse_v,3),
     "mape": round(mape_v,1), "n_features": len(FEATURE_COLS), "val_days": VAL_DAYS,
     "n_estimators": model.best_iteration_,
     "feature_importance_top20": fi.head(20).to_dict()},
    open("data/metricas.json","w"), indent=2
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ARIMA PARA SERIES REGULARES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 4: Ajustando ARIMA para series regulares...")
print("=" * 65)

regular_pairs = series_stats[series_stats["demand_type"] == "regular"][["origin_state","sku"]]
print(f"  Series regulares: {len(regular_pairs):,}")

_sub_raw   = pd.read_csv(SUBMISSION_CSV)
_ext       = _sub_raw["id"].str.extract(r"^(\d{4}-\d{2}-\d{2})_([A-Za-z]{2,})_(SKU[-_][\w-]+)$")
forecast_dates = sorted(pd.to_datetime(_ext[0].dropna().unique()))
n_forecast     = len(forecast_dates)
print(f"  Horizonte: {n_forecast} dias ({forecast_dates[0].date()} → {forecast_dates[-1].date()})")

ts_lookup = agg.groupby(["origin_state","sku"])


def _fit_arima(state, sku, ts_values):
    """ARIMA(2,1,1)(1,0,1)[7] — modelo secundario para series regulares."""
    try:
        mod = ARIMA(
            ts_values,
            order=(2, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = mod.fit(method_kwargs={"warn_convergence": False})
        fc  = np.maximum(0.0, fit.forecast(steps=n_forecast))
        return state, sku, dict(zip(forecast_dates, fc))
    except Exception:
        return state, sku, None


arima_preds = {}
with ThreadPoolExecutor(max_workers=ARIMA_MAX_WORKERS) as pool:
    futures = {
        pool.submit(
            _fit_arima,
            row["origin_state"], row["sku"],
            ts_lookup.get_group((row["origin_state"], row["sku"]))
                     .sort_values("date")["total_quantity"].values.astype(float),
        ): (row["origin_state"], row["sku"])
        for _, row in regular_pairs.iterrows()
    }
    done = 0
    for future in as_completed(futures):
        state, sku, fc = future.result()
        done += 1
        if fc is not None:
            arima_preds[(state, sku)] = fc
        if done % 50 == 0 or done == len(futures):
            print(f"    {done}/{len(futures)} ARIMA ajustados...")

print(f"  Exitosos: {len(arima_preds):,}  |  Solo LGB: {len(regular_pairs)-len(arima_preds):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PREDICCION RECURSIVA LIGHTGBM (loop vectorizado)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 5: Prediccion recursiva LightGBM...")
print("=" * 65)

sub  = pd.read_csv(SUBMISSION_CSV)
ext1 = sub["id"].str.extract(r"^(\d{4}-\d{2}-\d{2})_([A-Za-z]{2,})_(SKU[-_][\w-]+)$")
ext2 = sub["id"].str.extract(r"^(\d{4}-\d{2}-\d{2})_(SKU[-_][\w-]+)$")

mask_type1 = ext1[0].notna()
mask_type2  = (~mask_type1) & ext2[0].notna()

sub["date"]         = pd.to_datetime(ext1[0].where(mask_type1, ext2[0]))
sub["origin_state"] = ext1[1].where(mask_type1, "ALL")
sub["sku"]          = ext1[2].where(mask_type1, ext2[1])
sub["pred_type"]    = mask_type1.map({True: "state_sku", False: "sku_only"})
sub = sub.dropna(subset=["date","sku"]).reset_index(drop=True)

print(f"  Tipo fecha_estado_SKU : {mask_type1.sum():,}")
print(f"  Tipo fecha_SKU (total): {mask_type2.sum():,}")
print(f"  Fechas: {sub['date'].min().date()} -> {sub['date'].max().date()}")

# Lookups historicos para medias rapidas
dow_stats   = agg.groupby(["origin_state","sku","day_of_week"])["total_quantity"].mean().to_dict()
month_stats = agg.groupby(["origin_state","sku","month"])["total_quantity"].mean().to_dict()

# Lookups para estadisticas historicas por mes y Q1
_mh = monthly_stats.set_index(["origin_state","sku","month"])
month_hist_mean_lkp    = _mh["month_hist_mean"].to_dict()
month_hist_median_lkp  = _mh["month_hist_median"].to_dict()
month_hist_max_lkp     = _mh["month_hist_max"].to_dict()
month_hist_nonzero_lkp = _mh["month_hist_nonzero"].to_dict()

_q1 = q1_stats.set_index(["origin_state","sku"])
q1_mean_lkp    = _q1["q1_hist_mean"].to_dict()
q1_nonzero_lkp = _q1["q1_hist_nonzero"].to_dict()
q1_total_lkp   = _q1["q1_hist_total"].to_dict()

last_known    = agg[["date","origin_state","sku","total_quantity"]].copy()
future_dates  = sorted(sub["date"].unique())
lgb_preds_map = {}

pairs_sub = (
    sub[sub["pred_type"] == "state_sku"][["origin_state","sku"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
state_arr = pairs_sub["origin_state"].values
sku_arr   = pairs_sub["sku"].values

print(f"  Prediciendo {len(future_dates)} dias...")
for i, fdate in enumerate(future_dates):
    if i % 30 == 0:
        print(f"    Dia {i+1}/{len(future_dates)}: {fdate.date()}")

    rows = pairs_sub.copy()
    rows["date"] = fdate

    # Encodings
    rows["state_enc"]       = le_state.transform(state_arr)
    rows["sku_enc"]         = le_sku.transform(sku_arr)
    rows["demand_type_enc"] = [
        DEMAND_ENC.get(demand_type_map.get((s, sk), "intermittent"), 1)
        for s, sk in zip(state_arr, sku_arr)
    ]
    rows["cluster_id"] = [cluster_map.get((s, sk), 0) for s, sk in zip(state_arr, sku_arr)]

    # Calendar
    rows["day_of_week"]    = fdate.dayofweek
    rows["day_of_month"]   = fdate.day
    rows["week_of_year"]   = fdate.isocalendar()[1]
    rows["month"]          = fdate.month
    rows["quarter"]        = (fdate.month - 1) // 3 + 1
    rows["day_of_year"]    = fdate.timetuple().tm_yday
    rows["is_weekend"]     = int(fdate.dayofweek >= 5)
    rows["is_month_end"]   = int(fdate == fdate + pd.offsets.MonthEnd(0))
    rows["is_month_start"] = int(fdate.day == 1)
    rows["is_holiday"]     = int(fdate in US_HOLIDAYS_SET)
    rows["days_to_holiday"]= min(abs((fdate - h).days) for h in US_HOLIDAYS)
    rows["is_campaign"]    = 0
    rows["has_discount"]   = 0
    rows["discount_amount"]= 0.0

    # Calendario comercial (solo depende de la fecha, sin historial)
    comm = commercial_features_for_date(fdate)
    for col, val in comm.items():
        rows[col] = val

    # Estadisticas historicas por mes (Jan-Abr 2025 como referencia directa)
    m = fdate.month
    rows["month_hist_mean"]    = [month_hist_mean_lkp.get((s, sk, m), 0)    for s, sk in zip(state_arr, sku_arr)]
    rows["month_hist_median"]  = [month_hist_median_lkp.get((s, sk, m), 0)  for s, sk in zip(state_arr, sku_arr)]
    rows["month_hist_max"]     = [month_hist_max_lkp.get((s, sk, m), 0)     for s, sk in zip(state_arr, sku_arr)]
    rows["month_hist_nonzero"] = [month_hist_nonzero_lkp.get((s, sk, m), 0) for s, sk in zip(state_arr, sku_arr)]
    rows["q1_hist_mean"]       = [q1_mean_lkp.get((s, sk), 0)               for s, sk in zip(state_arr, sku_arr)]
    rows["q1_hist_nonzero"]    = [q1_nonzero_lkp.get((s, sk), 0)            for s, sk in zip(state_arr, sku_arr)]
    rows["q1_hist_total"]      = [q1_total_lkp.get((s, sk), 0)              for s, sk in zip(state_arr, sku_arr)]

    # Lags vectorizados
    hist_dict = last_known.set_index(["origin_state","sku","date"])["total_quantity"].to_dict()

    for lag in LAG_DAYS:
        lag_date = fdate - pd.Timedelta(days=lag)
        rows[f"lag_{lag}"] = [
            hist_dict.get((s, sk, lag_date), 0)
            for s, sk in zip(state_arr, sku_arr)
        ]


    # Lags anuales y semestrales para fechas futuras
    lag_365 = [hist_dict.get((s, sk, fdate - pd.Timedelta(days=365)), 0) for s, sk in zip(state_arr, sku_arr)]
    lag_182 = [hist_dict.get((s, sk, fdate - pd.Timedelta(days=182)), 0) for s, sk in zip(state_arr, sku_arr)]
    rows["lag_365"] = lag_365
    rows["lag_182"] = lag_182

    # Rolling vectorizado
    for win in ROLLING_WINS:
        win_dates = [fdate - pd.Timedelta(days=d) for d in range(1, win + 1)]
        r_mean, r_std, r_max = [], [], []
        for s, sk in zip(state_arr, sku_arr):
            window = [hist_dict.get((s, sk, wd), 0) for wd in win_dates]
            r_mean.append(float(np.mean(window)))
            r_std.append(float(np.std(window)))
            r_max.append(float(np.max(window)))
        rows[f"roll_mean_{win}"] = r_mean
        rows[f"roll_std_{win}"]  = r_std
        rows[f"roll_max_{win}"]  = r_max

    # Medias historicas
    rows["dow_mean"]   = [dow_stats.get((s, sk, fdate.dayofweek), 0) for s, sk in zip(state_arr, sku_arr)]
    rows["month_mean"] = [month_stats.get((s, sk, fdate.month), 0)   for s, sk in zip(state_arr, sku_arr)]

    # Tendencia y volatilidad
    dates_7  = [fdate - pd.Timedelta(days=d) for d in range(1, 8)]
    dates_8  = [fdate - pd.Timedelta(days=d) for d in range(2, 9)]
    dates_28 = [fdate - pd.Timedelta(days=d) for d in range(1, 29)]
    trend_v, vol_v = [], []
    for s, sk in zip(state_arr, sku_arr):
        w7  = [hist_dict.get((s, sk, wd), 0) for wd in dates_7]
        w8  = [hist_dict.get((s, sk, wd), 0) for wd in dates_8]
        w28 = [hist_dict.get((s, sk, wd), 0) for wd in dates_28]
        trend_v.append(float(np.mean(w7) - np.mean(w8)))
        vol_v.append(float(np.std(w28)))
    rows["trend_7"]       = trend_v
    rows["volatility_28"] = vol_v

    rows[FEATURE_COLS] = rows[FEATURE_COLS].fillna(0)
    preds = np.maximum(0.0, np.expm1(model.predict(rows[FEATURE_COLS])))

    for idx, (s, sk) in enumerate(zip(state_arr, sku_arr)):
        lgb_preds_map[(fdate, s, sk)] = preds[idx]

    new_rows = rows[["date","origin_state","sku"]].copy()
    new_rows["total_quantity"] = np.round(preds).astype(int)
    last_known = pd.concat([last_known, new_rows], ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ENSEMBLE LGB + ARIMA → SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 6: Ensemble LGB + ARIMA → submission...")
print("=" * 65)


def _ensemble_pred(r):
    if r["pred_type"] == "state_sku":
        lgb_p = lgb_preds_map.get((r["date"], r["origin_state"], r["sku"]), 0.0)
        key   = (r["origin_state"], r["sku"])
        dtype = demand_type_map.get(key, "intermittent")
        if dtype == "zero":
            return 0.0
        if dtype == "regular" and key in arima_preds:
            arima_p = float(arima_preds[key].get(r["date"], lgb_p))
            return (1.0 - ARIMA_BLEND) * lgb_p + ARIMA_BLEND * arima_p
        return lgb_p
    else:
        return sum(v for (d, _, sk), v in lgb_preds_map.items()
                   if d == r["date"] and sk == r["sku"])


sub["total_item_quantity"] = sub.apply(_ensemble_pred, axis=1)
sub["total_item_quantity"] = sub["total_item_quantity"].clip(lower=0)

# ── Post-processing: floor + cap por serie y mes ──────────────────────────────
# Floor: evita colapso de lags recursivos en series con demanda historica real
# Cap:   evita predicciones extremas en series con maximo historico conocido
def _apply_bounds(row):
    pred = row["total_item_quantity"]
    if row["pred_type"] != "state_sku":
        return pred
    s, sk = row["origin_state"], row["sku"]
    if demand_type_map.get((s, sk), "intermittent") == "zero":
        return 0.0
    m = row["date"].month
    hist_mean    = month_hist_mean_lkp.get((s, sk, m), 0)
    hist_nonzero = month_hist_nonzero_lkp.get((s, sk, m), 0)
    hist_max     = month_hist_max_lkp.get((s, sk, m), 0)
    # floor solo si la serie estuvo activa >= 30% de los dias de ese mes
    if hist_nonzero >= 0.30 and hist_mean > 0:
        pred = max(pred, hist_mean * FLOOR_RATIO)
    # cap absoluto
    if hist_max > 0:
        pred = min(pred, hist_max * CAP_RATIO)
    return pred

sub["total_item_quantity"] = sub.apply(_apply_bounds, axis=1)
sub["total_item_quantity"] = sub["total_item_quantity"].round(2)

sub[["id","total_item_quantity"]].to_csv(OUTPUT_CSV, index=False)
print(f"\n  Submission guardada: {OUTPUT_CSV}")
print(sub[["id","total_item_quantity"]].head(10).to_string())
print("\n  Estadisticas:")
print(sub["total_item_quantity"].describe())
n_zeros = (sub["total_item_quantity"] == 0).sum()
print(f"\n  Predicciones == 0: {n_zeros:,} ({100*n_zeros/len(sub):.1f}%)")

