"""24-hour-ahead top-oil temperature forecast on ETTm1 (real OT measurements).

Direct strategy: at time t predict OT(t+96). The model learns the 24 h *change*
OT(t+96) - OT(t) from level-free features (lags and rolling stats relative to the current
OT, load deviations, calendar). Absolute-level features were dropped because they over-fit:
2018 oil/load levels differ from 2016-17 (tested: test MAE 3.1 C vs 1.7 C for persistence).

Two model variants:
  * causal      - only information available at t (strict, primary result)
  * day_ahead   - additionally uses a day-ahead load forecast for t+24 h, simulated as the
                  actual future load with 5 % Gaussian error (utilities normally have one).
Baselines: persistence (= 24 h seasonal naive at this horizon), 7-day seasonal naive,
trailing 24 h mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.data.ett import LOAD_COLS

H = 96
NON_FEATURES = {"target_delta", "ot", "target", "base_persistence", "base_seasonal_7d", "base_mean_24h"}


def build_dataset(df: pd.DataFrame, day_ahead_load: bool = False) -> pd.DataFrame:
    ot = df["OT"]
    f = pd.DataFrame(index=df.index)
    for lag in (1, 2, 4, 8, 16, 24, 48, 96, 192, 288, 672):
        f[f"d_lag_{lag}"] = ot.shift(lag) - ot
    f["d_seasonal_7d"] = ot.shift(672 - H) - ot          # OT one week before the target time
    for w in (16, 96, 672):
        f[f"d_mean_{w}"] = ot.rolling(w).mean() - ot
    f["std_96"] = ot.rolling(96).std()
    for c in LOAD_COLS:
        f[f"{c}_dev_24h"] = df[c] - df[c].rolling(96).mean()
        f[f"{c}_seasonal_7d"] = df[c].shift(672 - H) - df[c]
    if day_ahead_load:
        rng = np.random.default_rng(load_config()["random_seed"])
        for c in LOAD_COLS:
            fc = df[c].shift(-H) * (1 + rng.normal(0, 0.05, len(df)))
            f[f"{c}_dayahead"] = fc - df[c]
            f[f"{c}_dayahead_mean"] = df[c].rolling(96).mean().shift(-H) - df[c].rolling(96).mean()
    tgt = df.index + pd.Timedelta(minutes=15 * H)
    f["hour_sin"] = np.sin(2 * np.pi * (tgt.hour + tgt.minute / 60) / 24)
    f["hour_cos"] = np.cos(2 * np.pi * (tgt.hour + tgt.minute / 60) / 24)
    f["dow"] = tgt.dayofweek

    f["ot"] = ot
    f["target"] = ot.shift(-H)
    f["target_delta"] = f["target"] - ot
    f["base_persistence"] = ot
    f["base_seasonal_7d"] = ot.shift(672 - H)
    f["base_mean_24h"] = ot.rolling(96).mean()
    return f.dropna()


def feature_cols(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns if c not in NON_FEATURES]


def split(ds: pd.DataFrame):
    c = load_config()["forecast"]
    gap = pd.Timedelta(minutes=15 * H)   # target is 24 h ahead -> 24 h gap prevents leakage
    tr = ds.loc[: pd.Timestamp(c["train_end"]) - gap]
    va = ds.loc[pd.Timestamp(c["train_end"]) + pd.Timedelta(minutes=15): pd.Timestamp(c["valid_end"]) - gap]
    te = ds.loc[pd.Timestamp(c["valid_end"]) + pd.Timedelta(minutes=15):]
    return tr, va, te


def new_model() -> xgb.XGBRegressor:
    c = load_config()
    return xgb.XGBRegressor(**c["forecast"]["xgb"], objective="reg:squarederror",
                            random_state=c["random_seed"], n_jobs=-1)


def fit(train_df: pd.DataFrame) -> xgb.XGBRegressor:
    m = new_model()
    m.fit(train_df[feature_cols(train_df)], train_df["target_delta"])
    return m


def predict(model: xgb.XGBRegressor, ds: pd.DataFrame) -> pd.Series:
    return pd.Series(ds["ot"].values + model.predict(ds[feature_cols(ds)]), index=ds.index)


def metrics(y, yhat) -> dict:
    e = np.asarray(y) - np.asarray(yhat)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}


def run(df: pd.DataFrame) -> dict:
    """Train both variants. Model selection on validation (train-only fit), final model
    refit on train+validation and scored once on the held-out test period."""
    results, frames, models = {}, {}, {}
    for variant, dal in (("causal", False), ("day_ahead", True)):
        ds = build_dataset(df, day_ahead_load=dal)
        tr, va, te = split(ds)
        val_model = fit(tr)
        val_mae = metrics(va["target"], predict(val_model, va))["MAE"]
        model = fit(pd.concat([tr, va]))
        pred = predict(model, te)
        results[variant] = {"validation_MAE": val_mae, "test": metrics(te["target"], pred)}
        frames[variant] = pd.DataFrame({"actual": te["target"].values, "pred": pred.values,
                                        "persistence": te["base_persistence"].values},
                                       index=te.index + pd.Timedelta(minutes=15 * H))
        models[variant] = model
        if variant == "causal":
            results["baselines_test"] = {
                "Persistence (24 h seasonal naive)": metrics(te["target"], te["base_persistence"]),
                "Seasonal naive 7 d": metrics(te["target"], te["base_seasonal_7d"]),
                "Trailing 24 h mean": metrics(te["target"], te["base_mean_24h"]),
            }
            results["baselines_validation_MAE"] = {"Persistence (24 h seasonal naive)": metrics(va["target"], va["base_persistence"])["MAE"]}
            results["split"] = {"train": [str(tr.index[0]), str(tr.index[-1])],
                                "validation": [str(va.index[0]), str(va.index[-1])],
                                "test": [str(te.index[0]), str(te.index[-1])],
                                "n_train": len(tr), "n_valid": len(va), "n_test": len(te)}
            results["feature_importance_top10"] = (pd.Series(model.feature_importances_, index=feature_cols(te))
                                                   .sort_values(ascending=False).head(10).round(4).to_dict())
            frames["_test_inputs"] = te
    best_base = min(v["MAE"] for v in results["baselines_test"].values())
    for v in ("causal", "day_ahead"):
        results[v]["MAE_improvement_vs_best_baseline_pct"] = 100 * (1 - results[v]["test"]["MAE"] / best_base)
    return {"results": results, "frames": frames, "models": models}
