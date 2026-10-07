"""Drift monitoring.

1. Data drift  - Population Stability Index per key input, reference = training window,
                 current = a recent window.  PSI > 0.2 => significant drift (threshold in config).
2. Performance - rolling 7-day MAE of the 24 h top-oil forecast vs its validation MAE.
                 ratio > 1.5 => model degraded (actuals arrive 24 h later, so this is observable).
3. DGA         - share of new samples whose ML class disagrees with Duval Triangle 1; a jump
                 signals a new fault population / data issue (reported, not a hard trigger).
Any hard trigger -> retraining (see retraining_triggers()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def psi(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def data_drift(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str]) -> dict:
    m = load_config()["mlops"]["drift"]
    out = {}
    for c in cols:
        v = psi(reference[c], current[c], m["psi_bins"])
        out[c] = {"psi": round(v, 3), "drift": v > m["psi_threshold"]}
    return out


def rolling_mae(frame: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    days = load_config()["mlops"]["drift"]["rolling_days"]
    err = (frame["actual"] - frame[pred_col]).abs()
    return err.rolling(f"{days}D").mean()


def performance_drift(frame: pd.DataFrame, validation_mae: float) -> dict:
    thr = load_config()["mlops"]["drift"]["mae_ratio_threshold"]
    r = rolling_mae(frame)
    r = r[r.index >= r.index[0] + pd.Timedelta(days=load_config()["mlops"]["drift"]["rolling_days"])]
    ratio = r / validation_mae
    breaches = ratio[ratio > thr]
    return {
        "validation_mae": validation_mae,
        "threshold_ratio": thr,
        "max_ratio": round(float(ratio.max()), 2),
        "days_in_breach": int(breaches.resample("D").size().gt(0).sum()),
        "first_breach": str(breaches.index[0]) if len(breaches) else None,
        "latest_ratio": round(float(ratio.iloc[-1]), 2),
    }


def retraining_triggers(data_drift_report: dict, perf: dict, new_labelled_dga: int = 0,
                        days_since_training: int = 0) -> list[str]:
    t = []
    drifted = [k for k, v in data_drift_report.items() if v["drift"]]
    if drifted:
        t.append(f"Data drift (PSI > {load_config()['mlops']['drift']['psi_threshold']}) on: {', '.join(drifted)}")
    if perf["latest_ratio"] > perf["threshold_ratio"]:
        t.append(f"Forecast rolling MAE {perf['latest_ratio']}x validation MAE")
    if new_labelled_dga >= 25:
        t.append(f"{new_labelled_dga} new inspection-confirmed DGA cases available")
    if days_since_training >= 90:
        t.append("Scheduled quarterly retrain (90 days since last training)")
    return t
