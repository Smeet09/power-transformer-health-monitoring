"""Build one health-index snapshot per day from the 15-min thermal/loading data, daily DGA,
quarterly oil tests and the event log - this is what an online system would run nightly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dga.classifier import features
from src.health_index.scoring import compute

COMBUSTIBLE = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]


def daily_snapshots(thermal: pd.DataFrame, dga: pd.DataFrame, oil: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "hot_spot_max_c": thermal["hot_spot_c"].resample("D").max(),
        "top_oil_max_c": thermal["top_oil_c"].resample("D").max(),
        "peak_load_pu": thermal["K"].resample("D").max(),
        "ageing_rate_day": thermal["ageing_rate"].resample("D").mean(),
        "hours_over_1pu": (thermal["K"] > 1.0).resample("D").sum() * 0.25,
    })
    d["ageing_rate_30d"] = d["ageing_rate_day"].rolling(30, min_periods=1).mean()
    d["hours_over_1pu_30d"] = d["hours_over_1pu"].rolling(30, min_periods=1).sum()

    ev = events.copy()
    ev["day"] = ev["timestamp"].dt.normalize()
    alarms = ev[ev["event"].str.contains("OTI|WTI")].groupby("day").size().reindex(d.index, fill_value=0)
    d["alarm_count_30d"] = alarms.rolling(30, min_periods=1).sum()
    bh = ev[ev["event"] == "BUCHHOLZ_ALARM"].groupby("day").size().reindex(d.index, fill_value=0)
    d["buchholz_alarm"] = bh.rolling(30, min_periods=1).sum() > 0

    g = dga.reindex(d.index).ffill()
    d = d.join(g)
    tdcg = g[COMBUSTIBLE].sum(axis=1)
    d["tdcg_rate_ppm_per_day"] = (tdcg - tdcg.shift(30)) / 30
    d = d.join(oil.reindex(d.index, method="ffill"))
    return d


def score_timeline(daily: pd.DataFrame, model, le) -> pd.DataFrame:
    proba = model.predict_proba(features(daily))
    cls = le.inverse_transform(proba.argmax(1))
    rows = []
    for (ts, r), c, pmax in zip(daily.iterrows(), cls, proba.max(1)):
        snap = snapshot_from_row(r)
        res = compute(snap, {"class": c, "probability": float(pmax)})
        rows.append({"date": ts, "health_index": res["health_index"], "band": res["band"],
                     **{f"S_{k}": v for k, v in res["components"].items()},
                     "dga_status": res["dga_diagnosis"]["status"], "ml_class": c,
                     "duval": res["dga_diagnosis"]["duval_triangle_1"],
                     "driver_1": _fmt(res["top_drivers"], 0), "driver_2": _fmt(res["top_drivers"], 1),
                     "driver_3": _fmt(res["top_drivers"], 2)})
    return pd.DataFrame(rows).set_index("date")


def snapshot_from_row(r: pd.Series) -> dict:
    nz = lambda v: None if pd.isna(v) else float(v)
    return {
        "thermal": {"hot_spot_max_c": nz(r["hot_spot_max_c"]), "ageing_rate_30d": nz(r["ageing_rate_30d"]),
                    "alarm_count_30d": nz(r["alarm_count_30d"])},
        "loading": {"peak_load_pu": nz(r["peak_load_pu"]), "hours_over_1pu_30d": nz(r["hours_over_1pu_30d"])},
        "dga": {g: nz(r[g]) for g in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]}
               | {"tdcg_rate_ppm_per_day": nz(r["tdcg_rate_ppm_per_day"])},
        "oil": {"moisture_ppm": nz(r.get("moisture_ppm")), "bdv_kv": nz(r.get("bdv_kv"))},
        "events": {"buchholz_alarm": bool(r["buchholz_alarm"])},
    }


def _fmt(drivers, i):
    if i >= len(drivers):
        return ""
    d = drivers[i]
    v = d["value"]
    v = f"{v:.1f}" if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else str(v)
    return f"{d['parameter']} = {v} (-{d['points_lost']:.1f})"
