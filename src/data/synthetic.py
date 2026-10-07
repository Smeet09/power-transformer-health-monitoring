"""Synthetic signals for parameters that are not available in public data.

Everything here is clearly an assumption (see config.yaml -> synthetic) and is seeded so
the whole pipeline is reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config

GASES = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(load_config()["random_seed"] + offset)


def ambient_temperature(index: pd.DatetimeIndex) -> pd.Series:
    a = load_config()["synthetic"]["ambient"]
    # smooth seasonal curve: monthly means placed mid-month, periodic linear interpolation by day of year
    mid = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    m = np.asarray(a["monthly_mean_c"], dtype=float)
    seasonal = np.interp(index.dayofyear.values, np.r_[mid[-1] - 365, mid, mid[0] + 365], np.r_[m[-1], m, m[0]])
    hour = index.hour.values + index.minute.values / 60
    daily = a["daily_amplitude_c"] * np.cos(2 * np.pi * (hour - 15) / 24)
    noise = pd.Series(_rng(1).normal(0, a["noise_c"], len(index))).rolling(8, min_periods=1).mean().values
    return pd.Series(seasonal + daily + noise, index=index, name="ambient_c")


def electrical_signals(K: pd.Series) -> pd.DataFrame:
    """Three-phase currents (HV and LV), phase voltages and OLTC tap from load factor K."""
    cfg = load_config()
    t, s = cfg["transformer"], cfg["synthetic"]
    rng = _rng(2)
    n = len(K)
    out = {}
    for side, rated in (("hv", t["hv_rated_a"]), ("lv", t["lv_rated_a"])):
        for ph in "abc":
            imb = 1 + rng.normal(0, s["phase_imbalance_pct"] / 100, n)
            out[f"i_{side}_{ph}_a"] = (K.values * rated * imb).round(1)
    # LV voltage sags with load; OLTC raises the tap to hold 11 kV
    v_drop_pct = 4.0 * K.values                         # ~4 % regulation at full load (assumption)
    tap = np.clip(np.round(t["oltc_nominal_tap"] + v_drop_pct / t["oltc_step_pct"]), 1, t["oltc_taps"])
    v_lv = t["lv_kv"] * (1 - v_drop_pct / 100 + (tap - t["oltc_nominal_tap"]) * t["oltc_step_pct"] / 100)
    for ph in "abc":
        out[f"v_lv_{ph}_kv"] = (v_lv * (1 + rng.normal(0, s["voltage_noise_pct"] / 100, n))).round(3)
        out[f"v_hv_{ph}_kv"] = (t["hv_kv"] * (1 + rng.normal(0, s["voltage_noise_pct"] / 100, n))).round(3)
    out["tap_position"] = tap.astype(int)
    return pd.DataFrame(out, index=K.index)


def dga_timeline(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily online-DGA readings for the monitored unit: healthy baseline with slow paper
    ageing (CO/CO2 creep), then an incipient T2 thermal fault (CH4/C2H4 growth)."""
    cfg = load_config()
    s = cfg["synthetic"]["dga"]
    days = pd.date_range(index[0].normalize(), index[-1].normalize(), freq=f"{s['sample_every_days']}D")
    n = len(days)
    rng = _rng(3)
    # healthy CO2/CO ~ 8: inside the 3..10 band of normal paper ageing (USBR FIST 3-30 p.59)
    base = {"H2": 35, "CH4": 25, "C2H2": 0.5, "C2H4": 12, "C2H6": 18, "CO": 250, "CO2": 2000}
    creep = {"H2": 0.3, "CH4": 0.4, "C2H2": 0.0, "C2H4": 0.3, "C2H6": 0.4, "CO": 0.6, "CO2": 0.6}  # rel. growth over the period
    fault_end = {"H2": 160, "CH4": 420, "C2H2": 2.0, "C2H4": 310, "C2H6": 140, "CO": 150, "CO2": 1500}  # extra ppm at end
    frac = np.linspace(0, 1, n)
    onset = max(n - s["fault_onset_days_before_end"], 0)
    fault_prog = np.zeros(n)
    m = n - onset
    fault_prog[onset:] = (np.exp(np.linspace(0, 3, m)) - 1) / (np.e ** 3 - 1)  # accelerating growth
    data = {}
    for g in GASES:
        v = base[g] * (1 + creep[g] * frac) + fault_end[g] * fault_prog
        v = v * rng.lognormal(0, 0.06, n)
        data[g] = np.round(v, 1)
    return pd.DataFrame(data, index=days)


def oil_quality_tests(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Quarterly lab results: moisture rising and BDV falling (ageing + fault heating)."""
    o = load_config()["synthetic"]["oil"]
    dates = pd.date_range(index[0].normalize(), index[-1].normalize(), freq="QS") + pd.Timedelta(days=14)
    dates = dates[dates <= index[-1]]
    n = len(dates)
    rng = _rng(4)
    frac = np.linspace(0, 1, n) ** 1.5
    moist = o["moisture_start_ppm"] + (o["moisture_end_ppm"] - o["moisture_start_ppm"]) * frac + rng.normal(0, 1, n)
    bdv = o["bdv_start_kv"] + (o["bdv_end_kv"] - o["bdv_start_kv"]) * frac + rng.normal(0, 1.5, n)
    return pd.DataFrame({"moisture_ppm": moist.round(1), "bdv_kv": bdv.round(1)}, index=dates)


def protection_events(thermal: pd.DataFrame) -> pd.DataFrame:
    """Event log: OTI/WTI alarms/trips derived from temperatures (threshold crossings, rising
    edge) + a synthetic Buchholz gas alarm when the developing fault produces free gas."""
    cfg = load_config()
    al = cfg["thermal"]["alarms"]
    rows = []
    for col, name in (("oti_c", "OTI"), ("wti_c", "WTI")):
        for level in ("alarm", "trip"):
            thr = al[f"{name.lower()}_{level}_c"]
            above = thermal[col] >= thr
            edges = above & ~above.shift(1, fill_value=False)
            for ts in thermal.index[edges]:
                rows.append({"timestamp": ts, "event": f"{name}_{level.upper()}", "value": round(float(thermal.at[ts, col]), 1)})
    bd = cfg["synthetic"]["events"]["buchholz_alarm_days_before_end"]
    rows.append({"timestamp": thermal.index[-1] - pd.Timedelta(days=bd), "event": "BUCHHOLZ_ALARM", "value": np.nan})
    ev = pd.DataFrame(rows, columns=["timestamp", "event", "value"]).sort_values("timestamp").reset_index(drop=True)
    return ev
