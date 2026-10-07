"""Health index (0-100) = weighted sum of four component scores, each built from
piecewise-linear parameter scores (100 = good limit, 0 = bad limit, see config.yaml).

    HI = sum_c w_c * S_c ,  S_c = sum_p a_p * s_p / sum_p a_p

Because HI = 100 - sum of deficits, every point lost is attributable to one parameter:
    deficit_p = w_c * a_p / sum(a) * (100 - s_p)
The top-3 deficits are the "drivers" of the score. DGA fault severity caps the DGA component
and the overall index; protection events cap the overall index; a cap's lost points are
reported as their own driver.
"""
from __future__ import annotations

import numpy as np

from src.config import load_config
from src.dga.duval import diagnose, is_normal

GASES7 = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]


def lin_score(value: float, good: float, bad: float) -> float:
    """100 at `good`, 0 at `bad`, linear in between (works for rising or falling scales)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    t = (value - good) / (bad - good)
    return float(np.clip(100 * (1 - t), 0, 100))


def band_for(score: float) -> dict:
    for b in load_config()["health_index"]["bands"]:
        if score >= b["min"]:
            return b
    return load_config()["health_index"]["bands"][-1]


def _component(params: list[tuple]) -> tuple[float, list[dict]]:
    """params: (name, display_value, score, weight). NaN scores are skipped (missing data)."""
    valid = [x for x in params if not np.isnan(x[2])]
    if not valid:
        return np.nan, []
    wsum = sum(x[3] for x in valid)
    score = sum(x[2] * x[3] for x in valid) / wsum
    return score, [{"parameter": n, "value": round(float(v), 2), "score": round(s, 1), "rel_weight": w / wsum} for n, v, s, w in valid]


def compute(snapshot: dict, dga_fault: dict | None = None) -> dict:
    """snapshot keys: thermal, loading, dga, oil, events (see api schema). dga_fault is the
    ML classifier output {class, probability}; Duval result is computed here."""
    cfg = load_config()
    hc = cfg["health_index"]
    W = hc["weights"]
    comps, details = {}, {}

    # ---- thermal
    t = snapshot.get("thermal", {})
    th = hc["thermal"]
    comps["thermal"], details["thermal"] = _component([
        ("Hot-spot max (C)", t.get("hot_spot_max_c"), lin_score(t.get("hot_spot_max_c", np.nan), **th["hotspot_max_c"]), 2),
        ("Ageing rate V (30 d mean)", t.get("ageing_rate_30d"), lin_score(t.get("ageing_rate_30d", np.nan), **th["aging_rate_30d"]), 2),
        ("OTI/WTI alarms (30 d)", t.get("alarm_count_30d"), lin_score(t.get("alarm_count_30d", np.nan), **th["alarm_count_30d"]), 1),
    ])

    # ---- loading
    l = snapshot.get("loading", {})
    lc = hc["loading"]
    comps["loading"], details["loading"] = _component([
        ("Peak load (pu, 24 h)", l.get("peak_load_pu"), lin_score(l.get("peak_load_pu", np.nan), **lc["peak_pu"]), 2),
        ("Hours above 1.0 pu (30 d)", l.get("hours_over_1pu_30d"), lin_score(l.get("hours_over_1pu_30d", np.nan), **lc["hours_over_1pu_30d"]), 1),
    ])

    # ---- oil quality
    o = snapshot.get("oil", {})
    oc = hc["oil"]
    comps["oil"], details["oil"] = _component([
        ("Moisture (ppm)", o.get("moisture_ppm"), lin_score(o.get("moisture_ppm", np.nan), **oc["moisture_ppm"]), 1),
        ("Breakdown voltage (kV)", o.get("bdv_kv"), lin_score(o.get("bdv_kv", np.nan), **oc["bdv_kv"]), 1),
    ])

    # ---- DGA
    g = snapshot.get("dga", {})
    typ, gw, mult = cfg["dga"]["typical_ppm"], hc["dga_gas_weights"], hc["dga_bad_multiple"]
    params = [(f"{gas} (ppm)", g.get(gas), lin_score(g[gas], 0.5 * typ[gas], mult * typ[gas]) if g.get(gas) is not None else np.nan, gw[gas])
              for gas in GASES7]
    if g.get("tdcg_rate_ppm_per_day") is not None:
        params.append(("TDCG rate (ppm/day)", g["tdcg_rate_ppm_per_day"],
                       lin_score(g["tdcg_rate_ppm_per_day"], **hc["dga_tdcg_rate_ppm_day"]), 6))
    gas_score, details["dga"] = _component(params)

    duval = diagnose(g) if all(k in g for k in ("CH4", "C2H4", "C2H2")) else "n/a"
    normal = is_normal(g) if g else True
    fault = "Normal" if normal else (dga_fault or {}).get("class", duval)
    sev = cfg["dga"]["severity"].get(fault, 0.5)
    dga_cap = 100 * (1 - 0.7 * sev)
    comps["dga"] = min(gas_score, dga_cap) if not np.isnan(gas_score) else np.nan

    # ---- combine (re-normalise weights if a component is missing)
    avail = {c: s for c, s in comps.items() if not np.isnan(s)}
    wsum = sum(W[c] for c in avail)
    hi = sum(W[c] * s for c, s in avail.items()) / wsum

    drivers = []
    for c, plist in details.items():
        if c not in avail:
            continue
        for d in plist:
            drivers.append({"component": c, "parameter": d["parameter"], "value": d["value"],
                            "points_lost": W[c] / wsum * d["rel_weight"] * (100 - d["score"])})
    if not np.isnan(gas_score) and gas_score > dga_cap:
        drivers.append({"component": "dga", "parameter": f"Fault type {fault} (severity cap)", "value": fault,
                        "points_lost": W["dga"] / wsum * (gas_score - dga_cap)})

    caps = hc["caps"]
    # A diagnosed internal fault must not be averaged away by good thermal/oil scores:
    # cap the overall index by fault severity (T2 -> 67, i.e. at best "Fair").
    if not normal and sev > 0:
        fault_cap = 100 - caps["dga_fault_overall_k"] * sev
        if hi > fault_cap:
            drivers.append({"component": "dga", "parameter": f"Active {fault} fault (overall cap)", "value": fault,
                            "points_lost": hi - fault_cap})
            hi = fault_cap

    ev = snapshot.get("events", {}) or {}
    for key, cap in (("buchholz_trip", caps["buchholz_trip"]), ("prd_operated", caps["prd_operated"]),
                     ("buchholz_alarm", caps["buchholz_alarm"])):
        if ev.get(key) and hi > cap:
            drivers.append({"component": "events", "parameter": key.replace("_", " ").title() + " (cap)",
                            "value": True, "points_lost": hi - cap})
            hi = cap

    drivers.sort(key=lambda d: d["points_lost"], reverse=True)
    band = band_for(hi)
    return {
        "health_index": round(float(hi), 1),
        "band": band["name"],
        "band_action": band["action"],
        "components": {c: (round(float(s), 1) if not np.isnan(s) else None) for c, s in comps.items()},
        "top_drivers": [{**d, "points_lost": round(float(d["points_lost"]), 1)} for d in drivers[:3]],
        "dga_diagnosis": {"status": "Normal" if normal else "Abnormal", "ml_class": (dga_fault or {}).get("class"),
                          "ml_probability": (dga_fault or {}).get("probability"), "duval_triangle_1": duval},
    }
