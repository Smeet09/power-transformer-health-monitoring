"""IEC 60076-7:2018 thermal model - difference-equation solution (Annex C / clause 8.2.3)
plus relative ageing rate and loss of life (clause 6).

    D_theta_o  = Dt/(k11*tau_o) * [ ((1+K^2 R)/(1+R))^x * d_theta_or - (theta_o - theta_a) ]
    D_dth1     = Dt/(k22*tau_w) * [ k21 * d_theta_hr * K^y - dth1 ]
    D_dth2     = Dt/((1/k22)*tau_o) * [ (k21-1) * d_theta_hr * K^y - dth2 ]
    theta_h    = theta_o + dth1 - dth2,         d_theta_hr = H * gr

Ageing:  non-thermally-upgraded paper  V = 2^((theta_h - 98)/6)
         thermally upgraded paper      V = exp(15000/110+273 - 15000/(theta_h+273))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def steady_state(K: float, theta_a: float, cfg: dict | None = None) -> tuple[float, float]:
    """Steady-state top-oil and hot-spot temperature for a constant load K (used by the API
    when only a snapshot is available)."""
    c = cfg or load_config()["thermal"]
    d_or = c["delta_theta_or_k"] * ((1 + K ** 2 * c["R"]) / (1 + c["R"])) ** c["x"]
    theta_o = theta_a + d_or
    theta_h = theta_o + c["H"] * c["gr_k"] * K ** c["y"]
    return theta_o, theta_h


def ageing_rate(theta_h, paper: str = "non_upgraded"):
    theta_h = np.asarray(theta_h, dtype=float)
    if paper == "upgraded":
        return np.exp(15000 / (110 + 273) - 15000 / (theta_h + 273))
    return 2.0 ** ((theta_h - 98.0) / 6.0)


def simulate(K: pd.Series, theta_a: pd.Series, cfg: dict | None = None) -> pd.DataFrame:
    """Dynamic top-oil / hot-spot simulation on the 15-min grid of K, integrating with
    sub-steps of `substep_min` minutes (load and ambient held constant within a record)."""
    c = cfg or load_config()["thermal"]
    dt_rec = (K.index[1] - K.index[0]).total_seconds() / 60
    n_sub = max(int(round(dt_rec / c["substep_min"])), 1)
    dt = dt_rec / n_sub
    x, y, R = c["x"], c["y"], c["R"]
    d_or, d_hr = c["delta_theta_or_k"], c["H"] * c["gr_k"]
    k11, k21, k22, to, tw = c["k11"], c["k21"], c["k22"], c["tau_o_min"], c["tau_w_min"]

    k = K.values.astype(float)
    ta = theta_a.values.astype(float)
    # initialise at steady state of the first record
    th_o = ta[0] + d_or * ((1 + k[0] ** 2 * R) / (1 + R)) ** x
    dth1 = k21 * d_hr * k[0] ** y
    dth2 = (k21 - 1) * d_hr * k[0] ** y
    out_o = np.empty(len(k))
    out_h = np.empty(len(k))
    a_o, a_1, a_2 = dt / (k11 * to), dt / (k22 * tw), dt / ((1 / k22) * to)
    for i in range(len(k)):
        oil_ss = ((1 + k[i] ** 2 * R) / (1 + R)) ** x * d_or
        w = d_hr * k[i] ** y
        for _ in range(n_sub):
            th_o += a_o * (oil_ss - (th_o - ta[i]))
            dth1 += a_1 * (k21 * w - dth1)
            dth2 += a_2 * ((k21 - 1) * w - dth2)
        out_o[i] = th_o
        # The two-exponential winding term models the hot-spot overshoot on load increase; on a sharp
        # load DROP it undershoots (dth1 decays with k22*tau_w, dth2 with tau_o/k22) and can push the
        # computed hot-spot below top-oil, which is not physical -> clamp the gradient at >= 0.
        out_h[i] = th_o + max(dth1 - dth2, 0.0)

    V = ageing_rate(out_h, c["paper"])
    hours = dt_rec / 60
    return pd.DataFrame(
        {"K": k, "ambient_c": ta, "top_oil_c": out_o, "hot_spot_c": out_h,
         "ageing_rate": V, "lol_hours": np.cumsum(V * hours)},
        index=K.index,
    )


def loss_of_life_summary(sim: pd.DataFrame) -> dict:
    c = load_config()["thermal"]
    hours = (sim.index[-1] - sim.index[0]).total_seconds() / 3600 + 0.25
    lol = float(sim["lol_hours"].iloc[-1])
    return {
        "period_hours": hours,
        "loss_of_life_hours": lol,
        "equivalent_ageing_factor": lol / hours,
        "life_consumed_pct": 100 * lol / c["normal_life_hours"],
        "normal_life_hours": c["normal_life_hours"],
        "max_hot_spot_c": float(sim["hot_spot_c"].max()),
        "p99_hot_spot_c": float(sim["hot_spot_c"].quantile(0.99)),
        "max_top_oil_c": float(sim["top_oil_c"].max()),
        "paper": c["paper"],
    }
