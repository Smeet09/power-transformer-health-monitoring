"""Duval Triangle 1 (IEC 60599:2022 Annex B) - rule-based baseline.

Relative percentages of CH4, C2H4, C2H2. Zone limits (Duval 2002):
  PD : %CH4 >= 98
  T1 : %C2H2 < 4,  %C2H4 < 20
  T2 : %C2H2 < 4,  20 <= %C2H4 < 50
  T3 : %C2H2 < 15, %C2H4 >= 50
  D1 : %C2H2 > 13, %C2H4 < 23
  D2 : %C2H2 > 13, %C2H4 >= 23, and not (%C2H4 >= 40 and %C2H2 < 29)
  DT : everything else (mixture of thermal and electrical faults)

The triangle always returns a fault zone, so a "Normal" pre-screen (all gases below IEC 60599
typical values) is applied first when `screen_normal=True`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def percentages(ch4, c2h4, c2h2):
    tot = ch4 + c2h4 + c2h2
    if tot <= 0:
        return np.nan, np.nan, np.nan
    return 100 * ch4 / tot, 100 * c2h4 / tot, 100 * c2h2 / tot


def zone(ch4: float, c2h4: float, c2h2: float) -> str:
    p_ch4, p_c2h4, p_c2h2 = percentages(ch4, c2h4, c2h2)
    if np.isnan(p_ch4):
        return "ND"                       # no hydrocarbons detected
    if p_ch4 >= 98:
        return "PD"
    if p_c2h2 < 4 and p_c2h4 < 20:
        return "T1"
    if p_c2h2 < 4 and p_c2h4 < 50:
        return "T2"
    if p_c2h2 < 15 and p_c2h4 >= 50:
        return "T3"
    if p_c2h2 > 13 and p_c2h4 < 23:
        return "D1"
    if p_c2h2 > 13 and p_c2h4 >= 23 and not (p_c2h4 >= 40 and p_c2h2 < 29):
        return "D2"
    return "DT"


def is_normal(row: dict | pd.Series) -> bool:
    typ = load_config()["dga"]["typical_ppm"]
    return all(float(row.get(g, 0) or 0) <= typ[g] for g in ("H2", "CH4", "C2H2", "C2H4", "C2H6"))


def diagnose(row: dict | pd.Series, screen_normal: bool = True) -> str:
    if screen_normal and is_normal(row):
        return "Normal"
    return zone(float(row["CH4"]), float(row["C2H4"]), float(row["C2H2"]))


def diagnose_frame(df: pd.DataFrame, screen_normal: bool = False) -> pd.Series:
    return df.apply(lambda r: diagnose(r, screen_normal), axis=1)
