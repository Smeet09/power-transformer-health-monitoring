"""ETT (Electricity Transformer Temperature) loader.

ETTm1: 15-min records 2016-07-01 .. 2018-06-26 from one Chinese substation transformer.
Columns: HUFL/HULL/MUFL/MULL/LUFL/LULL = high/middle/low useful/useless load, OT = oil temp.
The load columns are in unpublished units, so we map HUFL to a per-unit load factor K
for our 10 MVA unit (assumption documented in config.yaml).
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config, p

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def load_ett() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(p(cfg["paths"]["ett_csv"]), parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated()].asfreq("15min")
    df = df.interpolate(limit=8)  # ETTm1 has no gaps, but stay safe
    return df


def load_factor(df: pd.DataFrame) -> pd.Series:
    """Per-unit load K from the high-side useful load (negative values = reverse flow/metering
    artefacts; we take the magnitude of the apparent load)."""
    q = load_config()["synthetic"]["load_scaling_quantile"]
    s = df["HUFL"].abs()
    k = s / s.quantile(q)
    return k.clip(lower=0.05).rename("K")
