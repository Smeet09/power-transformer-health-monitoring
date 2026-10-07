"""Public DGA fault datasets.

* Training set  - 589 labelled cases compiled from literature (alan-456/transformer-fault-dataset,
  originally IEEE DataPort, E. Li 2019). Labels are Chinese -> IEC 60599 classes.
  Values of 0.0001 are placeholders for "not detected" -> 0.
* External test - IEC TC 10 database (Duval & dePablo, IEEE EI Magazine 2001), 113 inspected
  faults incl. CO/CO2, via fedebenelli/DGANN. Codes: 0 PD, 1 D1, 2 D2, 3 T1+T2 (merged), 4 T3.

109 of the 113 IEC TC 10 cases are also inside the 589 set; they are REMOVED from training
so the IEC TC 10 benchmark is truly unseen.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config, p

FIVE_GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
CLASSES = ["PD", "D1", "D2", "T1", "T2", "T3"]
CN_LABELS = {"局部放电": "PD", "低能放电": "D1", "高能放电": "D2",
             "低温过热": "T1", "中温过热": "T2", "高温过热": "T3"}
TC10_CODES = {0: "PD", 1: "D1", 2: "D2", 3: "T12", 4: "T3"}


def _key(df: pd.DataFrame) -> pd.Series:
    return df[FIVE_GASES].astype(float).round(1).astype(str).agg("|".join, axis=1)


def load_iec_tc10() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(p(cfg["paths"]["dga_iec_tc10_csv"]), encoding="gbk")
    df = df[["Equip.", "Inspection", "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "fault"]].copy()
    df["label"] = df["fault"].map(TC10_CODES)
    return df.drop(columns="fault")


def load_training(drop_tc10_overlap: bool = True) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()
    raw = pd.read_excel(p(cfg["paths"]["dga_train_xlsx"]))
    label_col = raw.columns[-1]
    df = raw[FIVE_GASES].copy()
    df = df.mask(df < 0.001, 0.0)
    df["label"] = raw[label_col].map(CN_LABELS)
    info = {"rows_raw": len(df), "unmapped_labels": int(df["label"].isna().sum())}
    df = df.dropna(subset=["label"])
    if drop_tc10_overlap:
        tc10_keys = set(_key(load_iec_tc10()))
        overlap = _key(df).isin(tc10_keys)
        info["removed_overlap_with_iec_tc10"] = int(overlap.sum())
        df = df[~overlap]
    dup = _key(df).duplicated()
    info["removed_duplicates"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    info["rows_final"] = len(df)
    info["class_counts"] = df["label"].value_counts().reindex(CLASSES).to_dict()
    return df, info
