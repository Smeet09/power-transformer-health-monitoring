"""ML fault classifier for DGA, compared with Duval Triangle 1.

Features are scale-free (relative gas composition, Duval percentages, IEC ratios) plus log
concentrations, so the model generalises across transformers with very different gas levels.

Class imbalance (training: D2 ~2.4x T2): we compare
  (a) no handling, (b) class-balanced sample weights, (c) SMOTE inside each CV training fold
  (never on validation folds), and select on macro-F1 - the metric that weighs every fault
  class equally, so a model cannot look good by favouring the majority class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import load_config
from src.dga.datasets import CLASSES
from src.dga.duval import diagnose_frame, percentages

EPS = 0.1  # ppm, avoids division by zero in ratios


def features(df: pd.DataFrame) -> pd.DataFrame:
    g = {c: df[c].astype(float).clip(lower=0) for c in ("H2", "CH4", "C2H6", "C2H4", "C2H2")}
    tot = sum(g.values()) + EPS
    f = pd.DataFrame(index=df.index)
    for c, v in g.items():
        f[f"log_{c}"] = np.log10(v + 1)
        f[f"pct_{c}"] = v / tot
    duv = np.array([percentages(a, b, c) for a, b, c in zip(g["CH4"], g["C2H4"], g["C2H2"])], dtype=float)
    f["duval_ch4"], f["duval_c2h4"], f["duval_c2h2"] = np.nan_to_num(duv, nan=0).T
    f["r_ch4_h2"] = np.log10((g["CH4"] + EPS) / (g["H2"] + EPS))
    f["r_c2h2_c2h4"] = np.log10((g["C2H2"] + EPS) / (g["C2H4"] + EPS))
    f["r_c2h4_c2h6"] = np.log10((g["C2H4"] + EPS) / (g["C2H6"] + EPS))
    f["r_c2h6_ch4"] = np.log10((g["C2H6"] + EPS) / (g["CH4"] + EPS))
    f["log_tdcg"] = np.log10(tot + 1)
    return f


def _models(seed: int) -> dict:
    xgb_kw = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="mlogloss")
    return {
        "XGBoost - no imbalance handling": ("plain", XGBClassifier(**xgb_kw)),
        "XGBoost - balanced sample weights": ("weights", XGBClassifier(**xgb_kw)),
        "XGBoost - SMOTE (train folds only)": ("smote", ImbPipeline([("smote", SMOTE(random_state=seed, k_neighbors=5)),
                                                                     ("clf", XGBClassifier(**xgb_kw))])),
        "Random forest - class_weight=balanced": ("plain", RandomForestClassifier(
            n_estimators=500, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1)),
    }


def _fit(kind, model, X, y):
    if kind == "weights":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def report(y_true, y_pred, labels) -> dict:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "per_class": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score", "support")}
                      for k, v in rep.items() if k in labels},
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 3),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 3),
        "accuracy": round(float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def run(train: pd.DataFrame, tc10: pd.DataFrame) -> dict:
    cfg = load_config()
    seed = cfg["random_seed"]
    X, y_lab = features(train), train["label"].values
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(y_lab)
    skf = StratifiedKFold(n_splits=cfg["dga"]["cv_folds"], shuffle=True, random_state=seed)

    out = {"cv": {}}
    # Duval Triangle 1 baseline on the same samples (DT / ND count as errors)
    duval_train = diagnose_frame(train, screen_normal=False)
    out["cv"]["Duval Triangle 1 (rule-based)"] = report(y_lab, duval_train.values, CLASSES)

    best_name, best_f1 = None, -1
    for name, (kind, model) in _models(seed).items():
        oof = np.empty(len(y), dtype=int)
        for tr_idx, va_idx in skf.split(X, y):
            m = _fit(kind, _models(seed)[name][1], X.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict(X.iloc[va_idx])
        rep = report(y_lab, le.inverse_transform(oof), CLASSES)
        out["cv"][name] = rep
        if rep["macro_f1"] > best_f1:
            best_name, best_f1 = name, rep["macro_f1"]

    kind, model = _models(seed)[best_name]
    model = _fit(kind, model, X, y)
    out["selected_model"] = best_name

    # External, unseen IEC TC 10 benchmark (T1 and T2 merged as the database does)
    tc_labels = ["PD", "D1", "D2", "T12", "T3"]
    merge = lambda s: pd.Series(s).replace({"T1": "T12", "T2": "T12"}).values
    ml_tc = merge(le.inverse_transform(model.predict(features(tc10))))
    duval_tc = merge(diagnose_frame(tc10, screen_normal=False).values)
    out["iec_tc10"] = {
        "n": len(tc10),
        "ML": report(tc10["label"].values, ml_tc, tc_labels),
        "Duval Triangle 1": report(tc10["label"].values, duval_tc, tc_labels),
        "duval_DT_or_ND_count": int(np.isin(duval_tc, ["DT", "ND"]).sum()),
    }
    return {"results": out, "model": model, "label_encoder": le}


def predict_fault(model, le, gases: dict) -> dict:
    """Single-sample inference used by the health index / API."""
    row = pd.DataFrame([{g: float(gases.get(g, 0) or 0) for g in ("H2", "CH4", "C2H6", "C2H4", "C2H2")}])
    proba = model.predict_proba(features(row))[0]
    order = np.argsort(proba)[::-1]
    return {"class": le.inverse_transform([order[0]])[0],
            "probability": round(float(proba[order[0]]), 3),
            "top3": {le.inverse_transform([i])[0]: round(float(proba[i]), 3) for i in order[:3]}}
