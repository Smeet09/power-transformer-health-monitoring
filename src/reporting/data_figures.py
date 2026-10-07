"""Data-understanding figures for the study guide (raw datasets, before any modelling)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import out_dir
from src.data.ett import LOAD_COLS, load_ett
from src.dga.datasets import CLASSES, CN_LABELS, FIVE_GASES, load_iec_tc10, load_training
from src.reporting.figures import INK2, S1, S2, S3, S4, _save, plt

S5 = "#e87ba4"   # categorical slot 5 (magenta)


def ett_overview():
    df = load_ett()
    d = df.resample("D").mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    a1.plot(d.index, d["OT"], color=S1)
    a1.set_ylabel("OT (C), daily mean")
    a1.set_title("ETTm1 - target OT (oil temperature) and main load column, daily means")
    for yr in (2017, 2018):
        m = d.loc[f"{yr}-06", "OT"]
        a1.annotate(f"June {yr}: {m.mean():.1f} C", (m.index[15], m.mean()), xytext=(0, 18),
                    textcoords="offset points", fontsize=7, color=INK2, ha="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a2.plot(d.index, d["HUFL"], color=S2)
    a2.axhline(0, color=INK2, lw=0.6, ls=":")
    a2.set_ylabel("HUFL, daily mean")
    _save(fig, "fig_data_ett.png")

    corr = df[LOAD_COLS].corrwith(df["OT"])
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.barh(corr.index[::-1], corr.values[::-1], color=S1, height=0.6)
    for y, v in enumerate(corr.values[::-1]):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("Pearson correlation with OT")
    ax.set_title("Load columns correlate only weakly with oil temperature")
    _save(fig, "fig_data_ett_corr.png")


def dga_classes():
    raw = pd.read_excel("Dataset/dga_589.xlsx")
    raw_counts = raw.iloc[:, -1].map(CN_LABELS).value_counts().reindex(CLASSES)
    train, _ = load_training()
    fin = train["label"].value_counts().reindex(CLASSES)
    x = np.arange(len(CLASSES))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 2.9), gridspec_kw={"width_ratios": [3, 2]})
    a1.bar(x - 0.2, raw_counts.values, 0.38, color="#9a9893", label="Raw file (589)")
    a1.bar(x + 0.2, fin.values, 0.38, color=S1, label="After cleaning (473)")
    a1.set_xticks(x, CLASSES)
    a1.set_ylabel("Cases")
    a1.set_title("DGA training set - class counts")
    a1.legend(fontsize=7, loc="upper left")
    tc = load_iec_tc10()["label"].value_counts().reindex(["PD", "D1", "D2", "T12", "T3"])
    a2.bar(tc.index, tc.values, color=S3, width=0.6)
    for i, v in enumerate(tc.values):
        a2.text(i, v + 0.8, str(v), ha="center", fontsize=7, color=INK2)
    a2.set_title("IEC TC 10 test set (113)")
    _save(fig, "fig_data_dga_classes.png")

    comp = train[FIVE_GASES].div(train[FIVE_GASES].sum(axis=1) + 1e-9, axis=0)
    comp["label"] = train["label"]
    med = comp.groupby("label")[FIVE_GASES].mean().reindex(CLASSES) * 100
    fig, ax = plt.subplots(figsize=(8, 2.9))
    left = np.zeros(len(CLASSES))
    for g, c in zip(FIVE_GASES, (S1, S2, S3, S4, S5)):
        ax.barh(CLASSES[::-1], med[g].values[::-1], left=left[::-1], color=c, label=g, height=0.62,
                edgecolor="#fcfcfb", linewidth=1)
        left = left + med[g].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Average share of total gas (%)")
    ax.set_title("Each fault type has a different gas 'fingerprint' (training set)")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    _save(fig, "fig_data_dga_composition.png")


def detection_timeline():
    """How early the developing fault is seen: DGA gases, health index and key dates."""
    out = out_dir()
    dga = pd.read_csv(out / "synthetic_dga_daily.csv", index_col=0, parse_dates=True)
    hi = pd.read_csv(out / "health_index_daily.csv", index_col=0, parse_dates=True)
    ev = pd.read_csv(out / "synthetic_events.csv", parse_dates=["timestamp"])
    start = dga.index[-1] - pd.Timedelta(days=210)
    dga, hi = dga.loc[start:], hi.loc[start:]
    onset = dga.index[-1] - pd.Timedelta(days=150)
    detected = hi.index[hi["dga_status"].eq("Abnormal")][0]
    buchholz = ev.loc[ev["event"] == "BUCHHOLZ_ALARM", "timestamp"].iloc[0].normalize()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    a1.plot(dga.index, dga["CH4"], color=S1, label="CH4 (methane)")
    a1.plot(dga.index, dga["C2H4"], color=S2, label="C2H4 (ethylene)")
    a1.axhline(110, color=S1, lw=0.7, ls=":")
    a1.axhline(280, color=S2, lw=0.7, ls=":")
    a1.text(dga.index[0], 115, "CH4 typical limit 110", fontsize=6.5, color=INK2)
    a1.text(dga.index[0], 285, "C2H4 typical limit 280", fontsize=6.5, color=INK2)
    a1.set_ylabel("ppm")
    a1.set_title("Early detection: fault gases, health index and key dates (synthetic T2 scenario)")
    a1.legend(fontsize=7, loc="upper left")
    a2.plot(hi.index, hi["health_index"], color=S3)
    a2.set_ylim(0, 105)
    a2.set_ylabel("Health index")
    for ax in (a1, a2):
        for d, lab, c in ((onset, "fault starts", INK2), (detected, "DETECTED (T2)", "#1c5cab"),
                          (buchholz, "Buchholz alarm", "#d03b3b")):
            ax.axvline(d, color=c, lw=1, ls="--")
    for d, lab, c in ((onset, "fault starts", INK2), (detected, f"detected: {detected.date()}", "#1c5cab"),
                      (buchholz, "Buchholz alarm", "#d03b3b")):
        a2.text(d + pd.Timedelta(days=2), 8, lab, fontsize=7, color=c)
    a2.annotate("", xy=(buchholz, 30), xytext=(detected, 30),
                arrowprops=dict(arrowstyle="<->", color="#1c5cab", lw=1))
    a2.text(detected + (buchholz - detected) / 2, 33, f"{(buchholz - detected).days} days of warning",
            ha="center", fontsize=8, color="#1c5cab", weight="bold")
    _save(fig, "fig_detection_timeline.png")


def make_all():
    ett_overview()
    dga_classes()
    if (out_dir() / "health_index_daily.csv").exists():
        detection_timeline()


if __name__ == "__main__":
    make_all()
