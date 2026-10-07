"""Static figures (PNG) for the technical note. Palette: categorical slots in fixed order
(blue, orange, aqua, yellow), status colours only for health bands, one y-axis per panel."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import load_config, out_dir

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BAND_COL = {"Very Good": "#0ca30c", "Good": "#0ca30c", "Fair": "#fab219", "Poor": "#ec835a", "Very Poor": "#d03b3b"}
SEQ = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(out_dir() / name, dpi=160)
    plt.close(fig)


def forecast_plot(frames):
    c, d = frames["causal"], frames["day_ahead"]
    end = c.index[-1] - pd.Timedelta(days=60)
    w = slice(end - pd.Timedelta(days=14), end)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(c.loc[w].index, c.loc[w, "actual"], color=INK, lw=1.4, label="Actual OT")
    ax.plot(c.loc[w].index, c.loc[w, "persistence"], color="#9a9893", lw=1.2, ls="--", label="Persistence baseline")
    ax.plot(c.loc[w].index, c.loc[w, "pred"], color=S1, label="XGBoost causal")
    ax.plot(d.loc[w].index, d.loc[w, "pred"], color=S2, label="XGBoost + day-ahead load")
    ax.set_title("24 h-ahead top-oil temperature forecast - 14-day test window (ETTm1)")
    ax.set_ylabel("Oil temperature (C)")
    ax.legend(ncol=4, loc="upper left", fontsize=8)
    _save(fig, "fig_forecast.png")


def thermal_plot(th):
    day = th["hot_spot_c"].resample("D").max().idxmax()
    w = slice(day - pd.Timedelta(days=3), day + pd.Timedelta(days=3))
    al = load_config()["thermal"]["alarms"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    a1.plot(th.loc[w].index, th.loc[w, "K"], color=S1)
    a1.set_ylabel("Load K (pu)")
    a1.set_title("IEC 60076-7 dynamic thermal model - week with the highest hot-spot")
    a2.plot(th.loc[w].index, th.loc[w, "hot_spot_c"], color=S2, label="Hot-spot (WTI)")
    a2.plot(th.loc[w].index, th.loc[w, "top_oil_c"], color=S1, label="Top-oil (OTI)")
    a2.plot(th.loc[w].index, th.loc[w, "ambient_c"], color=S3, label="Ambient (synthetic)")
    for v, lab in ((al["wti_alarm_c"], "WTI alarm"), (al["oti_alarm_c"], "OTI alarm")):
        a2.axhline(v, color=INK2, lw=0.8, ls=":")
        a2.text(th.loc[w].index[-1], v + 1, lab, color=INK2, fontsize=7, ha="right")
    a2.set_ylabel("Temperature (C)")
    a2.set_ylim(20, 150)
    a2.legend(ncol=3, loc="upper left", fontsize=8)
    _save(fig, "fig_thermal.png")


def confusion_plot(dres):
    cv = dres["results"]["cv"]
    sel = dres["results"]["selected_model"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for ax, (name, title) in zip(axes, (("Duval Triangle 1 (rule-based)", "Duval Triangle 1"),
                                        (sel, f"ML: {sel.split(' - ')[0]} (5-fold CV)"))):
        r = cv[name]
        cm = np.array(r["confusion_matrix"])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        labs = r["labels"]
        ax.set_xticks(range(len(labs)), labs)
        ax.set_yticks(range(len(labs)), labs)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nmacro-F1 {r['macro_f1']:.2f}", fontsize=9)
    _save(fig, "fig_dga_confusion.png")


def hi_plot(hi):
    bands = load_config()["health_index"]["bands"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    upper = 100
    for b in bands:
        a1.axhspan(b["min"], upper, color=BAND_COL[b["name"]], alpha=0.10 if b["name"] != "Very Good" else 0.05, lw=0)
        a1.text(hi.index[-1] + pd.Timedelta(days=6), (b["min"] + upper) / 2, b["name"], va="center", fontsize=7, color=INK2)
        upper = b["min"]
    weekly = hi["health_index"].resample("W").min()
    a1.plot(weekly.index, weekly.values, color=S1, lw=1.8)
    a1.set_ylim(0, 100)
    a1.set_ylabel("Health index (weekly min)")
    a1.set_title("Health index timeline - synthetic T2 thermal fault develops in the last 150 days")
    for col, c, lab in (("S_dga", S1, "DGA"), ("S_thermal", S2, "Thermal"), ("S_oil", S3, "Oil quality"), ("S_loading", S4, "Loading")):
        s = hi[col].resample("W").min()
        a2.plot(s.index, s.values, color=c, label=lab, lw=1.4)
    a2.set_ylim(0, 105)
    a2.set_ylabel("Component score")
    a2.legend(ncol=4, loc="lower left", fontsize=8)
    _save(fig, "fig_health_index.png")


def drift_plot(perf_frame_ratio: pd.Series, thr: float):
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(perf_frame_ratio.index, perf_frame_ratio.values, color=S1, label="Rolling 7-day MAE / validation MAE")
    ax.axhline(thr, color="#d03b3b", lw=1, ls="--")
    ax.text(perf_frame_ratio.index[0], thr + 0.05, f"Retrain trigger ({thr}x)", color="#d03b3b", fontsize=7)
    ax.set_title("Forecast performance drift monitor - 2018 test period")
    ax.set_ylabel("MAE ratio")
    _save(fig, "fig_drift.png")


def architecture_plot():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=col, lw=1.4))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.4, color=INK2, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1))

    box(1, 40, 21, 18, "Data sources", "ETTm1 15-min load + OT\nIEC TC 10 + 589 DGA cases\nSynthetic: ambient, V/I,\ntap, oil tests, events", S1)
    box(1, 17, 21, 18, "O&M references", "USBR FIST 3-30\nUSBR FIST 3-31\n(public domain PDFs)", S1)
    box(28, 44, 22, 14, "Thermal analytics", "IEC 60076-7 hot-spot,\nageing, loss of life\nXGBoost 24 h OT forecast", S2)
    box(28, 27, 22, 14, "DGA diagnosis", "Duval Triangle 1 baseline\nXGBoost classifier\n(balanced weights)", S2)
    box(28, 10, 22, 14, "RAG assistant", "TF-IDF retrieval + local LLM\n(Qwen2.5-3B via Ollama)\ncitations, guardrails", S2)
    box(56, 33, 19, 20, "Health index", "4 components -> 0-100\n5 bands\ntop-3 drivers\n(points lost)", S3)
    box(56, 8, 19, 20, "FastAPI (Docker)", "POST /health-index\nGET /health-index/latest\nPOST /ask\nGET /healthz", S3)
    box(80, 33, 19, 20, "Outputs", "Maintenance advisory\n(1 page PDF)\nTechnical note\nfigures, metrics JSON", S4)
    box(80, 8, 19, 20, "MLOps", "file registry + metadata\nPSI data drift\nrolling-MAE drift\nretrain triggers", S4)
    for y in (51, 34):
        arrow(22.8, 49, 27.4, y)
    arrow(22.8, 26, 27.4, 17)
    arrow(50.6, 51, 55.4, 45)
    arrow(50.6, 34, 55.4, 41)
    arrow(50.6, 17, 55.4, 17)      # RAG -> API
    arrow(65.5, 32.6, 65.5, 28.6)  # HI -> API
    arrow(75.6, 43, 79.4, 43)      # HI -> advisory / note
    arrow(75.6, 18, 79.4, 18)      # served models -> monitoring
    _save(fig, "fig_architecture.png")


def make_all(th, fc, dres, hi, perf):
    from src.mlops.drift import rolling_mae
    forecast_plot(fc["frames"])
    thermal_plot(th)
    confusion_plot(dres)
    hi_plot(hi)
    r = rolling_mae(fc["frames"]["causal"]) / perf["validation_mae"]
    r = r[r.index >= r.index[0] + pd.Timedelta(days=7)]
    drift_plot(r, perf["threshold_ratio"])
    architecture_plot()
