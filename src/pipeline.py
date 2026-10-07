"""End-to-end pipeline:  python -m src.pipeline

1 data      ETT + synthetic signals
2 thermal   IEC 60076-7 hot-spot / loss of life, 24 h top-oil forecast vs baselines
3 DGA       Duval Triangle 1 baseline vs ML classifier (CV + unseen IEC TC 10)
4 HI        daily health index with drivers
5 MLOps     model registry, drift report, retraining triggers
6 advisory  one-page brief (md + pdf)
7 RAG       index + evaluation questions
8 report    figures + technical note PDF
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd

from src.config import load_config, out_dir, p, save_json
from src.data import synthetic as S
from src.data.ett import load_ett, load_factor
from src.dga import classifier as dga_clf
from src.dga.datasets import load_iec_tc10, load_training
from src.health_index.timeline import daily_snapshots, score_timeline, snapshot_from_row
from src.health_index.scoring import compute
from src.mlops import drift, registry
from src.thermal import forecast
from src.thermal.iec60076_7 import loss_of_life_summary, simulate

RAG_EVAL = [
    ("What does acetylene in transformer oil indicate?", True),
    ("How is a Buchholz relay used to detect transformer faults?", True),
    ("What are the limits for dielectric breakdown voltage of in-service oil?", True),
    ("How should moisture in transformer insulation be assessed?", True),
    ("What maintenance is needed for transformer cooling fans and radiators?", True),
    ("What does a high CO2/CO ratio mean for paper insulation?", True),
    ("What is the capital of France?", False),
    ("Who won the cricket world cup?", False),
    ("Write me a poem about the sea.", False),
]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main() -> dict:
    t0 = time.time()
    cfg = load_config()
    out = out_dir()
    summary: dict = {}

    # 1 -------------------------------------------------------------- data
    df = load_ett()
    K = load_factor(df)
    amb = S.ambient_temperature(df.index)
    elec = S.electrical_signals(K)
    log(f"ETT loaded: {len(df)} rows {df.index[0]} .. {df.index[-1]}", t0)

    # 2 -------------------------------------------------------------- thermal
    th = simulate(K, amb)
    noise = cfg["synthetic"]["sensor_noise_c"]
    r = np.random.default_rng(cfg["random_seed"] + 9)
    th["oti_c"] = th["top_oil_c"] + r.normal(0, noise, len(th))
    th["wti_c"] = th["hot_spot_c"] + r.normal(0, noise, len(th))     # WTI = thermal image of hot-spot
    life = loss_of_life_summary(th)
    summary["thermal_iec60076_7"] = life
    pd.concat([th, elec], axis=1).iloc[::4].to_csv(out / "online_15min_hourly_sample.csv")
    log(f"IEC 60076-7: max hot-spot {life['max_hot_spot_c']:.1f} C, life used {life['life_consumed_pct']:.2f} %", t0)

    fc = forecast.run(df)
    summary["forecast"] = fc["results"]
    for v, m in fc["models"].items():
        registry.save(f"top_oil_forecast_{v}", m, fc["results"][v], [p(cfg["paths"]["ett_csv"])],
                      {"features": forecast.feature_cols(fc["frames"]["_test_inputs"]) if v == "causal" else None,
                       "horizon_steps": forecast.H, "split": fc["results"]["split"]})
    fc["frames"]["causal"].to_csv(out / "forecast_test_predictions.csv")
    log(f"Forecast MAE test: causal {fc['results']['causal']['test']['MAE']:.3f}, day-ahead "
        f"{fc['results']['day_ahead']['test']['MAE']:.3f}, persistence "
        f"{fc['results']['baselines_test']['Persistence (24 h seasonal naive)']['MAE']:.3f}", t0)

    # 3 -------------------------------------------------------------- DGA
    train, info = load_training()
    tc10 = load_iec_tc10()
    dres = dga_clf.run(train, tc10)
    summary["dga"] = {"data": info, **dres["results"]}
    registry.save("dga_classifier", {"model": dres["model"], "label_encoder": dres["label_encoder"]},
                  {"cv_macro_f1": dres["results"]["cv"][dres["results"]["selected_model"]]["macro_f1"],
                   "iec_tc10_macro_f1": dres["results"]["iec_tc10"]["ML"]["macro_f1"]},
                  [p(cfg["paths"]["dga_train_xlsx"]), p(cfg["paths"]["dga_iec_tc10_csv"])],
                  {"selected": dres["results"]["selected_model"], "classes": list(dres["label_encoder"].classes_)})
    log(f"DGA: selected {dres['results']['selected_model']}", t0)

    # 4 -------------------------------------------------------------- health index
    dga_tl = S.dga_timeline(df.index)
    oil = S.oil_quality_tests(df.index)
    events = S.protection_events(th)
    daily = daily_snapshots(th, dga_tl, oil, events)
    hi = score_timeline(daily, dres["model"], dres["label_encoder"])
    hi.to_csv(out / "health_index_daily.csv")
    daily.to_csv(out / "daily_features.csv")
    dga_tl.to_csv(out / "synthetic_dga_daily.csv")
    oil.to_csv(out / "synthetic_oil_tests.csv")
    events.to_csv(out / "synthetic_events.csv", index=False)

    last_row = daily.iloc[-1]
    latest_snapshot = snapshot_from_row(last_row)
    latest_fault = dga_clf.predict_fault(dres["model"], dres["label_encoder"], latest_snapshot["dga"])
    latest = compute(latest_snapshot, latest_fault)
    save_json({"snapshot": latest_snapshot, "result": latest, "as_of": str(daily.index[-1].date())}, out / "latest_health_index.json")
    # early-detection check against the injected synthetic fault
    onset = daily.index[-1] - pd.Timedelta(days=cfg["synthetic"]["dga"]["fault_onset_days_before_end"])
    buchholz = events.loc[events["event"] == "BUCHHOLZ_ALARM", "timestamp"].min().normalize()
    after = hi.loc[onset:]
    first_abn = after.index[after["dga_status"].eq("Abnormal")].min()
    first_fair = after.index[after["band"].isin(["Fair", "Poor", "Very Poor"])].min()
    pre = hi.loc[:onset - pd.Timedelta(days=1)]
    detection = {"fault_onset": str(onset.date()), "first_dga_abnormal": str(first_abn.date()),
                 "first_band_fair_or_worse": str(first_fair.date()), "buchholz_alarm": str(buchholz.date()),
                 "lead_days_before_buchholz": int((buchholz - first_abn).days),
                 "ml_class_at_detection": str(hi.at[first_abn, "ml_class"]), "duval_at_detection": str(hi.at[first_abn, "duval"]),
                 "pre_onset_days": len(pre), "pre_onset_dga_false_alarms": int(pre["dga_status"].eq("Abnormal").sum())}
    summary["health_index"] = {"latest": latest, "detection": detection, "band_days": hi["band"].value_counts().to_dict(),
                               "examples": hi.iloc[[0, 365, 560, 650, 700, -1]].reset_index().astype(str).to_dict("records")}
    log(f"Health index latest: {latest['health_index']} ({latest['band']})", t0)

    # 5 -------------------------------------------------------------- MLOps
    tr, va, te = forecast.split(forecast.build_dataset(df))
    key_inputs = ["ot", "d_mean_672", "d_seasonal_7d", "HUFL_dev_24h", "std_96"]
    recent = te.loc[te.index >= te.index[-1] - pd.Timedelta(days=30)]
    # same-season reference (the same 30 days one year earlier) so normal seasonality is not flagged
    year = pd.Timedelta(days=365)
    reference = tr.loc[recent.index[0] - year: recent.index[-1] - year]
    drift_rep = drift.data_drift(reference, recent, key_inputs)
    ot_shift = {"reference_window": [str(reference.index[0].date()), str(reference.index[-1].date())],
                "ot_mean_reference": round(float(reference["ot"].mean()), 1),
                "ot_mean_recent": round(float(recent["ot"].mean()), 1)}
    perf = drift.performance_drift(fc["frames"]["causal"], fc["results"]["causal"]["validation_MAE"])
    triggers = drift.retraining_triggers(drift_rep, perf, new_labelled_dga=0, days_since_training=0)
    summary["mlops"] = {"data_drift_last_30d_vs_train": drift_rep, "drift_context": ot_shift,
                        "forecast_performance_drift": perf,
                        "retraining_triggers_now": triggers}
    drift.rolling_mae(fc["frames"]["causal"]).to_frame("rolling_mae_7d").to_csv(out / "forecast_rolling_mae.csv")
    log(f"Drift: triggers = {triggers}", t0)

    # 6 -------------------------------------------------------------- advisory
    from src.advisory import advisory as adv
    ev30 = events[events["timestamp"] >= events["timestamp"].max() - pd.Timedelta(days=30)]["event"].unique().tolist()
    a = adv.build(latest, life, sorted(ev30), str(daily.index[-1].date()))
    rep_dir = out_dir("reports_dir")
    (rep_dir / "maintenance_advisory.md").write_text(adv.to_markdown(a), encoding="utf-8")
    adv.to_pdf(a, rep_dir / "maintenance_advisory.pdf")
    summary["advisory"] = a
    log("Advisory written", t0)

    # 7 -------------------------------------------------------------- RAG
    from src.rag import assistant as rag
    rag.build_index()
    rag_rows = []
    for q, should_answer in RAG_EVAL:
        res = rag.ask(q)
        rag_rows.append({"question": q, "expected_answerable": should_answer, "answered": res["grounded"],
                         "correct_behaviour": res["grounded"] == should_answer, "mode": res["mode"],
                         "citations": [f"{c['document']} p.{c['page']}" for c in res["citations"]],
                         "answer": res["answer"][:600]})
    summary["rag_eval"] = {"accuracy_of_answer_or_refuse": sum(r["correct_behaviour"] for r in rag_rows) / len(rag_rows),
                           "rows": rag_rows}
    log(f"RAG eval: {summary['rag_eval']['accuracy_of_answer_or_refuse']:.0%} correct answer/refuse decisions", t0)

    save_json(summary, out / "summary.json")

    # 8 -------------------------------------------------------------- figures + technical note
    from src.reporting import data_figures, figures, tech_note
    figures.make_all(th, fc, dres, hi, perf)
    data_figures.make_all()
    tech_note.build(summary)
    log("Figures and technical note written - done", t0)
    return summary


if __name__ == "__main__":
    main()
