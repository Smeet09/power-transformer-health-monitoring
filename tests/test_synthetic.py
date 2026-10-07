"""Verification of the synthetic data: each test checks one property the generated signal must
have to be physically / standards-consistent with the scenario stated in config.yaml.

Reference keys [R1]..[R9] are listed in README.md -> "References for assumptions and checks".
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import synthetic as S
from src.data.ett import load_ett, load_factor
from src.dga.duval import diagnose, is_normal
from src.thermal.iec60076_7 import simulate

CFG = load_config()


@pytest.fixture(scope="module")
def base():
    df = load_ett()
    K = load_factor(df)
    amb = S.ambient_temperature(df.index)
    th = simulate(K, amb)
    th["oti_c"], th["wti_c"] = th["top_oil_c"], th["hot_spot_c"]
    return df, K, amb, th


# ---------------- reproducibility
def test_seeded_reproducible(base):
    df = base[0]
    pd.testing.assert_series_equal(S.ambient_temperature(df.index), S.ambient_temperature(df.index))
    pd.testing.assert_frame_equal(S.dga_timeline(df.index), S.dga_timeline(df.index))


# ---------------- ambient
def test_ambient_climate_plausible(base):
    _, _, amb, _ = base
    a = CFG["synthetic"]["ambient"]
    monthly = amb.groupby(amb.index.month).mean()
    assert np.allclose(monthly.values, a["monthly_mean_c"], atol=1.5)  # WMO/JMA normals, Ahmedabad [R1]
    assert 10 < amb.min() and amb.max() < 45                       # Western-India range
    hourly = amb.groupby(amb.index.hour).mean()
    assert 13 <= hourly.idxmax() <= 16                             # afternoon peak
    assert monthly.idxmax() in (4, 5, 6) and monthly.idxmin() in (12, 1, 2)
    assert monthly[[7, 8, 9]].mean() < monthly[5]                   # monsoon cooling


# ---------------- electrical
def test_currents_consistent_with_load_and_rating(base):
    _, K, _, _ = base
    e = S.electrical_signals(K)
    t = CFG["transformer"]
    lv = e[["i_lv_a_a", "i_lv_b_a", "i_lv_c_a"]].mean(axis=1)
    hv = e[["i_hv_a_a", "i_hv_b_a", "i_hv_c_a"]].mean(axis=1)
    dev = (lv / t["lv_rated_a"] - K).abs()                          # I = K * I_rated (+ 3 % imbalance noise)
    assert dev.mean() < 0.01 and dev.quantile(0.999) < 0.06
    assert abs((lv / hv).median() - t["lv_rated_a"] / t["hv_rated_a"]) < 0.05  # turns ratio 33/11 = 3
    imb = e[["i_lv_a_a", "i_lv_b_a", "i_lv_c_a"]].std(axis=1) / lv
    assert imb.median() < 0.05                                      # ~3 % phase imbalance


def test_oltc_regulates_lv_voltage(base):
    _, K, _, _ = base
    e = S.electrical_signals(K)
    t = CFG["transformer"]
    assert e["tap_position"].between(1, t["oltc_taps"]).all()
    assert np.corrcoef(K, e["tap_position"])[0, 1] > 0.8            # tap rises with load
    v = e[["v_lv_a_kv", "v_lv_b_kv", "v_lv_c_kv"]].mean(axis=1)
    assert v.between(0.95 * t["lv_kv"], 1.05 * t["lv_kv"]).mean() > 0.99   # held within +/-5 %


# ---------------- thermal + events
def test_thermal_ordering(base):
    th = base[3]
    assert (th["top_oil_c"] > th["ambient_c"]).mean() > 0.99        # oil hotter than air
    assert (th["hot_spot_c"] >= th["top_oil_c"]).all()              # winding never colder than top oil


def test_events_match_temperature_thresholds(base):
    th = base[3]
    ev = S.protection_events(th)
    al = CFG["thermal"]["alarms"]
    for _, r in ev[ev["event"].str.startswith(("OTI", "WTI"))].iterrows():   # thresholds = assumption [R9]
        name, level = r["event"].split("_")
        assert r["value"] >= al[f"{name.lower()}_{level.lower()}_c"]
    assert (ev["event"] == "BUCHHOLZ_ALARM").sum() == 1


# ---------------- DGA scenario
def test_dga_healthy_then_t2_fault(base):
    d = S.dga_timeline(base[0].index)
    onset = len(d) - CFG["synthetic"]["dga"]["fault_onset_days_before_end"]
    early = d.iloc[: onset - 30]
    assert early.apply(is_normal, axis=1).mean() > 0.95             # below IEC 60599 typical values before onset [R2]
    assert diagnose(d.iloc[-1]) == "T2"                             # Duval confirms injected fault type
    assert d["C2H2"].max() < 5     # thermal fault: only trace C2H2 ("a few ppm" only above ~500 C, FIST 3-31 p.7 [R4])
    assert (d.iloc[-30:]["C2H4"].mean() > 10 * early["C2H4"].mean())
    assert (d >= 0).all().all()


def test_paper_ageing_gases_realistic(base):
    d = S.dga_timeline(base[0].index)
    ratio = d["CO2"] / d["CO"]
    # CO2/CO < 3 or > 10 suggests abnormal cellulose degradation (USBR FIST 3-30 p.59 [R3]; IEC 60599 limit 3 [R2])
    assert ratio.between(3, 10).mean() > 0.95


# ---------------- oil quality
def test_oil_tests_quarterly_and_degrading(base):
    o = S.oil_quality_tests(base[0].index)
    gaps = o.index.to_series().diff().dropna().dt.days
    assert gaps.between(85, 95).all()                               # quarterly
    assert o["moisture_ppm"].iloc[-1] > o["moisture_ppm"].iloc[0]
    assert o["bdv_kv"].iloc[-1] < o["bdv_kv"].iloc[0]
    assert np.corrcoef(o["moisture_ppm"], o["bdv_kv"])[0, 1] < -0.8  # wetter oil -> lower BDV (Vaisala white paper [R7])
    # scenario stays inside IEC 60422 cat. C "good"/"fair" bands, never "poor" (water > 40 ppm, BDV < 30 kV) [R6]
    assert o["moisture_ppm"].between(5, 40).all() and o["bdv_kv"].between(30, 75).all()
    assert o["moisture_ppm"].iloc[-1] > 30 and o["bdv_kv"].iloc[-1] < 40   # ends in the "fair" band
