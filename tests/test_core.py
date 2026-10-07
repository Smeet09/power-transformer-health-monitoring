import pytest

from src.config import out_dir
from src.dga.duval import diagnose, zone
from src.health_index.scoring import band_for, compute, lin_score
from src.thermal.iec60076_7 import ageing_rate, steady_state


# ---------------- Duval Triangle 1
@pytest.mark.parametrize("ch4,c2h4,c2h2,expected", [
    (99, 0.5, 0.5, "PD"),
    (80, 15, 1, "T1"),
    (60, 38, 2, "T2"),
    (30, 65, 5, "T3"),
    (60, 10, 30, "D1"),
    (20, 30, 50, "D2"),
    (50, 40, 10, "DT"),
])
def test_duval_zones(ch4, c2h4, c2h2, expected):
    assert zone(ch4, c2h4, c2h2) == expected


def test_duval_normal_screen():
    low = {"H2": 20, "CH4": 20, "C2H2": 0, "C2H4": 3, "C2H6": 8}   # %C2H4 = 13 -> T1 zone
    assert diagnose(low) == "Normal"
    assert diagnose(low, screen_normal=False) == "T1"


# ---------------- IEC 60076-7
def test_iec_steady_state_rated_load():
    oil, hs = steady_state(1.0, 20.0)
    assert oil == pytest.approx(72.0)                # 20 + 52 K
    assert hs == pytest.approx(72.0 + 1.3 * 26.0)    # + H*gr


def test_ageing_rate_reference_point():
    assert ageing_rate(98.0) == pytest.approx(1.0)
    assert ageing_rate(104.0) == pytest.approx(2.0)  # doubles every 6 K
    assert ageing_rate(110.0, "upgraded") == pytest.approx(1.0)


# ---------------- health index
def test_lin_score_both_directions():
    assert lin_score(15, good=15, bad=40) == 100
    assert lin_score(40, good=15, bad=40) == 0
    assert lin_score(45, good=60, bad=30) == pytest.approx(50)   # falling scale (BDV)


def test_bands():
    assert band_for(90)["name"] == "Very Good"
    assert band_for(70)["name"] == "Good"
    assert band_for(55)["name"] == "Fair"
    assert band_for(31)["name"] == "Poor"
    assert band_for(10)["name"] == "Very Poor"


HEALTHY = {
    "thermal": {"hot_spot_max_c": 85, "ageing_rate_30d": 0.2, "alarm_count_30d": 0},
    "loading": {"peak_load_pu": 0.7, "hours_over_1pu_30d": 0},
    "dga": {"H2": 30, "CH4": 20, "C2H2": 0.5, "C2H4": 10, "C2H6": 15, "CO": 250, "CO2": 2500},
    "oil": {"moisture_ppm": 10, "bdv_kv": 65},
    "events": {},
}


def test_healthy_unit_scores_very_good():
    r = compute(HEALTHY)
    assert r["band"] == "Very Good"
    assert r["dga_diagnosis"]["status"] == "Normal"


def test_active_dga_fault_cannot_be_rated_good():
    """Good thermal/oil/loading must not average away a diagnosed internal fault."""
    t2 = {**HEALTHY, "dga": {"H2": 180, "CH4": 410, "C2H2": 2.5, "C2H4": 330, "C2H6": 150, "CO": 520, "CO2": 6200}}
    r = compute(t2, {"class": "T2", "probability": 0.99})
    assert r["dga_diagnosis"]["status"] == "Abnormal"
    assert r["band"] == "Fair" and r["health_index"] <= 67
    assert any("overall cap" in d["parameter"] for d in r["top_drivers"])


def test_drivers_explain_the_score_and_events_cap():
    bad = {**HEALTHY, "oil": {"moisture_ppm": 38, "bdv_kv": 32}, "events": {"buchholz_alarm": True}}
    r = compute(bad)
    assert r["health_index"] <= 45
    params = [d["parameter"] for d in r["top_drivers"]]
    assert any("Buchholz" in p for p in params)
    assert any("Moisture" in p or "Breakdown" in p for p in params)


# ---------------- RAG grounding guardrail (no LLM needed)
def test_rag_grounding_check_drops_unsupported_sentences():
    from src.rag.assistant import ground_check
    hits = [{"text": "Acetylene indicates high energy arcing. Generation above 35 ppm requires weekly sampling."}]
    answer = ("Acetylene indicates high energy arcing [1]. "
              "Sampling is needed weekly above 35 ppm [1]. "
              "Acetylene above 90 ppm requires immediate replacement of the bushing [1]. "   # number + words not in source
              "Transformers are usually painted grey.")                                         # no citation
    kept, removed = ground_check(answer, hits)
    assert removed == 2
    assert "arcing" in kept and "35 ppm" in kept and "90" not in kept and "grey" not in kept


# ---------------- API (needs trained models: python -m src.pipeline)
models_ready = (out_dir("models_dir") / "dga_classifier" / "LATEST").exists()


@pytest.mark.skipif(not models_ready, reason="run the pipeline first")
def test_api_health_index_and_guardrail():
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    assert c.get("/healthz").json()["status"] == "ok"
    body = HealthIndexRequest_example()
    r = c.post("/health-index", json=body)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["health_index"] <= 100 and len(j["top_drivers"]) == 3
    assert j["derived_inputs"]["peak_load_pu"] == pytest.approx(480 / 525, abs=1e-3)
    refused = c.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert refused["grounded"] is False
    page = c.get("/assistant")
    assert page.status_code == 200 and "Maintenance Assistant" in page.text
    pdf = c.get("/references/0")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert c.get("/references/99").status_code == 404


def HealthIndexRequest_example():
    from api.main import HealthIndexRequest
    return HealthIndexRequest.model_config["json_schema_extra"]["example"]
