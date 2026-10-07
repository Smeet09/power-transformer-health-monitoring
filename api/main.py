"""FastAPI service.

POST /health-index          score a transformer snapshot (thermal, loading, DGA, oil, events)
GET  /health-index/latest   latest score of the monitored unit from the nightly pipeline
POST /ask                   maintenance Q&A over O&M references (RAG, cited, guarded)
GET  /healthz               liveness + served model versions
Docs: /docs (OpenAPI)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path
from pydantic import BaseModel, Field

from src.config import load_config, out_dir, p
from src.dga.classifier import predict_fault
from src.health_index.scoring import compute
from src.mlops import registry
from src.thermal.iec60076_7 import steady_state

app = FastAPI(title="Transformer Health Index API", version=load_config()["mlops"]["model_version"],
              description="10 MVA 33/11 kV transformer health monitoring prototype")
_STATE: dict = {}


def _dga_model():
    if "dga" not in _STATE:
        try:
            obj, meta = registry.load("dga_classifier")
        except FileNotFoundError as e:
            raise HTTPException(503, "Models not trained yet - run `python -m src.pipeline`") from e
        _STATE["dga"] = (obj["model"], obj["label_encoder"], meta)
    return _STATE["dga"]


# ----------------------------------------------------------------- schemas
class Thermal(BaseModel):
    hot_spot_max_c: float | None = Field(None, description="Max hot-spot / WTI over last 24 h")
    top_oil_c: float | None = Field(None, description="Measured top-oil (OTI) at peak load")
    ambient_c: float | None = Field(None, description="Ambient at peak load (used if no OTI/WTI)")
    ageing_rate_30d: float | None = Field(None, description="Mean IEC 60076-7 relative ageing rate, 30 d")
    alarm_count_30d: int = 0


class Loading(BaseModel):
    peak_load_pu: float | None = Field(None, description="Peak load in per-unit of rating, 24 h")
    lv_current_a: list[float] | None = Field(None, description="Peak LV phase currents [A, B, C] in amps")
    hours_over_1pu_30d: float = 0.0


class DGA(BaseModel):
    H2: float
    CH4: float
    C2H2: float
    C2H4: float
    C2H6: float
    CO: float | None = None
    CO2: float | None = None
    tdcg_rate_ppm_per_day: float | None = None


class Oil(BaseModel):
    moisture_ppm: float | None = None
    bdv_kv: float | None = None


class Events(BaseModel):
    buchholz_alarm: bool = False
    buchholz_trip: bool = False
    prd_operated: bool = False


class HealthIndexRequest(BaseModel):
    transformer_id: str = "TX-33-11-01"
    thermal: Thermal = Thermal()
    loading: Loading = Loading()
    dga: DGA
    oil: Oil = Oil()
    events: Events = Events()

    model_config = {"json_schema_extra": {"example": {
        "transformer_id": "TX-33-11-01",
        "thermal": {"top_oil_c": 78, "ageing_rate_30d": 0.6, "alarm_count_30d": 1},
        "loading": {"lv_current_a": [470, 455, 480], "hours_over_1pu_30d": 4},
        "dga": {"H2": 180, "CH4": 410, "C2H2": 2.5, "C2H4": 330, "C2H6": 150, "CO": 520, "CO2": 6200, "tdcg_rate_ppm_per_day": 15},
        "oil": {"moisture_ppm": 26, "bdv_kv": 41},
        "events": {"buchholz_alarm": True}}}}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What does acetylene in transformer oil indicate?"])


# ----------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    """Opening the bare address goes to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/healthz")
def healthz():
    versions = {}
    base = out_dir("models_dir")
    for d in base.iterdir() if base.exists() else []:
        if (d / "LATEST").exists():
            versions[d.name] = (d / "LATEST").read_text().strip()
    from src.rag.assistant import backend_status
    return {"status": "ok", "models": versions, "rag": backend_status()}


@app.post("/health-index")
def health_index(req: HealthIndexRequest):
    cfg = load_config()
    model, le, meta = _dga_model()
    derived = {}

    K = req.loading.peak_load_pu
    if K is None and req.loading.lv_current_a:
        K = max(req.loading.lv_current_a) / cfg["transformer"]["lv_rated_a"]
        derived["peak_load_pu"] = round(K, 3)

    hs = req.thermal.hot_spot_max_c
    if hs is None and K is not None:
        tc = cfg["thermal"]
        if req.thermal.top_oil_c is not None:          # measured OTI + IEC hot-spot gradient
            hs = req.thermal.top_oil_c + tc["H"] * tc["gr_k"] * K ** tc["y"]
            derived["hot_spot_method"] = "OTI + IEC 60076-7 gradient H*gr*K^y"
        elif req.thermal.ambient_c is not None:        # steady-state IEC model
            _, hs = steady_state(K, req.thermal.ambient_c)
            derived["hot_spot_method"] = "IEC 60076-7 steady state from ambient and K"
        if hs is not None:
            derived["hot_spot_max_c"] = round(hs, 1)

    snapshot = {
        "thermal": {"hot_spot_max_c": hs, "ageing_rate_30d": req.thermal.ageing_rate_30d,
                    "alarm_count_30d": req.thermal.alarm_count_30d},
        "loading": {"peak_load_pu": K, "hours_over_1pu_30d": req.loading.hours_over_1pu_30d},
        "dga": req.dga.model_dump(),
        "oil": req.oil.model_dump(),
        "events": req.events.model_dump(),
    }
    fault = predict_fault(model, le, snapshot["dga"])
    result = compute(snapshot, fault)
    resp = {"transformer_id": req.transformer_id, **result, "dga_ml_top3": fault["top3"], "derived_inputs": derived,
            "model_version": meta["version"], "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # request log -> input for drift monitoring
    with open(out_dir() / "api_requests.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": resp["scored_at"], "input": snapshot, "hi": result["health_index"]}, default=float) + "\n")
    return resp


@app.get("/health-index/latest")
def latest():
    f = out_dir() / "latest_health_index.json"
    if not f.exists():
        raise HTTPException(503, "Pipeline has not run yet - run `python -m src.pipeline`")
    return json.loads(f.read_text(encoding="utf-8"))


@app.post("/ask")
def ask(req: AskRequest):
    from src.rag.assistant import ask as rag_ask
    res = rag_ask(req.question)
    # add a link that opens the cited manual at the cited page (used by the /assistant chat page)
    titles = [d["title"] for d in load_config()["paths"]["rag_docs"]]
    for c in res.get("citations", []):
        if c["document"] in titles:
            c["url"] = f"/references/{titles.index(c['document'])}#page={c['page']}"
    return res


STATIC = Path(__file__).parent / "static"


@app.get("/assistant", include_in_schema=False)
def assistant_page():
    """Simple chat page for the RAG assistant (calls POST /ask)."""
    return FileResponse(STATIC / "assistant.html")


@app.get("/references/{doc_id}", include_in_schema=False)
def reference_pdf(doc_id: int):
    """Serve the O&M manual PDFs so citations can be checked against the source page."""
    docs = load_config()["paths"]["rag_docs"]
    if not 0 <= doc_id < len(docs):
        raise HTTPException(404, "Unknown reference")
    return FileResponse(p(docs[doc_id]["path"]), media_type="application/pdf")
