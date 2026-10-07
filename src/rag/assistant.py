"""Maintenance Q&A assistant over public O&M references (USBR FIST 3-30, FIST 3-31).

Retrieval : PDF pages -> overlapping chunks -> TF-IDF (word 1-2 grams) cosine similarity.
            Lightweight and deterministic, no model download inside the container.
Answering : everything runs locally - no hosted API (client requirement).
            * LLM mode: a locally hosted open-source model (default Qwen2.5-3B-Instruct) served by
              Ollama answers ONLY from the retrieved chunks and must cite them as [n].
            * Extractive mode (Ollama not running, or RAG_MODE=extractive): returns the most
              relevant sentences verbatim, with citations - always grounded, works offline.
Guardrails: 1. refuse when the best chunk similarity < rag.min_similarity (out of scope)
            2. system prompt forbids outside knowledge; model returns INSUFFICIENT_CONTEXT
               when sources don't support an answer -> refusal
            3. sentence-level grounding check: every LLM sentence must cite a valid source [n],
               >= 60 % of its content words and all of its numbers must appear in the cited
               chunk; failing sentences are removed; if none survive -> extractive answer
            4. safety notice appended (de-energise / follow switching procedures)
"""
from __future__ import annotations

import os
import re

import httpx
import joblib
import numpy as np
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config, out_dir, p

REFUSAL = ("I could not find support for this in the loaded O&M references (USBR FIST 3-30 / 3-31), "
           "so I will not answer it. Please rephrase as a transformer maintenance question or consult the OEM manual.")
SAFETY = "Safety: perform any internal work only after isolation, earthing and permit-to-work per your utility's switching procedure."
INDEX_FILE = "rag_index.joblib"


DOMAIN_TERMS = re.compile(
    r"transformer|oil|gas|dga|acetylene|hydrogen|methane|ethylene|ethane|carbon|winding|bushing|relay|buchholz|"
    r"moisture|water|temperature|hot.?spot|insulation|paper|tap|oltc|changer|cooling|radiator|fan|pump|conservator|"
    r"breather|desiccant|silica|dielectric|breakdown|bdv|power factor|tan delta|megger|resistance|core|tank|leak|"
    r"pressure|relief|arc|discharge|overheat|thermal|load|maintenance|inspection|test|sampl|fault|alarm|trip|nitrogen|"
    r"furan|sfra|frequency response|excitation|thermograph|infrared|gauge|valve|gasket|corona|sludge|acid|ifs|"
    r"interfacial|filter|degas|dehydrat|lightning|arrester|ground|earth", re.I)


def _clean(text: str) -> str:
    text = text.replace("�", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def build_index() -> dict:
    cfg = load_config()["rag"]
    chunks = []
    for doc in load_config()["paths"]["rag_docs"]:
        pdf = pymupdf.open(p(doc["path"]))
        for pno, page in enumerate(pdf, start=1):
            text = _clean(page.get_text())
            letters = sum(ch.isalpha() for ch in text)
            if len(text) < 200 or letters / max(len(text), 1) < 0.6:   # skip blank / garbled pages
                continue
            step = cfg["chunk_chars"] - cfg["chunk_overlap"]
            for start in range(0, max(len(text) - cfg["chunk_overlap"], 1), step):
                piece = text[start:start + cfg["chunk_chars"]]
                if len(piece) > 150:
                    chunks.append({"doc": doc["title"], "page": pno, "text": piece})
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    X = vec.fit_transform([c["text"] for c in chunks])
    index = {"chunks": chunks, "vectorizer": vec, "matrix": X}
    joblib.dump(index, out_dir("models_dir") / INDEX_FILE)
    return index


_INDEX = None


def get_index() -> dict:
    global _INDEX
    if _INDEX is None:
        f = out_dir("models_dir") / INDEX_FILE
        _INDEX = joblib.load(f) if f.exists() else build_index()
    return _INDEX


def retrieve(question: str, k: int | None = None) -> list[dict]:
    idx = get_index()
    k = k or load_config()["rag"]["top_k"]
    sims = cosine_similarity(idx["vectorizer"].transform([question]), idx["matrix"])[0]
    top = np.argsort(sims)[::-1][:k]
    return [{**idx["chunks"][i], "score": float(sims[i])} for i in top]


def _extractive(question: str, hits: list[dict], n_sent: int = 4) -> str:
    vec = get_index()["vectorizer"]
    cands = []
    for sid, h in enumerate(hits, start=1):
        for s in re.split(r"(?<=[.!?])\s+", h["text"].replace("\n", " ")):
            if 40 < len(s) < 500:
                cands.append((s.strip(), sid))
    if not cands:
        return ""
    sims = cosine_similarity(vec.transform([question]), vec.transform([c[0] for c in cands]))[0]
    best = sorted(np.argsort(sims)[::-1][:n_sent])
    return " ".join(f"{cands[i][0]} [{cands[i][1]}]" for i in best if sims[i] > 0)


SYSTEM_PROMPT = """You are a maintenance assistant for an oil-immersed 10 MVA, 33/11 kV distribution power transformer.
Answer ONLY from the numbered SOURCES provided in the user message. Do not use outside knowledge, even if you know the answer.
Cite every factual sentence with its source number in square brackets, e.g. [2]. Keep the answer under 180 words, practical, and in plain English.
If the sources do not contain enough information to answer, reply with exactly: INSUFFICIENT_CONTEXT"""


_STOP = set("""a an the and or of to in on for with by at from as is are was were be been being this that these those it its
which what when where how why can could should would may might must will shall not no if then than also any all each other such
into over under about between your their there here they them he she we you our""".split())


def ground_check(answer: str, hits: list[dict], min_support: float = 0.6) -> tuple[str, int]:
    """Sentence-level grounding guardrail for LLM answers. A sentence is kept only if it cites a
    source [n] and (a) >= min_support of its content words and (b) every number in it appear in
    the cited chunk(s). Returns (kept text, number of sentences removed)."""
    kept, removed = [], 0
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer):
        s = sent.strip()
        if not s or s.endswith(":"):
            continue
        ids = {int(n) for n in re.findall(r"\[(\d+)\]", s) if 1 <= int(n) <= len(hits)}
        if not ids:
            removed += 1
            continue
        src = " ".join(hits[i - 1]["text"] for i in ids).lower()
        body = re.sub(r"\[\d+\]", " ", s).lower()
        words = [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", body) if w not in _STOP]
        support = sum(w in src for w in words) / max(len(words), 1)
        numbers_ok = all(n in src for n in re.findall(r"\d+(?:\.\d+)?", body))
        if support >= min_support and numbers_ok:
            kept.append(s)
        else:
            removed += 1
    return " ".join(kept), removed


def _ollama_url() -> str:
    return os.getenv("OLLAMA_URL", load_config()["rag"]["ollama_url"]).rstrip("/")


def _llm_answer(question: str, hits: list[dict]) -> str | None:
    """Ask the locally hosted model (Ollama /api/chat). Returns None if it can't be reached."""
    cfg = load_config()["rag"]
    sources = "\n\n".join(f"[{i}] ({h['doc']}, p. {h['page']})\n{h['text']}" for i, h in enumerate(hits, start=1))
    payload = {
        "model": cfg["llm_model"],
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"SOURCES:\n{sources}\n\nQUESTION: {question}"}],
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400},
    }
    try:
        r = httpx.post(f"{_ollama_url()}/api/chat", json=payload, timeout=cfg["llm_timeout_s"])
        r.raise_for_status()
    except httpx.HTTPError:
        return None   # server down / model missing / timeout -> caller falls back to extractive mode
    return r.json().get("message", {}).get("content", "").strip()


def _llm_available() -> bool:
    if os.getenv("RAG_MODE", "auto") == "extractive":
        return False
    try:
        tags = httpx.get(f"{_ollama_url()}/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return False
    model = load_config()["rag"]["llm_model"]
    return any(m.get("name", "").split(":latest")[0] == model.split(":latest")[0] for m in tags.get("models", []))


def backend_status() -> dict:
    """Which answering backend /ask will use right now (shown by /healthz)."""
    return {"llm_backend": "ollama", "model": load_config()["rag"]["llm_model"], "url": _ollama_url(),
            "available": _llm_available()}


def ask(question: str) -> dict:
    cfg = load_config()["rag"]
    question = (question or "").strip()[:1000]
    hits = retrieve(question) if question else []
    citations = [{"id": i, "document": h["doc"], "page": h["page"], "score": round(h["score"], 3)}
                 for i, h in enumerate(hits, start=1)]
    if not DOMAIN_TERMS.search(question):
        return {"answer": REFUSAL, "grounded": False, "mode": "guardrail", "citations": [],
                "reason": "question is outside the transformer O&M domain"}
    if not hits or hits[0]["score"] < cfg["min_similarity"]:
        return {"answer": REFUSAL, "grounded": False, "mode": "guardrail", "citations": [],
                "reason": f"best retrieval similarity {hits[0]['score'] if hits else 0:.3f} < {cfg['min_similarity']}"}

    mode, answer, n_removed = "extractive", None, 0
    if _llm_available():
        llm = _llm_answer(question, hits)
        if llm and "INSUFFICIENT_CONTEXT" in llm:     # small local models may wrap the keyword in extra text
            return {"answer": REFUSAL, "grounded": False, "mode": "llm-refused", "citations": citations}
        if llm:
            grounded, n_removed = ground_check(llm, hits)
            if grounded:
                mode, answer = "llm", grounded
    if answer is None:
        answer = _extractive(question, hits)
        if not answer:
            return {"answer": REFUSAL, "grounded": False, "mode": "guardrail", "citations": []}
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return {"answer": f"{answer}\n\n{SAFETY}", "grounded": True, "mode": mode,
            "citations": [c for c in citations if c["id"] in used],
            "sentences_removed_by_grounding_check": n_removed}
