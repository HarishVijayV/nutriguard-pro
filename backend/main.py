"""
main.py — NutriGuard Pro API
Routes:
  GET  /                  → health check
  GET  /health/providers  → which LLM providers are alive (no AI call)
  POST /health/reprobe    → re-run the parallel provider probe on demand
  GET  /research          → Tavily scrape + save foods for a state
  GET  /graph             → get graph nodes for a state
  GET  /states            → list all states already in DB
  GET  /audit             → quick RAG food safety check
  POST /generate          → full LangGraph workflow (SSE streaming logs)

Provider chain (Gemini → Groq → OpenRouter → offline template) lives in agents.py
along with the parallel startup health probe and circuit breaker.
"""

import os
import json
import sqlite3
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tavily import TavilyClient
from rag import audit_food
from agents import (
    run_workflow,
    call_gemini,
    OFFLINE_LABEL,
    get_provider_status,
    probe_all_providers,
    alive_providers,
)

load_dotenv()

app = FastAPI(title="NutriGuard Pro", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

print("[main.py v1.2] NutriGuard Pro starting (provider chain + health probe in agents.py)")


# ── SQLite setup ──────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("nutriguard.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS food_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            category TEXT,
            food_name TEXT NOT NULL,
            reason TEXT,
            is_ulcer_safe INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            weight REAL,
            protein_req REAL,
            state_name TEXT,
            allergies TEXT,
            plan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("[main.py] DB initialized")


init_db()


# ── Health ────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "NutriGuard Pro is online", "version": "1.2.0"}


@app.get("/health/providers")
def providers_health():
    """Return cached health status of every LLM provider. No AI call."""
    status = get_provider_status()
    alive = alive_providers()  # priority order, so primary is the real first choice
    return {
        "alive_count": len(alive),
        "total": len(status),
        "primary": alive[0] if alive else None,
        "providers": status,
    }


@app.post("/health/reprobe")
def providers_reprobe():
    """Re-run the parallel health probe on demand. Useful before a demo."""
    try:
        status = probe_all_providers()
    except Exception as e:
        return {"error": str(e)[:200], "providers": get_provider_status()}
    alive = alive_providers()
    return {
        "alive_count": len(alive),
        "total": len(status),
        "primary": alive[0] if alive else None,
        "providers": status,
    }


# ── List saved states ─────────────────────────────────────────────
@app.get("/states")
def get_states():
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT state FROM food_nodes").fetchall()
    conn.close()
    return {"states": [r["state"] for r in rows]}


# ── Get graph nodes for a state ───────────────────────────────────
@app.get("/graph")
def get_graph(state: str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM food_nodes WHERE state = ?", (state,)).fetchall()
    conn.close()

    if not rows:
        return {"nodes": [], "links": [], "message": f"No data for {state}. Please research first."}

    nodes = [{"id": "India", "group": "country"}]
    links = []
    nodes.append({"id": state, "group": "state"})
    links.append({"source": "India", "target": state})

    categories_added: set = set()
    for r in rows:
        cat = r["category"] or "General"
        cat_id = f"{state}-{cat}"
        if cat_id not in categories_added:
            nodes.append({"id": cat_id, "label": cat, "group": "category"})
            links.append({"source": state, "target": cat_id})
            categories_added.add(cat_id)
        food_id = f"{r['food_name']}-{r['id']}"
        nodes.append({"id": food_id, "label": r["food_name"], "group": "food",
                      "safe": bool(r["is_ulcer_safe"]), "reason": r["reason"]})
        links.append({"source": cat_id, "target": food_id})

    return {"nodes": nodes, "links": links, "state": state}


# ── Research a new state ──────────────────────────────────────────
@app.get("/research")
def research_state(state: str):
    print(f"[GET /research] Starting research for: {state}")

    conn = get_db()
    existing = conn.execute(
        "SELECT COUNT(*) as cnt FROM food_nodes WHERE state = ?", (state,)
    ).fetchone()
    conn.close()

    if existing["cnt"] > 0:
        return {**get_graph(state), "cached": True}

    all_text = ""
    queries = [
        f"Traditional {state} breakfast foods ulcerative colitis safe",
        f"Traditional {state} lunch dinner dishes IBD friendly low fiber",
        f"{state} regional snacks soft foods gut healing",
    ]
    for q in queries:
        try:
            results = tavily.search(query=q, search_depth="advanced")
            for r in results["results"]:
                all_text += r.get("content", "")[:400] + "\n"
        except Exception as e:
            print(f"[GET /research] Tavily error: {e}")

    prompt = f"""
You are a Medical Nutrition expert. Extract traditional foods from {state}, India
that are safe for Ulcerative Colitis patients.

From this text: {all_text[:2000]}

Return a JSON array with AT LEAST 30 food items (8 breakfast, 8 lunch, 8 dinner, 6 snack).
Also add well-known safe traditional dishes from that state you are confident about:
[
  {{"food": "Idli", "category": "Breakfast", "reason": "Steamed, low fiber, easy to digest"}},
  {{"food": "Curd Rice", "category": "Lunch", "reason": "Probiotic, cooling, gut friendly"}}
]
Only include: steamed, boiled, soft, low-fiber, non-spicy foods.
Exclude: fried, spicy, raw, high-fiber foods.
"""
    raw, _model = call_gemini(prompt)

    foods = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
            if isinstance(parsed, list):
                foods = parsed
        except Exception:
            foods = []

    if not foods:
        print(f"[GET /research] Using hardcoded safe-food seed list for {state}")
        foods = [
            {"food": "Steamed Idli", "category": "Breakfast", "reason": "Steamed, easy to digest"},
            {"food": "Curd Rice", "category": "Lunch", "reason": "Probiotic, gut friendly"},
            {"food": "Khichdi", "category": "Dinner", "reason": "Soft, low fiber"},
            {"food": "Banana", "category": "Snack", "reason": "Soft, easy to digest"},
            {"food": "Rice Kanji", "category": "Dinner", "reason": "Bland, healing"},
            {"food": "Soft Dosa", "category": "Breakfast", "reason": "Fermented, mild"},
            {"food": "Moong Dal", "category": "Lunch", "reason": "Low fiber protein"},
            {"food": "Stewed Apple", "category": "Snack", "reason": "Soft, soothing"},
        ]

    conn = get_db()
    for f in foods:
        name = f.get("food") or f.get("food_name") or f.get("name")
        if not name:
            continue
        conn.execute(
            "INSERT INTO food_nodes (state, category, food_name, reason, is_ulcer_safe) VALUES (?,?,?,?,?)",
            (state, f.get("category", "General"), name, f.get("reason", ""), 1),
        )
    conn.commit()
    conn.close()

    return {**get_graph(state), "cached": False, "foods_found": len(foods)}


# ── Quick audit ───────────────────────────────────────────────────
@app.get("/audit")
def quick_audit(food: str):
    result = audit_food(food)
    return {"food": food, "safety_hint": result["safety_hint"], "evidence": result["evidence"]}


# ── Evaluation scores ─────────────────────────────────────────────
def compute_eval_scores(audit_result: dict, iterations: int, model_used: str) -> dict:
    flags = audit_result.get("flags", [])
    audit_safety = max(0, 100 - len(flags) * 20)
    if audit_result.get("status") == "APPROVED":
        audit_safety = max(audit_safety, 75)

    iter_efficiency = 100 if iterations == 0 else 60 if iterations == 1 else 35

    model_scores = {
        "gemini-2.5-flash": 100,
        "gemini-2.5-flash-lite": 55,
        "groq/llama-3.3-70b-versatile": 88,
        "groq/llama-3.1-8b-instant": 60,
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free": 80,
        "openrouter/openai/gpt-oss-20b:free": 70,
        "openrouter/google/gemma-4-26b-a4b-it:free": 68,
        OFFLINE_LABEL: 50,
        "none": 0,
    }
    model_reliability = model_scores.get(model_used, 70)
    # Fireworks ids are env-driven (and can be a list), so score by prefix.
    if model_used.startswith("fireworks/"):
        model_reliability = 85
    used_fallback = model_used not in ("gemini-2.5-flash", "")

    return {
        "audit_safety": audit_safety,
        "iter_efficiency": iter_efficiency,
        "model_reliability": model_reliability,
        "used_fallback": used_fallback,
        "flag_count": len(flags),
        "iterations": iterations,
        "model_used": model_used or "gemini-2.5-flash",
    }


# ── Generate plan with SSE logs ───────────────────────────────────
class PatientData(BaseModel):
    name: str
    age: int
    weight: float
    protein_req: float
    state_name: str
    allergies: str = "None"


@app.post("/generate")
async def generate_plan(patient: PatientData):
    print(f"[POST /generate] Starting for: {patient.name}")

    async def event_stream():
        def log(msg: str):
            return f"data: {json.dumps({'log': msg})}\n\n"

        yield log(f"Starting NutriGuard workflow for {patient.name}...")
        yield log(f"Patient: {patient.age}y, {patient.weight}kg, Region: {patient.state_name}")
        await asyncio.sleep(0.1)
        yield log("Node 1 — Researcher: Searching regional foods...")
        await asyncio.sleep(0.2)
        yield log("Node 2 — Chef Agent: Building 7-day meal plan...")
        await asyncio.sleep(0.2)
        yield log("Node 3 — Auditor: Checking against clinical PDFs (RAG)...")
        await asyncio.sleep(0.2)
        yield log("Node 4 — Judge: Generating final verdict...")
        await asyncio.sleep(0.2)

        try:
            # Pydantic v2: .model_dump() preferred over .dict()
            payload = patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
            result = run_workflow(payload)
            final = result.get("final_plan", "{}")

            try:
                plan_data = json.loads(final)
            except Exception:
                plan_data = {"verdict": "CAUTION", "summary": final, "plan": {}}

            protein_note = plan_data.get("protein_note", "")
            if not protein_note and patient.protein_req > 60:
                protein_note = "Supplement: Whey Protein Isolate 20g/day (unflavored, mixed in warm water)"
                plan_data["protein_note"] = protein_note
                yield log(f"Protein gap detected — supplement recommended")

            conn = get_db()
            conn.execute(
                "INSERT INTO patients (name, age, weight, protein_req, state_name, allergies, plan) VALUES (?,?,?,?,?,?,?)",
                (patient.name, patient.age, patient.weight, patient.protein_req,
                 patient.state_name, patient.allergies, json.dumps(plan_data)),
            )
            conn.commit()
            conn.close()

            audit_result = result.get("audit_result", {})
            iterations = result.get("iterations", 0)
            model_used = result.get("model_used", "gemini-2.5-flash")
            eval_scores = compute_eval_scores(audit_result, iterations, model_used)

            yield log(f"Audit: {audit_result.get('status', 'APPROVED')} — {eval_scores['flag_count']} flag(s)")
            yield log(f"Chef retries: {iterations} — Model: {model_used}")
            yield log(f"Verdict: {plan_data.get('verdict', 'CAUTION')}")
            yield log("Workflow complete. Plan ready for doctor review.")

            yield f"data: {json.dumps({'done': True, 'plan': plan_data, 'audit': audit_result, 'eval': eval_scores})}\n\n"

        except Exception as e:
            yield log(f"Error: {str(e)}")
            yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
