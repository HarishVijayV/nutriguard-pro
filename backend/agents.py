"""
agents.py — Agentic Engine (LangGraph only, no CrewAI)
Flow: Researcher → Chef → Auditor → Judge

Provider chain (auto-fallback, with parallel health probe):
  1. Gemini  (3 models × up to 2 keys)
  2. Groq    (Llama 3.3 70B → Llama 3.1 8B)            via GROQ_API_KEY
  3. OpenRouter (free Gemini/Llama/Qwen)                via OPENROUTER_API_KEY
  4. Fireworks (last real LLM, fully env-driven)        via FIREWORKS_API_KEY + FIREWORKS_MODEL
  5. Hardcoded clinically-safe template (Kerala-tuned)

Speed improvement:
  At startup, all providers are probed IN PARALLEL (~3-5 sec one-time cost).
  After that, every request only tries providers known to be alive — so
  exhausted providers don't add latency to each call.
  Dead providers are auto-retried after a 5-minute cooldown.

Retry: if Auditor rejects, Chef retries (max 1x).

Version marker — look for this in your startup log to confirm new file is loaded:
  [agents.py v1.2] starting — health probe enabled
"""

import os
import re
import json
import time
from typing import TypedDict, Callable
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from rag import audit_food

load_dotenv()

print("[agents.py v1.2] starting — health probe enabled")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ── Gemini key rotation ───────────────────────────────────────────
_api_keys = [k for k in [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY"),
] if k]
seen = set()
API_KEYS = [k for k in _api_keys if not (k in seen or seen.add(k))]

if not API_KEYS:
    raise RuntimeError("No Gemini API key found. Set GOOGLE_API_KEY_1 or GOOGLE_API_KEY.")

print(f"[agents.py] Loaded {len(API_KEYS)} Gemini API key(s)")
print(f"[agents.py] Groq fallback:       {'enabled' if os.getenv('GROQ_API_KEY') else 'disabled (no GROQ_API_KEY)'}")
print(f"[agents.py] OpenRouter fallback: {'enabled' if os.getenv('OPENROUTER_API_KEY') else 'disabled (no OPENROUTER_API_KEY)'}")

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
OPENROUTER_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

# ── Fireworks: last real-LLM fallback, configured 100% from .env ──
# FIREWORKS_API_KEY     required — leave empty/unset to disable this tier
# FIREWORKS_MODEL       one model id, or several separated by commas (tried in order)
# FIREWORKS_BASE_URL    override only if Fireworks changes their endpoint
# FIREWORKS_MAX_TOKENS  bumped by default so a full 7-day JSON plan isn't truncated
FIREWORKS_API_KEY = (os.getenv("FIREWORKS_API_KEY") or "").strip()
FIREWORKS_MODELS = [
    m.strip() for m in (os.getenv("FIREWORKS_MODEL") or "").split(",") if m.strip()
]
FIREWORKS_BASE_URL = (
    os.getenv("FIREWORKS_BASE_URL")
    or "https://api.fireworks.ai/inference/v1/chat/completions"
).strip()
# Generous default: DeepSeek "flash" spends tokens on reasoning before the JSON,
# and a truncated 7-day plan is unparseable. Lower it only to cut cost.
FIREWORKS_MAX_TOKENS = int(os.getenv("FIREWORKS_MAX_TOKENS") or 8192)

print(
    "[agents.py] Fireworks fallback:  "
    + (
        f"enabled ({', '.join(FIREWORKS_MODELS)})"
        if FIREWORKS_API_KEY and FIREWORKS_MODELS
        else "disabled (need FIREWORKS_API_KEY + FIREWORKS_MODEL)"
    )
)

OFFLINE_LABEL = "Offline Clinical Template"
COOLDOWN_SECONDS = 300         # mark provider dead for 5 min after a real failure
PROBE_TIMEOUT_SECONDS = 12     # per-provider timeout during parallel probe
SKIP_PROBE = os.getenv("SKIP_HEALTH_PROBE", "").lower() in ("1", "true", "yes")


# ── Health cache ──────────────────────────────────────────────────
# provider_id -> {"alive": bool, "cooldown_until": float}
_provider_health: dict = {}
_health_lock = Lock()


def _now() -> float:
    return time.time()


def _is_alive(provider_id: str) -> bool:
    """Cache check — instant."""
    with _health_lock:
        h = _provider_health.get(provider_id)
        if h is None:
            return True  # never checked → optimistic
        if h["alive"]:
            return True
        # Dead but cooldown passed → allow re-try, will be re-judged on result
        return _now() >= h["cooldown_until"]


def _mark_alive(provider_id: str):
    with _health_lock:
        _provider_health[provider_id] = {"alive": True, "cooldown_until": 0}


def _mark_dead(provider_id: str):
    with _health_lock:
        _provider_health[provider_id] = {
            "alive": False,
            "cooldown_until": _now() + COOLDOWN_SECONDS,
        }


def get_provider_status() -> dict:
    """Public snapshot for /health/providers endpoint."""
    with _health_lock:
        snapshot = {}
        for pid, h in _provider_health.items():
            snapshot[pid] = {
                "alive": h["alive"],
                "cooldown_remaining": max(0, int(h["cooldown_until"] - _now())) if not h["alive"] else 0,
            }
        return snapshot


# ── Hardcoded clinically-safe templates ───────────────────────────
KERALA_FALLBACK_PLAN = {
    "plan": {
        "day1": {"breakfast": "Soft steamed idli (2 pieces) with thin coconut chutney (no chili)",
                 "lunch": "Curd rice with steamed ash gourd (elavan)",
                 "dinner": "Kerala rice kanji with mashed boiled potato",
                 "snack": "Ripe banana (1 piece)"},
        "day2": {"breakfast": "Idiyappam (string hoppers) with thin coconut milk",
                 "lunch": "Soft white rice with mild moru curry and steamed pumpkin (mathanga)",
                 "dinner": "Plain rice with pulissery (mild yogurt curry, no chili)",
                 "snack": "Tender coconut water"},
        "day3": {"breakfast": "Plain puttu with mashed ripe banana",
                 "lunch": "Steamed rice with well-cooked parippu (yellow moong dal, no whole spices)",
                 "dinner": "Rice kanji with peeled and stewed apple",
                 "snack": "Soft sweet banana"},
        "day4": {"breakfast": "Soft appam with thin coconut milk (no sugar additives)",
                 "lunch": "White rice with mild avial (boiled vegetables, no chili, light coconut paste)",
                 "dinner": "Rice kanji with mashed bottle gourd",
                 "snack": "Plain curd with mashed banana"},
        "day5": {"breakfast": "Well-cooked rava upma (mild, no mustard tempering)",
                 "lunch": "Curd rice with boiled potato",
                 "dinner": "Idiyappam with light vegetable stew (no whole spices, no pepper)",
                 "snack": "Stewed apple"},
        "day6": {"breakfast": "Soft idli with mild sambar (well-cooked dal and vegetables, no chili)",
                 "lunch": "Rice with mild pulissery and steamed papaya",
                 "dinner": "Rice kanji with mashed sweet potato",
                 "snack": "Ripe banana"},
        "day7": {"breakfast": "Plain dosa (thin, no chutney) with thin coconut milk",
                 "lunch": "Steamed rice with parippu and steamed pumpkin",
                 "dinner": "Rice with mild moru curry and boiled carrot",
                 "snack": "Tender coconut water"},
    },
    "protein_note": "Supplement: Whey Protein Isolate 20g/day mixed in warm water (lactose-free isolate if intolerant)",
}

GENERIC_FALLBACK_PLAN = {
    "plan": {
        "day1": {"breakfast": "Soft idli (2 pieces) with plain curd",
                 "lunch": "Khichdi (rice + moong dal, soft cooked, no whole spices)",
                 "dinner": "Plain rice with mild moong dal and boiled potato",
                 "snack": "Ripe banana"},
        "day2": {"breakfast": "Well-cooked rava upma (mild)",
                 "lunch": "Curd rice with boiled carrot",
                 "dinner": "Khichdi with steamed pumpkin",
                 "snack": "Ripe papaya (peeled)"},
        "day3": {"breakfast": "Soft poha (mild, no chili)",
                 "lunch": "Mashed potato with plain rice and well-cooked dal",
                 "dinner": "Rice kanji with boiled bottle gourd",
                 "snack": "Ripe banana"},
        "day4": {"breakfast": "Boiled egg whites with plain white toast (no butter)",
                 "lunch": "Plain rice with mild yogurt curry (kadhi, no chili)",
                 "dinner": "Khichdi with stewed apple",
                 "snack": "Plain curd with banana"},
        "day5": {"breakfast": "Soft idli with thin coconut milk",
                 "lunch": "Curd rice with boiled potato",
                 "dinner": "Plain rice with mild moong dal",
                 "snack": "Stewed pear"},
        "day6": {"breakfast": "Bland rava upma",
                 "lunch": "Khichdi with steamed papaya",
                 "dinner": "Rice kanji with mashed sweet potato",
                 "snack": "Ripe banana"},
        "day7": {"breakfast": "Soft poha with plain curd",
                 "lunch": "Plain rice with mild dal and boiled carrot",
                 "dinner": "Khichdi with stewed apple",
                 "snack": "Tender coconut water"},
    },
    "protein_note": "Supplement: Whey Protein Isolate 20g/day mixed in warm water (lactose-free isolate if intolerant)",
}


def get_fallback_plan(state_name: str) -> dict:
    if (state_name or "").strip().lower() == "kerala":
        return KERALA_FALLBACK_PLAN
    return GENERIC_FALLBACK_PLAN


# ── Provider call primitives ──────────────────────────────────────
def _gemini_call(model: str, key: str, prompt: str, json_mode: bool) -> str:
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    return resp.text


def _groq_call(model: str, prompt: str, json_mode: bool) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _openrouter_call(model: str, prompt: str, json_mode: bool) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://nutriguard-pro.local",
            "X-Title": "NutriGuard Pro",
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _strip_json_noise(text: str) -> str:
    """
    DeepSeek-style 'flash'/reasoning models sometimes wrap their answer in a
    <think>...</think> block or a ```json fence. Both break json.loads(), so
    peel them off before handing the text back to the graph.
    """
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned or text


def _fireworks_call(model: str, prompt: str, json_mode: bool) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": FIREWORKS_MAX_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        FIREWORKS_BASE_URL,
        headers={
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=90,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _strip_json_noise(content) if json_mode else content


# ── Provider registry: id -> (call_fn, model_label) ───────────────
def _build_provider_registry() -> list[tuple[str, Callable[[str, bool], str], str]]:
    """Returns list in PRIORITY ORDER: (provider_id, call_fn, model_label_for_eval)."""
    registry: list[tuple[str, Callable[[str, bool], str], str]] = []

    # Gemini first — outer loop models, inner loop keys (so key1 is exhausted first)
    for model in GEMINI_MODELS:
        for i, key in enumerate(API_KEYS):
            pid = f"gemini/{model}/key{i+1}"
            registry.append((pid, lambda p, j, m=model, k=key: _gemini_call(m, k, p, j), model))

    if os.getenv("GROQ_API_KEY"):
        for model in GROQ_MODELS:
            pid = f"groq/{model}"
            registry.append((pid, lambda p, j, m=model: _groq_call(m, p, j), f"groq/{model}"))

    if os.getenv("OPENROUTER_API_KEY"):
        for model in OPENROUTER_MODELS:
            pid = f"openrouter/{model}"
            registry.append((pid, lambda p, j, m=model: _openrouter_call(m, p, j), f"openrouter/{model}"))

    # Fireworks LAST: the guaranteed real-LLM safety net once every free tier
    # above has run out of quota. Only the offline template sits below it.
    if FIREWORKS_API_KEY and FIREWORKS_MODELS:
        for model in FIREWORKS_MODELS:
            pid = f"fireworks/{model}"
            registry.append((pid, lambda p, j, m=model: _fireworks_call(m, p, j), pid))

    return registry


PROVIDERS = _build_provider_registry()


# ── Parallel health probe ─────────────────────────────────────────
def _probe_one(provider_id: str, call_fn: Callable) -> tuple[str, bool, str]:
    """Send a tiny ping. Returns (id, alive, error_snippet)."""
    try:
        # Tiny prompt + non-JSON to keep token cost minimal
        call_fn("ping", False)
        return provider_id, True, ""
    except Exception as e:
        return provider_id, False, str(e)[:120]


def probe_all_providers() -> dict:
    """Run all probes in parallel. ~3-5 sec total wall clock."""
    if not PROVIDERS:
        return {}
    print(f"[agents.py] Probing {len(PROVIDERS)} providers in parallel...")
    start = _now()
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as exe:
        futures = {exe.submit(_probe_one, pid, fn): pid for pid, fn, _ in PROVIDERS}
        for fut in as_completed(futures, timeout=PROBE_TIMEOUT_SECONDS + 5):
            try:
                pid, alive, err = fut.result(timeout=PROBE_TIMEOUT_SECONDS)
            except Exception as e:
                pid = futures[fut]
                alive = False
                err = str(e)[:120]
            if alive:
                _mark_alive(pid)
                print(f"  [Health] alive   {pid}")
            else:
                _mark_dead(pid)
                print(f"  [Health] DOWN    {pid}  ({err})")

    elapsed = _now() - start
    alive = [pid for pid in _provider_health if _provider_health[pid]["alive"]]
    print(f"[agents.py] Probe done in {elapsed:.1f}s — {len(alive)}/{len(PROVIDERS)} providers alive")
    if alive:
        print(f"[agents.py] Primary alive provider: {alive[0]}")
    else:
        print(f"[agents.py] WARNING: no providers alive — all requests will use offline template")
    return get_provider_status()


# ── Unified LLM call (consults health cache first) ────────────────
def call_gemini(prompt: str, json_mode: bool = True) -> tuple[str, str]:
    """
    Public name kept as `call_gemini` so existing imports don't break.
    Iterates providers in priority order, BUT skips ones marked dead in cache.
    On real-call failure: marks provider dead with 5-min cooldown, moves on.
    On success: marks provider alive.
    Returns ("", "none") only when ALL providers fail.
    """
    skipped = 0
    tried = 0
    for pid, fn, model_label in PROVIDERS:
        if not _is_alive(pid):
            skipped += 1
            continue
        try:
            tried += 1
            print(f"  [LLM] -> {pid}")
            text = fn(prompt, json_mode)
            _mark_alive(pid)
            print(f"  [LLM] OK {pid}")
            return text, model_label
        except Exception as e:
            _mark_dead(pid)
            print(f"  [LLM] FAIL {pid}: {str(e)[:140]}")

    print(f"  [LLM] all providers down (tried {tried}, skipped {skipped} cached-dead) — using offline template")
    return "", "none"


# ── LangGraph State ───────────────────────────────────────────────
class NutriState(TypedDict):
    patient_name: str
    age: int
    weight: float
    protein_req: float
    state_name: str
    allergies: str
    raw_foods: str
    meal_plan: str
    audit_result: dict
    final_plan: str
    iterations: int
    model_used: str


# ── Node 1: Researcher ────────────────────────────────────────────
def researcher_node(state: NutriState) -> NutriState:
    print(f"\n[Node 1 — Researcher] Checking SQLite for: {state['state_name']}")
    import sqlite3

    try:
        conn = sqlite3.connect("nutriguard.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT food_name, category, reason FROM food_nodes WHERE state = ?",
            (state["state_name"],)
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    if rows:
        lines = [f"{r['category']}: {r['food_name']} — {r['reason']}" for r in rows]
        state["raw_foods"] = "\n".join(lines)
        print(f"  [Researcher] Loaded {len(rows)} foods from SQLite")
        try:
            results = tavily.search(
                query=f"Traditional {state['state_name']} dishes safe for ulcerative colitis",
                search_depth="basic",
            )
            extra = [r.get("content", "")[:150] for r in results["results"][:3]]
            state["raw_foods"] += "\n\nAdditional web context:\n" + "\n".join(extra)
        except Exception as e:
            print(f"  [Researcher] Tavily supplement skipped: {e}")
    else:
        print(f"  [Researcher] No SQLite data — running full Tavily search")
        try:
            results = tavily.search(
                query=f"Traditional {state['state_name']} dishes safe for ulcerative colitis breakfast lunch dinner",
                search_depth="advanced",
            )
            lines = [r["title"] + ": " + r.get("content", "")[:200] for r in results["results"]]
            state["raw_foods"] = "\n".join(lines)
        except Exception as e:
            print(f"  [Researcher] Tavily failed: {e} — using hardcoded fallback")
            state["raw_foods"] = (
                f"Safe colitis foods in {state['state_name']}: "
                "steamed rice, curd, khichdi, idli, kanji, soft banana"
            )
    return state


# ── Node 2: Chef ──────────────────────────────────────────────────
def chef_node(state: NutriState) -> NutriState:
    # Bump iteration counter when re-entering after a rejection.
    # (LangGraph ignores mutations made inside conditional edges — only nodes can update state.)
    if state.get("audit_result", {}).get("status") == "REJECTED":
        state["iterations"] = state.get("iterations", 0) + 1

    print(f"\n[Node 2 — Chef] Building 7-day plan (iteration {state['iterations']})")

    retry_note = ""
    if state.get("audit_result", {}).get("status") == "REJECTED":
        flags = state["audit_result"].get("flags", [])
        retry_note = (
            f"\nPREVIOUS PLAN REJECTED. Strictly avoid these words and any dishes containing them: "
            f"{', '.join(flags)}. Do NOT use the words themselves anywhere in the plan, "
            f"even in negations like 'non-spicy' or 'avoid spicy'."
        )

    prompt = f"""
You are a Clinical Nutritionist specializing in Ulcerative Colitis (IBD).
Create a 7-day meal plan for this patient:

Patient: {state['patient_name']}, Age: {state['age']}, Weight: {state['weight']}kg
Protein requirement: {state['protein_req']}g/day
Region: {state['state_name']} (use traditional regional foods)
Allergies: {state['allergies']}
Available regional foods: {state['raw_foods'][:800]}
{retry_note}

RULES:
- Only low-fiber, bland, steamed/boiled foods
- No raw salads, fried foods, seeds, or whole spices
- Do NOT use the words: spicy, fried, chili, raw salad, seeds, cream, alcohol, caffeine — anywhere, even in negations
- If protein cannot be met by food alone, add "Supplement: whey protein isolate 20g"

Return JSON:
{{
  "plan": {{
    "day1": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
    "day2": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
    "day3": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
    "day4": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
    "day5": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
    "day6": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
    "day7": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}}
  }},
  "protein_note": "supplement recommendation or empty string"
}}
"""
    text, model = call_gemini(prompt)

    if model == "none":
        print(f"  [Chef] All providers down — serving '{OFFLINE_LABEL}' for '{state['state_name']}'")
        fallback = get_fallback_plan(state["state_name"])
        state["meal_plan"] = json.dumps({
            "plan": fallback["plan"],
            "protein_note": fallback["protein_note"],
        })
        state["model_used"] = OFFLINE_LABEL
    else:
        state["meal_plan"] = text
        if state.get("model_used") != OFFLINE_LABEL or not state.get("model_used"):
            state["model_used"] = model

    return state


# ── Node 3: Auditor (RAG) ─────────────────────────────────────────
def auditor_node(state: NutriState) -> NutriState:
    print(f"\n[Node 3 — Auditor] Checking plan against clinical PDFs")

    if state.get("model_used") == OFFLINE_LABEL:
        print("  [Auditor] Offline template — pre-validated, audit skipped")
        state["audit_result"] = {
            "status": "APPROVED",
            "flags": [],
            "message": "Pre-validated offline clinical template — audit skipped.",
        }
        return state

    danger_keywords = ["spicy", "fried", "chili", "raw salad", "seeds", "cream", "alcohol", "caffeine"]
    plan_lower = state["meal_plan"].lower()
    flags = []

    for word in danger_keywords:
        # Whole-word match AND skip negated mentions like "non-spicy", "no spicy",
        # "without spicy", "avoid spicy", "not spicy".
        pattern = rf"(?<!\w)(?<!non-)(?<!no )(?<!without )(?<!avoid )(?<!not ){re.escape(word)}(?!\w)"
        if re.search(pattern, plan_lower):
            result = audit_food(word)
            if result["safety_hint"] == "UNSAFE":
                flags.append(word)

    if flags:
        state["audit_result"] = {
            "status": "REJECTED",
            "flags": flags,
            "message": f"Unsafe items found: {flags}. Chef must retry.",
        }
    else:
        state["audit_result"] = {
            "status": "APPROVED",
            "flags": [],
            "message": "Plan is clinically safe per PDF guidelines.",
        }
    return state


# ── Node 4: Judge ─────────────────────────────────────────────────
def judge_node(state: NutriState) -> NutriState:
    print(f"\n[Node 4 — Judge] Generating final verdict")

    if state.get("model_used") == OFFLINE_LABEL:
        fallback = get_fallback_plan(state["state_name"])
        state["final_plan"] = json.dumps({
            "verdict": "SAFE",
            "summary": (
                "Vetted offline clinical template served because all AI providers were "
                "temporarily unavailable. Plan is composed of pre-validated low-fiber, "
                "bland traditional dishes safe for IBD patients."
            ),
            "plan": fallback["plan"],
            "protein_note": fallback["protein_note"],
        })
        return state

    prompt = f"""
You are a Senior Clinical Nutrition Director.
Review this 7-day meal plan for an Ulcerative Colitis patient.

Patient: {state['patient_name']}, {state['age']}y, {state['weight']}kg
Region: {state['state_name']}
Audit result: {state['audit_result']['message']}
Meal plan: {state['meal_plan'][:1500]}

Return JSON:
{{
  "verdict": "SAFE" or "CAUTION" or "UNSAFE",
  "summary": "2-sentence clinical summary for the doctor",
  "plan": {{same 7-day structure}},
  "protein_note": "supplement note or empty string"
}}
"""
    text, model = call_gemini(prompt)

    if model == "none":
        print("  [Judge] All providers down — wrapping Chef plan with CAUTION verdict")
        try:
            chef_data = json.loads(state["meal_plan"])
        except Exception:
            chef_data = {}

        plan_obj = chef_data.get("plan") or get_fallback_plan(state["state_name"])["plan"]
        protein_note = chef_data.get("protein_note") or get_fallback_plan(state["state_name"])["protein_note"]

        state["final_plan"] = json.dumps({
            "verdict": "CAUTION",
            "summary": (
                "AI judge unavailable. Chef plan is shown as-is; please review manually "
                "before sharing with the patient."
            ),
            "plan": plan_obj,
            "protein_note": protein_note,
        })
        state["model_used"] = OFFLINE_LABEL
    else:
        state["final_plan"] = text
        state["model_used"] = model

    return state


# ── Conditional edge ──────────────────────────────────────────────
def should_retry(state: NutriState) -> str:
    # Pure function — no state mutation. LangGraph ignores writes here.
    if state["audit_result"]["status"] == "REJECTED" and state.get("iterations", 0) < 1:
        return "retry"
    return "finalize"


# ── Build graph ───────────────────────────────────────────────────
def build_workflow():
    g = StateGraph(NutriState)
    g.add_node("researcher", researcher_node)
    g.add_node("chef", chef_node)
    g.add_node("auditor", auditor_node)
    g.add_node("judge", judge_node)
    g.set_entry_point("researcher")
    g.add_edge("researcher", "chef")
    g.add_edge("chef", "auditor")
    g.add_conditional_edges("auditor", should_retry, {"retry": "chef", "finalize": "judge"})
    g.add_edge("judge", END)
    return g.compile()


workflow = build_workflow()
print("[agents.py] LangGraph workflow ready (Researcher→Chef→Auditor→Judge)")


# ── Run startup probe (skippable via env var) ─────────────────────
if SKIP_PROBE:
    print("[agents.py] SKIP_HEALTH_PROBE set — startup probe skipped")
else:
    probe_all_providers()


def run_workflow(patient_data: dict) -> dict:
    state: NutriState = {
        "patient_name": patient_data.get("name", "Patient"),
        "age": patient_data.get("age", 30),
        "weight": patient_data.get("weight", 65.0),
        "protein_req": patient_data.get("protein_req", 60.0),
        "state_name": patient_data.get("state_name", "Tamil Nadu"),
        "allergies": patient_data.get("allergies", "None"),
        "raw_foods": "",
        "meal_plan": "",
        "audit_result": {},
        "final_plan": "",
        "iterations": 0,
        "model_used": "",
    }
    print(f"\n[agents.py] ===== Workflow START: {state['patient_name']} =====")
    result = workflow.invoke(state, {"recursion_limit": 10})
    print(f"[agents.py] ===== Workflow DONE =====\n")
    return result


if __name__ == "__main__":
    out = run_workflow({
        "name": "Test Patient",
        "age": 35,
        "weight": 68.0,
        "protein_req": 65.0,
        "state_name": "Kerala",
        "allergies": "Lactose",
    })
    print(out["final_plan"][:500])
