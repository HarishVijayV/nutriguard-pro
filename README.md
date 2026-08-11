
# NutriGuard Pro: Clinical AI Dietitian for IBD

NutriGuard Pro is an advanced, agentic AI platform built to assist healthcare providers in creating safe, region-specific nutrition plans for **Ulcerative Colitis** and **IBD** patients. 

By combining **Retrieval-Augmented Generation (RAG)** with a multi-agent orchestration layer, the system ensures that every food recommendation is grounded in clinical evidence and regional availability.

---

##  Key Features

*  Agentic Workflow (LangGraph): Orchestrates four specialized agents:
   **Researcher:** Scrapes real-time regional food data using Tavily.
   **Chef:** Crafts 7-day meal plans based on patient vitals and allergies.
   **Auditor:** A RAG-powered agent that cross-references plans against 4+ clinical PDFs.
   **Judge:** Provides a final clinical verdict and protein-gap analysis.
* Specialized RAG: Uses a **fine-tuned MiniLM-L6-v2 embedding model** and ChromaDB to verify food safety against medical literature.
* Resilient Architecture: Implements a **Fallback & Degradation strategy** with a parallel startup health probe and a 5-minute circuit breaker per provider: 3 Gemini models × 2 keys → Groq → OpenRouter → Fireworks → a pre-vetted offline clinical template. Uptime never depends on a single vendor.
* Knowledge Graph: Dynamically generates an interactive SQLite-backed food graph for different Indian states (Tamil Nadu, Kerala, Punjab, etc.).

---

## 🛠️Tech Stack

- **Backend:** FastAPI, Python 3.11
- **AI Orchestration:** LangGraph, Gemini 2.0/1.5 API
- **Database:** SQLite (Knowledge Graph), ChromaDB (Vector Store)
- **Embeddings:** Hugging Face (Fine-tuned sentence-transformers)
- **Search:** Tavily API
- **Deployment:** Docker, [Your Hosting Provider Here]

---

##  Quick Start — Docker (one command)

Requires only Docker Desktop. Keys live in `backend/.env` (copy from `backend/.env.example`).

```bash
docker compose up --build
```

| Service  | URL                            |
|----------|--------------------------------|
| Frontend | http://localhost:3000          |
| Backend  | http://localhost:8000          |
| API docs | http://localhost:8000/docs     |
| Provider health | http://localhost:8000/health/providers |

Notes:
- **The first build takes a while** (~10-15 min): it installs PyTorch + sentence-transformers and runs a production `next build`. Subsequent `docker compose up` starts in seconds.
- On startup the backend loads the fine-tuned embedding model, opens ChromaDB, and probes every LLM provider in parallel — it reports `(healthy)` in about 40s. Watch it with `docker compose logs -f backend`.
- `nutriguard.db` and `chroma_db/` are bind-mounted, so researched states and your PDF index survive `docker compose down`.
- Stop with `docker compose down`. Rebuild after changing Python/JS code with `docker compose up --build`.
- Demoing on a LAN? The API URL is baked into the browser bundle at build time, so rebuild with it set:
  ```bash
  NEXT_PUBLIC_API_URL=http://<your-lan-ip>:8000 docker compose up --build
  ```

---

##  Quick Start — local (no Docker)

### 1. Prerequisites
- Python 3.11+, Node 20+
- Conda or Virtualenv
- API Keys: Google Gemini (required), Tavily; optional Groq / OpenRouter / Fireworks

### 2. Installation
```bash
git clone https://github.com/YOUR_USERNAME/nutriguard-pro.git
cd nutriguard-pro

# backend  → http://localhost:8000
cd backend
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
uvicorn main:app --reload --port 8000

# frontend → http://localhost:3000 (in a second terminal)
cd frontend
npm install
npm run dev
```
