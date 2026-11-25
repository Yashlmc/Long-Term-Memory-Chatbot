# Long-Term Memory Chatbot

A modular, terminal-based conversational AI assistant that separates **short-term session context** from **persistent semantic long-term memory**. Powered by **OpenRouter**, local **Sentence Transformers** embeddings (`all-MiniLM-L6-v2`), and a local **Qdrant** vector database, the chatbot extracts, screens, stores, updates, and retrieves user-specific facts across multiple sessions with privacy guardrails and interactive conflict resolution.

## Key Capabilities

* **Cross-Session Semantic Memory (`memory/qdrant_store.py`, `memory/memory_retriever.py`):** Embeds approved user facts into 384-dimensional L2-normalized vectors stored in a local persistent Qdrant collection (`Distance.COSINE`), ranking retrieved memories by a blend of semantic similarity (`0.80`), importance (`0.15`), and confidence (`0.05`).

* **Deterministic Memory Extraction (`memory/memory_extractor.py`):** Uses a shared OpenRouter client at `temperature=0.0` (with automatic retry on malformed JSON and strict type/boolean validation) to extract stable personal facts, preferences, interests, skills, projects, goals, and habits while ignoring greetings and transient chatter.

* **Privacy-Aware Screening (`memory/privacy_policy.py`):** Enforces three runtime modes (`AUTO`, `ASK`, `DISABLED`) and automatically blocks credentials (`sk-`, `gsk_`, Bearer tokens, passwords), payment/ID numbers (formatted card numbers, Aadhaar, PAN), and both first-person and third-person sensitive health/financial statements before any vector is written to disk.

* **Duplicate Merging & Interactive Conflict Resolution (`memory/memory_manager.py`):** Filters candidate matches directly by `memory_type` in Qdrant, automatically updates near-duplicates (similarity `>= 0.92`), and prompts the user to `[R]eplace`, `[K]eep both`, or `[S]kip` potential conflicts (similarity `0.78–0.92`).

* **Persistent Bounded Sessions (`session/session_manager.py`):** Stores full conversation transcripts atomically as UTF-8 JSON files under `data/sessions/` while sending a bounded sliding window (`MAX_HISTORY_MESSAGES`, default `20`) to the LLM to prevent context-window overflow.

* **Safe Resource Lifecycle (`main.py`):** Releases the local Qdrant storage lock cleanly inside a `try/finally` block on `/exit` or `Ctrl+C`, and skips extractor LLM calls when `/memory_mode DISABLED` is active.

## Repository Structure

| File / Directory | Responsibility | 
| ----- | ----- | 
| `main.py` | Interactive CLI loop, command router, conflict-resolution prompt, and clean Qdrant shutdown. | 
| `config.py` | Environment-driven configuration for API key, OpenRouter model, embedding model, Qdrant path, session directory, user ID, and history window. | 
| `llm/openrouter_client.py` | OpenRouter chat wrapper supporting configurable temperature and empty-response validation. | 
| `embeddings/embedding_model.py` | Local Sentence Transformers loader producing L2-normalized dense vectors. | 
| `session/session_manager.py` | Multi-session persistence with atomic JSON writes, UTC timestamps, corrupt-file handling, and bounded `get_recent_history()`. | 
| `memory/qdrant_store.py` | Local Qdrant wrapper with `user_id` + `memory_type` filtering, CRUD operations, and explicit `close()` lock release. | 
| `memory/memory_extractor.py` | Structured JSON fact extractor with fenced-block parsing, schema validation, and retry logic. | 
| `memory/privacy_policy.py` | Policy gate (`AUTO` / `ASK` / `DISABLED`) screening secrets, IDs, and first-/third-person sensitive attributes. | 
| `memory/memory_manager.py` | Orchestrates privacy screening, deduplication, updates, and `resolve_review_item()` conflict handling. | 
| `memory/memory_retriever.py` | Retrieves top-k user memories above `min_similarity` and formats bounded system-prompt context. | 
| `test_*.py` | Component test scripts for embeddings, Qdrant, extractor, privacy policy, manager, and retriever. | 

## Technology Stack

| Component | Technology | Role in System | 
| ----- | ----- | ----- | 
| LLM Inference | OpenRouter (`openrouter/free` default) | Conversational response generation (`temperature=0.7`) and memory extraction (`temperature=0.0`). | 
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) | 384-dimensional L2-normalized local semantic embeddings. | 
| Vector Database | Qdrant (`qdrant-client`, local disk) | Persistent vector storage and filtered cosine similarity search under `./qdrant_data`. | 
| Session Storage | Python `json` (atomic writes) | Persistent multi-session transcripts under `data/sessions/`. | 
| Configuration | `python-dotenv` | Loads credentials and runtime parameters from `.env`. | 

## Quickstart

### 1. Prerequisites

* Python 3.10+

* An OpenRouter API key (`OPENROUTER_API_KEY`)

### 2. Installation

```
git clone <your-repository-url>
cd LongTermMemory_Chatbot
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and add your OpenRouter key:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/free

```

### 4. Run the Chatbot

```
python main.py

```

## CLI Commands

| Command | Description | 
| ----- | ----- | 
| `/new` | Create and switch to a new conversation session | 
| `/sessions` | List all saved sessions ordered by most recently updated | 
| `/use <session_id>` | Switch to an existing session | 
| `/current` | Show the active session ID | 
| `/history` | Print the full message history of the active session | 
| `/delete <session_id>` | Delete a saved session | 
| `/memories` | List all stored long-term memories for the active user | 
| `/search <query>` | Inspect live semantic memory retrieval and similarity scores | 
| `/forget <memory_id>` | Delete a specific long-term memory | 
| `/memory_mode <AUTO|ASK|DISABLED>` | Switch runtime memory approval policy | 
| `/clear_memory` | Confirm (`YES`) and erase all long-term memories for the user | 
| `/exit` | Close Qdrant cleanly and exit | 

## Running Component Tests

Run individual component scripts from the project root (ensure `main.py` is not running concurrently so the local Qdrant storage lock is free):

```
python test_embeddings.py
python test_privacy_policy.py
python test_qdrant.py
python test_memory_extractor.py
python test_memory_manager.py
python test_memory_retriever.py

```

## Privacy & Limitations

* **External LLM Calls:** Messages and retrieved memory context sent to OpenRouter are subject to the model provider's data policies.

* **Regex Screening Scope:** Automated secret and health/PII screening catches common patterns but cannot guarantee 100% coverage of unstructured sensitive disclosures; use `/memory_mode ASK` or `/memory_mode DISABLED` when handling confidential topics.

* **Single Local Instance:** Embedded local Qdrant (`./qdrant_data`) supports one active process at a time; for multi-user or concurrent web deployments, point `QdrantClient` to a hosted Qdrant server.