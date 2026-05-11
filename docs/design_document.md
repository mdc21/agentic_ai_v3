# Architecture & Design Document: Insurance Agentic AI

## 1. Executive Summary
This platform is a multimodal (Voice + Chat) Agentic AI designed to automate insurance policy servicing. It combines real-time data retrieval from a System of Record (SoR) with grounded knowledge from a RAG (Retrieval-Augmented Generation) engine, all while maintaining a secure, state-aware conversation.

---

## 2. Technology Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **User Interface** | Streamlit (Web-based GUI) |
| **LLM Orchestration** | OpenAI (GPT-4o-mini) & Groq (Llama-3-70B) |
| **ASR (Speech-to-Text)** | Groq (Whisper-large-v3) - High Speed |
| **TTS (Text-to-Speech)** | gTTS (Google TTS) / OpenAI TTS-1 |
| **Vector Database** | Pinecone / ChromaDB / pgvector |
| **String Similarity** | RapidFuzz (Jaro-Winkler / Levenshtein) |

---

## 3. Core Design Patterns

### A. Mediator (Orchestrator) Pattern
The `AgentOrchestrator` acts as a central hub. Instead of components talking to each other, they all communicate through the Orchestrator. This ensures that state management, auditing, and error handling are centralized.

### B. Finite State Machine (FSM)
The conversation is managed via an explicit state machine (`AgentState`):
- `GREET` → `COLLECT_POLICY` → `VERIFY_IDENTITY` → `SERVE_INTENT` → `RESOLVE`.
This prevents the LLM from "wandering" and ensures the insurance compliance flow is followed strictly.

### C. Adapter Pattern
The `ASRClient` and `TTSClient` use the Adapter pattern to wrap different providers (Groq, Google, OpenAI, Mock). This allows the system to swap voice providers via a single `.env` change without modifying the core logic.

### D. Strategy Pattern (Verification)
Different identity fields use different "strategies" for matching:
- **Names**: Nickname Map + Phonetic (Soundex) + Fuzzy Ratio.
- **Postcodes**: ASR Confusion Map + Exact match.
- **DOB**: Normalised ISO matching.

---

## 4. Advanced Techniques

### A. Parallel Async Lookups
To minimize caller wait time, the system uses `asyncio` to fire off **SoR Data** and **RAG Knowledge** searches simultaneously.
- **Latency Gain**: ~1.2 seconds saved per turn.

### B. Hybrid RAG (Semantic + Lexical)
Our RAG implementation uses **Reciprocal Rank Fusion (RRF)**:
1. **Semantic Search**: Understands the *meaning* (e.g., "cash in" means "surrender").
2. **Keyword Search (BM25)**: Ensures exact matches for terms like "IHT" or "MVA".

### C. Layered Fuzzy Matching
A "Defense-in-Depth" approach to identity verification:
1. **Normalisation**: Strips whitespace and performs **Abbreviation Expansion** for addresses (e.g., "St" → "Street", "Rd" → "Road").
2. **Token Sort Ratio**: Specifically for addresses, we use `token_sort_ratio` which ignores word order (e.g., "14 High Street" matches "High Street 14").
3. **Nickname Mapping**: Links common name variants like "John" and "Jonathan" using a business-logic dictionary.
4. **Phonetic Check**: Uses the **Soundex** algorithm to handle speech-to-text misspellings (e.g., "Smyth" vs "Smith").
5. **Levenshtein Distance**: Performs a final character-similarity check using `fuzz.WRatio`.

---

## 5. Integration & Fallbacks

### LLM Model Integration
- **Orchestration**: GPT-4o-mini handles intent classification and entity extraction.
- **Synthesis**: Llama-3 (via Groq) is used for high-speed, grounded response generation.

### ASR & TTS Models
- **ASR**: **Whisper-large-v3 (via Groq)**. Chosen for its sub-500ms latency.
- **TTS**: **gTTS** (Default/Free) with **OpenAI TTS-1** as a high-quality fallback.

### Resiliency & Fallback Hierarchy
1. **Standard Mode**: Real API calls to Groq/OpenAI.
2. **Mock Mode**: If an API key fails or `USE_MOCK` is set to `true`, the system falls back to local JSON stubs (`_MOCK_FAQ`, `_MOCK_POLICY`).
3. **Silent Capture Handling**: If ASR returns empty text, the orchestrator detects the silence and proactively asks the user to repeat.

---

## 6. Audit & Observability
Every interaction is logged in three layers:
1. **User/Bot Interaction**: Human-readable transcript.
2. **LLM Interaction**: Raw JSON inputs/outputs for debugging hallucinations.
3. **FAQ/RAG Logs**: Captures confidence scores and retrieved document IDs for tuning the threshold.
