# 🛡️ Agentic AI Platform v3.1 — Hardened Assistant
Platform Version: **3.1 | PRD v1.3 | Design v4**

The **Agentic AI Platform v3.1** is a state-aware, autonomous insurance service assistant hardened for enterprise precision. This version focuses on **Retrieval Quality Assurance**, **Observability**, and **Zero-Touch Knowledge Sync**.

---

## 🚀 Recent Enhancements (v3.1)

### 🏥 Hardened RAG Pipeline
*   **Precision Thresholding**: Enforced a strict **0.75 confidence score** requirement for all FAQ retrievals.
*   **Fallback Logic**: Automatically triggers human escalation if semantic relevance falls below the threshold, ensuring no "hallucinated" process advice is given.
*   **Specific Tuning**: Optimized retrieval for complex annuity taxation and payment queries.

### 📊 Management Dashboard
A dedicated operational layer for system administrators:
*   **Service Analytics**: Real-time turn-by-turn audit of LLM intent mapping and confidence scoring.
*   **Knowledge Base Admin**: 
    *   **FAQ Match Tester**: Live diagnostic tool for verifying RAG recall.
    *   **Cloud-Sync**: Integrated self-healing utility to push local policy updates to Pinecone Cloud vectors.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **UI Framework** | [Streamlit](https://streamlit.io/) (Premium Dark Mode Interface) |
| **LLM Inference** | [Groq](https://groq.com/) (Default: Llama-3.3-70b-versatile) |
| **Intelligence** | Hybrid (Llama 3.3, Claude 3.5 Sonnet, GPT-4o-mini) |
| **Vector DB** | [Pinecone](https://www.pinecone.io/) (Cloud) / ChromaDB (Local) |
| **RAG Threshold**| **0.75** (Hardened) |
| **Search Engine** | Hybrid Search (Pinecone Vector + Metadata Filtering) |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| **Orchestration** | Python State Machine (Custom Orchestrator) |
| **Data Integrity** | JSON Schema Validation / Strict Field-Level Sanitization |

---

## 🛠️ Key Functionality

### 1. State-Aware Orchestration
The platform follows a rigorous state machine to ensure compliance and security in insurance interactions:
`GREET` → `COLLECT_POLICY` → `ID & VERIFY (ID&V)` → `SERVE_INTENT` → `CLOSE`.

### 2. Multi-Step Identity & Verification (ID&V)
*   **Policyholder Verification**: 5-point match (Name, Address, Postcode, DOB).
*   **Adviser Verification**: Strict validation of firm details and representative names.
*   **Fuzzy Matching**: Uses Levenshtein distance and Phonetic similarity for robust identification even with typos.

### 3. Hybrid RAG Architecture
Combining semantic meaning with keyword precision:
*   **Dense Retrieval**: ChromaDB captures conceptual intent.
*   **Sparse Retrieval**: Rank-BM25 captures specific terminology (e.g., policy clauses).
*   **RRF (Reciprocal Rank Fusion)**: Merges both streams for the most accurate grounding possible.

### 4. Safety & Compliance
*   **Duress Detection**: LLM-monitored signals for customer distress or financial hardship.
*   **Loop Detection**: Prevents AI circular reasoning by tracking turn-to-turn progress.
*   **Audit Logging**: Every LLM turn, tool call, and state transition is audited in real-time.
*   **Contact History**: Automatically writes summaries to the System of Record (SoR) upon resolution.

---

## 🤖 LLM Implementation

The platform utilizes LLMs in **JSON Mode** for structured orchestration. Every response is parsed into an `AgentTurn` schema:

```json
{
  "intent": "policy_valuation",
  "entities": { "policy_number": "L1234567" },
  "action_intent": "compare_verification",
  "rag_query": null,
  "duress_signal": false,
  "caller_response": "Found that for you. What else can I help with?"
}
```

This allows for deterministic control over sensitive insurance data while maintaining a natural, empathetic conversation window.

---

## 📦 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Ensure `GROQ_API_KEY` is set to your [Groq Console](https://console.groq.com/keys) key.

### 3. Execution
Launch the modern Streamlit interface:
```bash
streamlit run app_streamlit.py
```
---

## ☁️ Cloud Deployment (Streamlit)

### 1. Vector Database (Pinecone)
For persistent knowledge access in the cloud, use **Pinecone**:
1. Create a free index at [Pinecone.io](https://www.pinecone.io/).
2. Set dimensions to **384** (matches `all-MiniLM-L6-v2`).
3. Set Metric to **Cosine**.
4. Ingest your data:
   ```bash
   export PINECONE_API_KEY=your_key
   export PINECONE_INDEX_NAME=your_index
   python scripts/ingest_to_pinecone.py --dir docs/faq/
   ```

### 2. Streamlit Secrets
Add the following to your Streamlit Cloud "Secrets" dashboard:
```toml
GROQ_API_KEY = "your_groq_key"
VECTOR_DB_BACKEND = "pinecone"
PINECONE_API_KEY = "your_pinecone_key"
PINECONE_INDEX_NAME = "your_index_name"
```

---

## 📁 Project Architecture

*   `app_streamlit.py`: The primary web interface and session manager.
*   `app/agent.py`: The core state machine and message router.
*   `app/llm_client.py`: Multi-backend client (Groq/Anthropic/OpenAI) with lazy-config loading.
*   `app/rag_client.py`: Hybrid search engine for policy FAQs.
*   `chroma_data/`: Local persistent vector storage.
*   `tools/`: SoR adapters, fuzzy matchers, and audit loggers.

---

## 🛡️ Escalation Matrix
The agent escalates to a human specialist in the following scenarios:
*   Explicit user request ("speak to a person").
*   Detection of caller distress or duress.
*   Repeated data capture failures (Policy/ID&V retries).
*   Vulnerability or Fraud (FCU) flags on the policy record.
*   Loop detection (agent circling on the same issue).
