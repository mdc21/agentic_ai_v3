# Detailed Design: Agentic AI Platform v4 (Hardened)

## 1. System Architecture Overview
The Agentic AI Platform v3.1 introduces a dual-layer architecture:
1.  **Conversational Engine**: The core state machine and LLM orchestrator handling real-time policyholder interactions.
2.  **Management Dashboard**: An administrative layer built in Streamlit for system observability, RAG diagnostics, and Knowledge Base lifecycle management.

## 2. RAG Quality Assurance (Hardening)

### 2.1. Dynamic Thresholding
The `RAGClient` (`app/rag_client.py`) has been refactored to prioritize environment-driven configuration over hardcoded defaults.
*   **Threshold Value**: Enforced at **0.75** (`RAG_SCORE_THRESHOLD`).
*   **Dynamic Lookup**: The `query()` method now checks for environment overrides on every invocation, allowing for immediate configuration updates without system restarts.
*   **Fail-Safe**: If the top Match Score < 0.75, the RAG client returns `answerable=False`, triggering the `escalate` intent within the `AgentOrchestrator`.

```python
# app/rag_client.py
self.threshold = float(os.getenv("RAG_SCORE_THRESHOLD", 0.65))
```

### 2.2. FAQ Knowledge Base Optimization
Targeted tuning was performed to resolve semantic drift in annuity taxation queries:
*   **Intent**: `annuity_taxation` (FAQ ID: `faq_053`).
*   **Objective**: Ensure "tax on annuity payment" matches with high confidence (>0.85).
*   **Implementation**: Broadened question stems and prioritized `Context Tags` to overlap with common user search patterns.

## 3. Dashboard UI & UX Design

### 3.1. Stateful Navigation
The dashboard now implements a stateful radio-based navigation system (`st.radio`) instead of native Streamlit tabs to prevent the "scroll-down" regression.
*   **Persistence**: The `st.chat_input` is pinned to the bottom of the viewport and is conditionally rendered **only** when the `active_tab` is set to "Policy Assistant."
*   **UX Separation**: This ensures a clean, modular experience where administrative utilities (Analytics, KB Sync) do not interfere with the primary chat interface.

### 3.2. Knowledge Base Administration
A new operational tab provides two critical diagnostic utilities:
1.  **FAQ Match Tester**: Real-time vector similarity testing. Uses the internal `RAGClient` and `SentenceTransformer` to provide absolute semantic grounding.
2.  **Cloud-Sync (Self-Healing)**: An internal utility (`app/ingest_utils.py`) that performs Pinecone upserts. This solves "Operation not permitted" issues in locked-down environments by running ingestion within the application's own authorized process context.

## 4. Data & Observability
The system produces three primary audit streams:
1.  **bot_faq_interaction.log**: Detailed RAG performance tracking including query, retrieved IDs, scores, and answerable status.
2.  **bot_llm_interaction.log**: Raw LLM prompts and JSON responses for intent mapping audit.
3.  **ser_performance_analytics**: Real-time dashboard view of the latest 50 interactions for rapid forensic analysis.

---
*Version: 3.1 (Hardened)*
*Ref: Design v4*
