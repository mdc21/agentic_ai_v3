import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables BEFORE importing app components
load_dotenv(override=True)

from app.agent import AgentOrchestrator, AgentState
from app.rag_client import RAGClient

# Page configuration
st.set_page_config(
    page_title="Agentic AI Platform v3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium feel
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .sidebar .sidebar-content {
        background-color: #0f172a;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator(channel="chat")

if "ctx" not in st.session_state:
    st.session_state.ctx = st.session_state.orchestrator.new_session()

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Trigger the initial agent greeting
    initial_response = st.session_state.orchestrator.process_turn(
        st.session_state.ctx, 
        text_input="hello"
    )
    st.session_state.messages.append({"role": "assistant", "content": initial_response})

def reset_session():
    st.session_state.ctx = st.session_state.orchestrator.new_session()
    st.session_state.messages = []
    st.rerun()

# Sidebar for State Monitoring
with st.sidebar:
    st.title("🤖 Status Monitor")
    st.write(f"**Session ID:** `{st.session_state.ctx.session_id}`")
    
    # State Display with color
    state_color = "🟢" if st.session_state.ctx.state == AgentState.RESOLVED else \
                  "🔴" if st.session_state.ctx.state == AgentState.ESCALATED else "🟡"
    st.subheader(f"Current State: {state_color}")
    st.info(f"**{st.session_state.ctx.state.value.replace('_', ' ').upper()}**")
    
    # RAG & LLM Status
    rag_backend = os.getenv("VECTOR_DB_BACKEND", "mock").upper()
    kb_str = f"**{rag_backend} CLOUD**" if rag_backend == "PINECONE" else f"**{rag_backend} LOCAL**"
    st.caption(f"Knowledge Base: {kb_str}")
    
    if st.session_state.ctx.active_model:
        st.caption(f"Active Model: **{st.session_state.ctx.active_model}**")
    
    # Detected Intent
    if st.session_state.ctx.call_intent:
        st.success(f"🎯 **Intent:** {st.session_state.ctx.call_intent.replace('_', ' ').title()}")

    # Policy Info
    if st.session_state.ctx.policy_number:
        with st.expander("📄 Policy Details", expanded=True):
            st.write(f"**Number:** {st.session_state.ctx.policy_number}")
            if st.session_state.ctx.product_type:
                st.write(f"**Product:** {st.session_state.ctx.product_type.title()}")
            if st.session_state.ctx.heritage_brand:
                st.write(f"**Brand:** {st.session_state.ctx.heritage_brand.title()}")

    # Collected Entities
    with st.expander("👤 Collected Entities", expanded=False):
        entities = st.session_state.ctx.caller_entities
        any_ent = False
        for field in entities.__dataclass_fields__:
            val = getattr(entities, field)
            if val:
                st.write(f"**{field.replace('_', ' ').title()}:** {val}")
                any_ent = True
        if not any_ent:
            st.write("No entities collected yet.")

    # Reset Button
    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        reset_session()

# Main Navigation Selection (Stateful tabs)
tabs = ["🛡️ Policy Assistant", "📊 Real-time Analytics", "📚 Knowledge Base"]
active_tab = st.radio("Navigation", tabs, horizontal=True, label_visibility="collapsed")

if active_tab == "🛡️ Policy Assistant":
    st.title("🛡️ Insurance Policy Assistant")
    st.info("💡 **Note**: This environment uses **synthetic dummy data** for testing conversational flows.")
    st.markdown("---")

    # 1. Render Historical Conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("is_terminal"):
                if st.session_state.ctx.state == AgentState.RESOLVED:
                    st.success("Conversation successfully resolved!")
                elif st.session_state.ctx.state == AgentState.ESCALATED:
                    st.warning(f"Handing over to specialist. Reason: {message.get('esc_reason')}")

    # 2. Placeholder for the LIVE turn (Current interaction)
    live_placeholder = st.container()

elif active_tab == "📊 Real-time Analytics":
    st.title("📊 Service Performance Analytics")
    st.markdown("---")
    
    analytics_file = "analytics.csv"
    if os.path.exists(analytics_file):
        import pandas as pd
        df = pd.read_csv(analytics_file)
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        total_sessions = len(df)
        avg_duration = df['duration_sec'].mean()
        total_cost = df['total_cost_usd'].sum()
        escalation_rate = (df['escalated'].astype(str).str.lower() == 'true').mean() * 100
        
        m1.metric("Total Sessions", f"{total_sessions:,}")
        m2.metric("Avg Duration", f"{avg_duration:.1f}s")
        m3.metric("Total LLM Cost", f"${total_cost:.4f}")
        m4.metric("Escalation Rate", f"{escalation_rate:.1f}%")
        
        # Data table
        st.subheader("Historical Records (Last 100)")
        st.dataframe(df.tail(100).sort_values(by="end_time", ascending=False), 
                     use_container_width=True,
                     column_config={
                         "total_cost_usd": st.column_config.NumberColumn("Cost ($)", format="$%.4f"),
                         "duration_sec": st.column_config.NumberColumn("Duration", format="%.1fs")
                     })
        
        if st.button("Refresh Dashboard"):
            st.rerun()
    else:
        st.info("No analytics data available yet. Complete a conversation to see records here.")

elif active_tab == "📚 Knowledge Base":
    from app.ingest_utils import sync_pension_faqs_to_pinecone
    
    st.title("📚 Knowledge Base (FAQ Reference)")
    st.markdown("---")
    
    st.caption("ℹ️ **Disclaimer**: The information provided here is based on **publicly available data** regarding insurance processes, HMRC regulations, and general policy benefits.")

    # ── ⚡ Admin Sync Utility ─────────────────────────────────────────────────
    with st.expander("⚡ Knowledge Base Administration", expanded=False):
        st.write("Push the latest local FAQ updates to the Pinecone Cloud index.")
        if st.button("Sync Knowledge Base to Cloud", use_container_width=True):
            with st.spinner("Synchronizing... this may take a minute."):
                try:
                    msg = sync_pension_faqs_to_pinecone("docs/faq/pension_faqs.txt")
                    st.success(msg)
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync failed: {e}")
    
    # ── 🔍 FAQ Match Tester (New Utility) ─────────────────────────────────────
    st.subheader("🔍 Search & Match Testing")
    st.info("Test how the RAG engine scores queries. Confidence threshold is currently **0.75** (75%).")
    
    col_q, col_p = st.columns([3, 1])
    test_query = col_q.text_input("Enter test query (e.g., 'annuity tax')", key="kb_search_input")
    test_prod = col_p.selectbox("Product Type", ["annuity", "pension", "life", "all"], index=0)
    
    if st.button("Search Knowledge Base", use_container_width=True):
        if test_query:
            rag = RAGClient(st.session_state.ctx.cache)
            threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.75"))
            
            with st.spinner("Retrieving matches..."):
                results = rag.query(
                    question=test_query,
                    product_type=None if test_prod == "all" else test_prod,
                    session_id="kb_test_query",
                    audit_logger=st.session_state.orchestrator._audit
                )
                
                if results.chunks:
                    st.write(f"### Top Matches for: *\"{test_query}\"*")
                    for i, chunk in enumerate(results.chunks, 1):
                        meets_threshold = chunk.score >= threshold
                        status_color = "green" if meets_threshold else "red"
                        status_icon = "✅" if meets_threshold else "❌"
                        
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: rgba(30, 41, 59, 0.4);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: bold; font-size: 1.1em;">Match #{i}: {chunk.section}</span>
                                    <span style="color: {status_color}; font-weight: bold; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 12px;">
                                        {status_icon} Score: {chunk.score:.4f}
                                    </span>
                                </div>
                                <p style="margin-top: 10px; font-style: italic;">"{chunk.text[:250]}..."</p>
                                <div style="margin-top: 5px; font-size: 0.85em; opacity: 0.8;">
                                    <b>Source:</b> {chunk.source_doc} | <b>Product:</b> {chunk.product_type}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No matches found in the knowledge base.")
        else:
            st.warning("Please enter a query to test.")

    st.markdown("---")
    
    # ── 📂 Browse All FAQs ────────────────────────────────────────────────────
    st.subheader("📂 Browse All Knowledge Base Entries")
    import json
    faq_path = "docs/faq/pension_faqs.txt"
    if os.path.exists(faq_path):
        try:
            with open(faq_path, 'r') as f:
                faq_data = json.load(f)
            
            faqs = faq_data.get("faqs", [])
            metadata = faq_data.get("metadata", {})
            
            # Metadata summary
            st.caption(f"**Version:** {metadata.get('version')} | **Total Intents:** {metadata.get('usage_note')}")
            
            # Simple Category Filter
            all_categories = sorted(list(set(f["category"] for f in faqs)))
            selected_cat = st.selectbox("Filter by Category", ["All"] + all_categories)
            
            display_faqs = faqs if selected_cat == "All" else [f for f in faqs if f["category"] == selected_cat]
            
            # Group by category for cleaner display
            for cat in (all_categories if selected_cat == "All" else [selected_cat]):
                cat_faqs = [f for f in display_faqs if f["category"] == cat]
                if not cat_faqs: continue
                
                with st.expander(f"📂 {cat} ({len(cat_faqs)})", expanded=(selected_cat != "All")):
                    for f in cat_faqs:
                        st.markdown(f"**Q: {f['question']}**")
                        st.write(f"A: {f['answer']}")
                        st.caption(f"**Intent:** `{f['intent']}` | **Tags:** {', '.join(f.get('context_tags', []))}")
                        st.divider()
        except Exception as e:
            st.error(f"Error loading FAQ knowledge base: {e}")
    else:
        st.warning("Knowledge base file `pension_faqs.txt` not found.")

# 3. GLOBAL Chat Input (Pinned to bottom if active tab is chat)
if active_tab == "🛡️ Policy Assistant":
    if prompt := st.chat_input("Type a message..."):
        with live_placeholder:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Show spinner in the assistant's slot
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.orchestrator.process_turn(
                        st.session_state.ctx, 
                        text_input=prompt
                    )
                    st.markdown(response)
        
        # 4. Save to history and refresh
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response, 
                                          "is_terminal": st.session_state.ctx.state in (AgentState.RESOLVED, AgentState.ESCALATED),
                                          "esc_reason": st.session_state.ctx.escalation_reason})
        st.rerun()

# End of script
