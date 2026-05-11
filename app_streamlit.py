import streamlit as st
import os
import base64
from dotenv import load_dotenv

# Load environment variables BEFORE importing app components
load_dotenv(override=True)

import sys
from pathlib import Path
root_dir = Path(__file__).parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.agent import AgentOrchestrator, AgentState
from app.rag_client import RAGClient

# Optional Mic Recorder
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except ImportError:
    HAS_MIC = False
    mic_recorder = None

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
    /* header {visibility: hidden;}  <-- Removed this as it hides the sidebar toggle */
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def show_asr_status():
    """Helper to show ASR health in sidebar."""
    if "channel" in st.session_state and st.session_state.channel == "voice":
        is_mock = st.session_state.orchestrator.asr_is_mock
        if is_mock:
            st.warning("⚠️ **ASR is in Mock Mode.** Audio interaction will not be transcribed. Please check your `GROQ_API_KEY` or `ASR_BACKEND` settings.")
        else:
            st.success("🎤 **ASR is Live (Groq)**")

def process_user_input(text=None, audio=None):
    """
    Unified handler for chat and voice inputs.
    Updates session state messages and triggers orchestrator logic.
    """
    if not text and not audio:
        return

    # 1. Process via Orchestrator
    with st.spinner("Agent is thinking..."):
        user_text = None
        if audio:
            # First pass: Transcription for UI feedback
            user_text = st.session_state.orchestrator.transcribe_turn(audio, st.session_state.ctx.session_id)
            if not user_text:
                st.warning("⚠️ **Transcription Failed.** Could not catch that. If you are in a noisy environment or using a weak microphone, please try again or switch to Chat mode.")
                return
            # Record user turn
            st.session_state.messages.append({"role": "user", "content": user_text})
        elif text:
            user_text = text
            st.session_state.messages.append({"role": "user", "content": user_text})

        # Second pass: Full agent logic (RAG, SoR, Verification)
        response = st.session_state.orchestrator.process_turn(
            st.session_state.ctx, 
            audio_bytes=audio, 
            text_input=text,
            user_text=user_text
        )
        
        # Record assistant turn
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Final rerun to update UI components (sidebar, status, message list)
        st.rerun()

# --- Sidebar for State Monitoring ---
with st.sidebar:
    st.title("🤖 Status Monitor")
    
    # --- New Channel Selection ---
    st.subheader("Interaction Channel")
    channel = st.radio("Select Channel", ["Chat", "Voice"], index=0, horizontal=True)
    st.session_state.channel = channel.lower()
    
    st.divider()
    
    # Initialize Orchestrator based on channel
    if ("orchestrator" not in st.session_state 
            or st.session_state.get("_last_channel") != st.session_state.channel
            or not hasattr(st.session_state.orchestrator, "transcribe_turn")):
        st.session_state.orchestrator = AgentOrchestrator(channel=st.session_state.channel)
        st.session_state.ctx = st.session_state.orchestrator.new_session()
        st.session_state.messages = []
        st.session_state._last_channel = st.session_state.channel
        
        # Initial greeting
        resp = st.session_state.orchestrator.process_turn(st.session_state.ctx, text_input="hello")
        st.session_state.messages.append({"role": "assistant", "content": resp})

    # Now show ASR status for the correctly initialized orchestrator
    show_asr_status()

    st.write(f"**Session ID:** `{st.session_state.ctx.session_id}`")
    
    # State Display
    state_color = "🟢" if st.session_state.ctx.state == AgentState.RESOLVED else \
                  "🔴" if st.session_state.ctx.state == AgentState.ESCALATED else "🟡"
    st.subheader(f"Current State: {state_color}")
    st.info(f"**{st.session_state.ctx.state.value.replace('_', ' ').upper()}**")
    
    # Intent & Policy
    if st.session_state.ctx.call_intent:
        st.success(f"🎯 **Intent:** {st.session_state.ctx.call_intent.replace('_', ' ').title()}")

    if st.session_state.ctx.policy_number:
        with st.expander("📄 Policy Details", expanded=True):
            st.write(f"**Number:** {st.session_state.ctx.policy_number}")
            if st.session_state.ctx.product_type:
                st.write(f"**Product:** {st.session_state.ctx.product_type.title()}")

    # Collected Entities
    with st.expander("👤 Collected Entities", expanded=True):
        entities = st.session_state.ctx.caller_entities
        any_ent = False
        for field in entities.__dataclass_fields__:
            val = getattr(entities, field)
            if val:
                st.write(f"**{field.replace('_', ' ').title()}:** {val}")
                any_ent = True
        
        # Show Verification Result Scores if available
        if st.session_state.ctx.verification_result:
            st.divider()
            st.caption("Fuzzy Match Scores:")
            for f, res in st.session_state.ctx.verification_result.results.items():
                icon = "✅" if res.passed else "❌"
                p_note = " (Phonetic)" if getattr(res, 'phonetic_match', False) else ""
                st.write(f"{icon} {f.title()}: {res.score:.0f}%{p_note}")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.ctx = st.session_state.orchestrator.new_session()
        st.session_state.messages = []
        st.rerun()



# Main UI
tabs = st.tabs(["🛡️ AI Assistant", "📊 Call Statistics", "📚 Knowledge Hub"])

with tabs[0]:
    st.title("🛡️ Insurance Policy Assistant")
    st.info(f"Mode: **{st.session_state.channel.upper()}**")

    # 1. Render History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio_b64") and st.session_state.channel == "voice":
                st.audio(base64.b64decode(message["audio_b64"]), format="audio/wav")

    # 2. Input Layer
    if st.session_state.channel == "voice":
        st.write("---")
        if HAS_MIC:
            cols = st.columns([1, 4])
            with cols[0]:
                audio_rec = mic_recorder(
                    start_prompt="🎤 Start Speaking",
                    stop_prompt="🛑 Stop",
                    key='recorder'
                )
            
            if audio_rec:
                import hashlib
                audio_hash = hashlib.md5(audio_rec['bytes']).hexdigest() if audio_rec['bytes'] else None
                if audio_hash and st.session_state.get('last_audio_hash') != audio_hash:
                    st.session_state.last_audio_hash = audio_hash
                    process_user_input(audio=audio_rec['bytes'])
        else:
            st.warning("⚠️ Microphone component (`streamlit-mic-recorder`) is not installed. Using file upload.")
        
        audio_file = st.file_uploader("Upload Audio Interaction (.wav, .mp3)", type=["wav", "mp3"])
        if audio_file:
            import hashlib
            file_bytes = audio_file.read()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            if st.session_state.get('last_file_hash') != file_hash:
                st.session_state.last_file_hash = file_hash
                process_user_input(audio=file_bytes)

    if prompt := st.chat_input("Type your message here..."):
        process_user_input(text=prompt)

with tabs[1]:
    st.title("📊 Call Statistics")
    if os.path.exists("analytics.csv"):
        try:
            import pandas as pd
            df = pd.read_csv("analytics.csv")
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Calls", len(df))
            esc_rate = (df['escalated'] == True).mean() * 100
            m2.metric("Escalation Rate", f"{esc_rate:.1f}%")
            m3.metric("Avg Duration", f"{df['duration_sec'].mean():.1f}s")
            
            st.divider()
            
            # Timeline
            st.subheader("Call Volume Over Time")
            df['start_time'] = pd.to_datetime(df['start_time'])
            vol_over_time = df.set_index('start_time').resample('H').size()
            st.line_chart(vol_over_time)
            
            # State Distribution
            st.subheader("Final State Distribution")
            state_counts = df['final_state'].value_counts()
            st.bar_chart(state_counts)
            
        except Exception as e:
            st.error(f"Could not load analytics: {e}")
            st.info("Ensure `pandas` is installed in your environment.")
    else:
        st.warning("No analytics data found yet. Complete some calls to see statistics.")

with tabs[2]:
    st.title("📚 Knowledge Hub")
    rag = RAGClient(st.session_state.ctx.cache)
    
    st.subheader("🔍 Search Knowledge Base")
    search_mode = st.radio("Search Type", ["Text Match", "Semantic AI Search"], horizontal=True, label_visibility="collapsed")
    search = st.text_input("Enter your query or keywords...", "")
    
    if search:
        if search_mode == "Semantic AI Search":
            with st.spinner("AI is searching vector database..."):
                res = rag.query(search, session_id="ui_hub")
                if res.chunks:
                    st.success(f"Found {len(res.chunks)} relevant results.")
                    for chunk in res.chunks:
                        score = chunk.score * 100
                        status = "✅ High" if score >= 75 else "⚠️ Medium" if score >= 60 else "🔴 Low"
                        with st.expander(f"📌 {chunk.section} ({score:.1f}% Confidence - {status})"):
                            st.write(chunk.text)
                            st.caption(f"Source: {chunk.source_doc} | Brand: {chunk.heritage_brand or 'All'}")
                else:
                    st.warning("No relevant articles found with high enough confidence.")
        else:
            # Standard Text Match Filter
            faqs = rag.list_all_faqs()
            found = 0
            for faq in faqs:
                title = faq.get('question') or faq.get('section') or "Untitled"
                content = faq.get('answer') or faq.get('text') or ""
                if search.lower() in title.lower() or search.lower() in content.lower():
                    found += 1
                    p_type = (faq.get('product_type') or faq.get('category') or 'General').title()
                    with st.expander(f"📌 {title} ({p_type})"):
                        st.write(content)
                        st.caption(f"Source: {faq.get('source_doc', 'Knowledge Base')}")
            if found == 0:
                st.info("No matching FAQs found. Try different keywords or switch to 'Semantic AI Search'.")
    else:
        # Default view: List all if no search
        st.info("Enter a search term above or browse all FAQs below.")
        faqs = rag.list_all_faqs()
        for faq in faqs:
            title = faq.get('question') or faq.get('section') or "Untitled"
            content = faq.get('answer') or faq.get('text') or ""
            p_type = (faq.get('product_type') or faq.get('category') or 'General').title()
            with st.expander(f"📌 {title} ({p_type})"):
                st.write(content)
                st.caption(f"Source: {faq.get('source_doc', 'Knowledge Base')}")
