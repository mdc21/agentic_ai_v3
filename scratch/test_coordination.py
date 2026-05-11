import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

def test_coordination_resilience():
    print("Testing RAG/SoR Coordination Resilience...")
    from app.agent import AgentOrchestrator, ConversationContext, AgentState, AgentTurn
    from app.rag_client import RAGResult, RAGChunk
    
    # Mock RAGResult with a low-score chunk and answerable=False
    low_score_chunk = RAGChunk(chunk_id="c1", text="some text", score=0.4, source_doc="doc1.txt", section="S1")
    mock_rag_result = RAGResult(
        query="What is my policy value?",
        query_hash="hash123",
        chunks=[low_score_chunk],
        answerable=False
    )
    
    # Mock ToolRegistry to return successful data
    mock_sor_data = {"policy_value": "£50,000", "status": "active"}

    orchestrator = AgentOrchestrator(channel="chat")
    ctx = orchestrator.new_session()
    ctx.state = AgentState.SERVE_INTENT
    ctx.policy_number = "TEST-123"
    ctx.product_type = "life"
    
    # Mocked Turn: LLM mistakenly requested RAG for a data query
    turn = AgentTurn(
        action="policy_valuation",
        rag_query="What is my policy value?", # LLM being cautious
        confidence=0.9
    )

    with patch.object(orchestrator, "_get_rag_context", return_value=mock_rag_result):
        with patch.object(orchestrator, "_get_sor_data", return_value=mock_sor_data):
            with patch.object(orchestrator, "_synthesize_unified_response", return_value="Your policy value is £50,000.") as mock_synth:
                
                response = orchestrator._dispatch(ctx, turn)
                
                print(f"  Agent Response: {response}")
                
                # VERIFICATION:
                # 1. It should NOT have escalated.
                if "Specialist" in response or "connecting you" in response:
                    print("❌ FAILED: Agent escalated despite having SoR data!")
                    return False
                
                # 2. It should have called synthesis with EMPTY rag_context but valid sor_data.
                args, kwargs = mock_synth.call_args
                passed_rag_context = args[2] if len(args) > 2 else None
                passed_sor_data = args[3] if len(args) > 3 else None
                
                if passed_rag_context == "":
                    print("✅ RAG context correctly emptied due to low confidence.")
                else:
                    print(f"❌ RAG context was not empty: {passed_rag_context}")
                    return False
                
                if passed_sor_data == mock_sor_data:
                    print("✅ SoR data correctly passed to synthesis.")
                else:
                    print(f"❌ SoR data mismatch: {passed_sor_data}")
                    return False

    print("✅ Coordination Resilience Test PASSED!")
    return True

if __name__ == "__main__":
    # Mock environment to avoid dependency issues in basic logic test
    os.environ["GROQ_API_KEY"] = "mock"
    os.environ["PINECONE_API_KEY"] = "mock"
    
    try:
        success = test_coordination_resilience()
        if not success: sys.exit(1)
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
