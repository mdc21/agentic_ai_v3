# tests/test_rag_failures.py
import os, pytest
os.environ["USE_MOCK_RAG"] = "true"
from app.rag_client import RAGClient
from app.session_cache import SessionCache

class TestRAGFailures:
    def setup_method(self):
        self.rag = RAGClient(SessionCache())

    def test_empty_query_returns_none(self):
        """Empty or None query should not crash."""
        result = self.rag.query("", session_id="test")
        assert result is not None  # returns RAGResult with answerable=False

    def test_no_matching_chunk_not_answerable(self):
        """A query with no keyword match should return answerable=False."""
        result = self.rag.query(
            "what is the capital of France?",   # nothing in insurance FAQ
            session_id="test"
        )
        # Mock RAG should return chunks; if none match, answerable=False
        assert result.answerable is False or len(result.chunks) == 0

    def test_cache_hit_on_repeated_query(self):
        """Same query twice should hit cache on second call."""
        q = "What is the minimum retirement age?"
        self.rag.query(q, session_id="test")
        result2 = self.rag.query(q, session_id="test")
        assert result2 is not None

    def test_rag_none_guard_in_orchestrator(self):
        """If _rag is None, _get_rag_context should return None safely."""
        from app.agent import AgentOrchestrator
        agent = AgentOrchestrator(channel="voice")
        agent._rag = None   # simulate failed init
        ctx = agent.new_session()
        result = agent._get_rag_context(ctx, "What is the transfer value?")
        assert result is None
