import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

def test_groq_failover():
    print("Testing Groq Model-Level Failover...")
    
    # We need to mock the groq library because we might actually be rate limited
    with patch("groq.Groq") as MockGroq:
        from groq import RateLimitError
        from app.llm_client import LLMClient
        
        mock_client = MockGroq.return_value
        
        # Define a side effect for chat.completions.create
        # 1st call: Raise RateLimitError
        # 2nd call: Return success
        def side_effect(model, **kwargs):
            if model == "llama-3.3-70b-versatile":
                print(f"  Simulating Rate Limit for {model}...")
                raise RateLimitError("Rate limit reached", response=MagicMock(), body={})
            else:
                print(f"  Success with fallback model: {model}")
                mock_resp = MagicMock()
                mock_resp.choices[0].message.content = '{"intent": "test", "action": "respond", "caller_response": "Hi"}'
                mock_resp.usage.prompt_tokens = 10
                mock_resp.usage.completion_tokens = 10
                return mock_resp
        
        mock_client.chat.completions.create.side_effect = side_effect
        
        # Initialize client
        os.environ["GROQ_API_KEY"] = "mock_key"
        llm = LLMClient()
        
        # Execute call
        try:
            turn = llm._call_groq([{"role": "user", "content": "hello"}])
            print(f"✅ Failover successful! Model used: {turn.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failover failed: {type(e).__name__}: {e}")
            return False

if __name__ == "__main__":
    # Ensure dependencies are available for the import
    # This might fail if groq isn't in sys.path/venv
    success = test_groq_failover()
    if not success:
        sys.exit(1)
