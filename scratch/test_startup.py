import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Mock missing Dialogflow environment variables (though likely already missing)
os.environ.pop("DIALOGFLOW_PROJECT_ID", None)
os.environ.pop("DIALOGFLOW_AGENT_ID", None)

def test_startup():
    print("Attempting to initialize AgentOrchestrator in 'chat' mode...")
    from app.agent import AgentOrchestrator
    try:
        orchestrator = AgentOrchestrator(channel="chat")
        ctx = orchestrator.new_session()
        print("✅ Orchestrator initialized successfully in 'chat' mode.")
        
        print("Testing process_turn in 'chat' mode...")
        response = orchestrator.process_turn(ctx, text_input="hello")
        print(f"✅ process_turn successful. Agent response: {response[:50]}...")
    except AttributeError as e:
        print(f"❌ Failed to process turn: AttributeError on {e}")
        return False
    except KeyError as e:
        print(f"❌ Failed to initialize Orchestrator: KeyError on {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize Orchestrator: {type(e).__name__}: {e}")
        return False

    print("\nAttempting to initialize AgentOrchestrator in 'voice' mode (should now be safer)...")
    try:
        orchestrator_voice = AgentOrchestrator(channel="voice")
        print("✅ Orchestrator initialized successfully in 'voice' mode (mocked fallback).")
    except Exception as e:
        print(f"❌ Failed to initialize Orchestrator in voice mode: {type(e).__name__}: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_startup()
    if not success:
        sys.exit(1)
