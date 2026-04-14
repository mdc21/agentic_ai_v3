
import sys
import os
sys.path.append(os.getcwd())

from app.agent import AgentOrchestrator, AgentState
from app.llm_client import Entities

def test_flow(channel: str):
    print(f"\n--- Testing Channel: {channel} ---")
    orch = AgentOrchestrator(channel=channel)
    ctx = orch.new_session()
    
    # Turn 1: User says hello and provides policy number
    print(f"Turn 1: User provides policy number")
    response = orch.process_turn(ctx, "Hello, my policy is ABC/123-45")
    print(f"Bot response: {response}")
    print(f"Target state: {ctx.state.name}")
    
    if channel == "chat":
        # In chat, it should jump straight to identification or directory check
        # (Actually, in our mock, it might go straight to IDENTIFY_CALLER if the directory check passes instantly)
        if "confirm" in response.lower():
            print("FAILURE: Bot asked for confirmation in CHAT channel.")
        else:
            print("SUCCESS: Bot skipped confirmation in CHAT channel.")
    else:
        # In voice, it MUST confirm
        if "confirm" in response.lower() or "is that correct" in response.lower():
            print("SUCCESS: Bot asked for confirmation in VOICE channel.")
        else:
            print("FAILURE: Bot skipped confirmation in VOICE channel.")

if __name__ == "__main__":
    test_flow("chat")
    test_flow("voice")
