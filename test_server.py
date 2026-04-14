import httpx
import json

def test_chat_flow():
    url = "http://localhost:8000/chat"
    session_id = None
    
    turns = [
        "hello",
        "I'd like to check my pension value",
        "My policy number is ABC slash 123 dash 45",
        "yes that is correct",
        "I am the policy holder",
        "Jonathan Smith",
        "14 High St",
        "M1 1AA",
        "22nd August 1975",
        "What is my current pension value?"
    ]
    
    print("Starting Chat Flow Test...\n")
    
    for i, msg in enumerate(turns):
        print(f"Turn {i+1} - User: {msg}")
        payload = {"message": msg}
        if session_id:
            payload["session_id"] = session_id
            
        try:
            r = httpx.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            session_id = data["session_id"]
            print(f"Turn {i+1} - Bot: {data['response']}")
            print(f"Turn {i+1} - State: {data['state']}\n")
            
        except Exception as e:
            print(f"Error on turn {i+1}: {e}")
            break

if __name__ == "__main__":
    test_chat_flow()
