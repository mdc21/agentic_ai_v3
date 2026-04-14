"""
run_demo.py v3 — exercises all major conversation paths.

Run: USE_MOCK_ASR=true USE_MOCK_POLICY_API=true USE_MOCK_RAG=true \
     ANTHROPIC_API_KEY=sk-ant-... python run_demo.py [1-8|all]
"""
import logging, os, sys
os.environ.setdefault("USE_MOCK_ASR","true")
os.environ.setdefault("USE_MOCK_POLICY_API","true")
os.environ.setdefault("USE_MOCK_RAG","true")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from app.agent import AgentOrchestrator

SCENARIOS = {
    "1": {
        "name": "Happy path — policy holder, pension valuation",
        "turns": [
            "hello",
            "I'd like to check my pension value",
            "My policy number is ABC slash 123 dash 45",
            "yes that is correct",
            "I am the policy holder",
            "Jonathan Smith", "14 High St", "M1 1AA", "22nd August 1975",
            "What is my current pension value?",
        ]
    },
    "2": {
        "name": "FA representative — full verification chain",
        "turns": [
            "hello",
            "I need to check a client's pension",
            "ABC slash 123 dash 45",
            "yes",
            "I am calling on behalf of a financial adviser",
            "Jonathan Smith", "14 High Street", "M1 1AA", "22 August 1975",
            "Blackwood Financial Services Ltd",
            "22 Market Square",
            "M2 3BB",
            "What is the transfer value?",
        ]
    },
    "3": {
        "name": "Trustee — immediate escalation",
        "turns": [
            "hello", "I need to discuss the policy",
            "ABC slash 123 dash 45", "yes that is right",
            "I am one of the trustees",
        ]
    },
    "4": {
        "name": "Verification failure — wrong DOB",
        "turns": [
            "hello", "I want to check my policy",
            "ABC slash 123 dash 45", "yes",
            "I am the policy holder",
            "Jonathan Smith", "14 High Street", "M1 1AA",
            "1st January 1990",   # wrong DOB
        ]
    },
    "5": {
        "name": "RAG query — general pension transfer question",
        "turns": [
            "hello",
            "I have a question about pension transfers",
            "ABC slash 123 dash 45", "yes correct",
            "I am the policy holder",
            "Jonathan Smith", "14 High St", "M1 1AA", "22 August 1975",
            "Can I transfer my pension to another provider?",
        ]
    },
    "6": {
        "name": "Caller requests human — mid-conversation",
        "turns": [
            "hello",
            "I want to check my policy",
            "ABC slash 123 dash 45", "yes",
            "I am the policy holder",
            "Jonathan Smith",
            "Actually, can I just speak to someone please",   # human request
        ]
    },
    "7": {
        "name": "Loop detection — repeated same question",
        "turns": [
            "hello", "I want to know my policy value",
            "ABC slash 123 dash 45", "yes",
            "I am the policy holder",
            "Jonathan Smith", "14 High Street", "M1 1AA", "22 August 1975",
            "What is my policy value?",
            "What is my policy value?",   # repeat 1
            "What is my policy value?",   # repeat 2 — loop trigger
        ]
    },
    "8": {
        "name": "Policy number retries exceeded",
        "turns": [
            "hello", "check my policy",
            "erm I'm not sure",         # retry 1
            "um something something",   # retry 2
            "I don't know it",          # retry 3 — escalate
        ]
    },
}


def run_scenario(key: str) -> None:
    sc = SCENARIOS[key]
    agent = AgentOrchestrator(channel="chat")
    ctx   = agent.new_session()

    print(f"\n{'='*65}")
    print(f"SCENARIO {key}: {sc['name']}")
    print(f"Session: {ctx.session_id}")
    print(f"{'='*65}")

    for user_input in sc["turns"]:
        response = agent.process_turn(ctx, text_input=user_input)
        print(f"\n  Caller : {user_input}")
        print(f"  Agent  : {response[:220]}")
        print(f"  State  : {ctx.state.value}  |  product={ctx.product_type}  brand={ctx.heritage_brand}")
        if ctx.state.value in ("resolved","escalated"):
            break

    print(f"\n{'-'*65}")
    if ctx.state.value == "resolved":
        print(f"  OUTCOME  : RESOLVED  |  Intent: {ctx.call_intent}")
    else:
        print(f"  OUTCOME  : ESCALATED  |  Reason: {ctx.escalation_reason}")
    print(f"  Cache    : {ctx.cache.keys()}")
    print(f"  Intents  : {ctx.intents_served}")
    print(f"{'='*65}")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key == "all":
        for k in SCENARIOS: run_scenario(k)
    elif key in SCENARIOS:
        run_scenario(key)
    else:
        print("Usage: python run_demo.py [1-8|all]")
        for k,v in SCENARIOS.items(): print(f"  {k}  {v['name']}")
