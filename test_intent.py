import json
import os
os.environ["OPENAI_API_KEY"] = "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"
from app.llm_client import LLMClient
client = LLMClient()
from app.prompts import SYSTEM_PROMPT
msg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "What tax is applicable to pension encashment?"}]
turn = client.call(msg, "")
print(turn.action_intent, turn.rag_query)
