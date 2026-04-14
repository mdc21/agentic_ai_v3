
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "policy_number": {"type": ["string", "null"]},
        "first_name": {"type": ["string", "null"]},
        "last_name": {"type": ["string", "null"]},
        "address_line1": {"type": ["string", "null"]},
        "postcode": {"type": ["string", "null"]},
        "date_of_birth": {"type": ["string", "null"]},
        "adviser_firm_name": {"type": ["string", "null"]},
        "adviser_address_line1": {"type": ["string", "null"]},
        "adviser_postcode": {"type": ["string", "null"]},
        "adviser_rep_name": {"type": ["string", "null"]}
    },
    "additionalProperties": False
}

AGENT_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "caller_type": {
            "type": "string",
            "enum": ["policy_holder", "fa_representative", "trustee", "employer", "third_party", "unknown"]
        },
        "entities": ENTITY_SCHEMA,
        "action_intent": {"type": "string"},
        "rag_query": {"type": ["string", "null"]},
        "duress_signal": {"type": "boolean"},
        "caller_response": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["intent", "caller_type", "entities", "action_intent", "caller_response", "confidence"],
    "additionalProperties": False
}
