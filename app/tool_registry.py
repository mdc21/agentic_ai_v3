"""
Tool registry — maps action_intent strings to SOR API endpoint configurations.

Loaded from tool_registry.yaml (or falls back to the built-in default config).
Each action has:
  - endpoint template (with {policy_number} placeholders)
  - HTTP method
  - response_fields: the ONLY fields passed to the LLM (data minimisation)
  - cache_key template
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    action_intent: str
    endpoint: str          # URL path template, e.g. /policies/{policy_number}/valuation
    method: str            # GET | POST
    response_fields: list[str]   # fields to extract from response (data minimisation)
    cache_key_template: str      # e.g. "valuation:{policy_number}"
    requires_auth: bool = True
    
    # v3 fields from YAML
    flags_fields: Optional[list[str]] = None
    party_flags_fields: Optional[list[str]] = None
    response_fields_pension: Optional[list[str]] = None
    response_fields_life: Optional[list[str]] = None
    response_fields_annuity: Optional[list[str]] = None
    response_fields_protection: Optional[list[str]] = None

    def resolve_endpoint(self, params: dict) -> str:
        return self.endpoint.format(**params)

    def resolve_cache_key(self, params: dict) -> str:
        return self.cache_key_template.format(**params)


# ── Default registry (used when no YAML file is present) ─────────────────────

DEFAULT_REGISTRY: dict[str, ToolConfig] = {
    "policy_exist_platform_directory_check": ToolConfig(
        action_intent="policy_exist_platform_directory_check",
        endpoint="/platform-directory/policies/{policy_number}",
        method="GET",
        response_fields=["policy_number", "heritage_brand", "product_type", "sor_system", "exists"],
        cache_key_template="platform_dir:{policy_number}",
    ),
    "policy_exist_SOR_check": ToolConfig(
        action_intent="policy_exist_SOR_check",
        endpoint="/sor1/policies/{policy_number}",
        method="GET",
        response_fields=["policy_number", "heritage_brand", "product_type", "status", "inception_date"],
        flags_fields=["value_restriction", "share_restriction"],
        cache_key_template="sor1:{policy_number}",
    ),
    "party_role_address_details": ToolConfig(
        action_intent="party_role_address_details",
        endpoint="/sor1/policies/{policy_number}/parties",
        method="GET",
        response_fields=["party_id", "role", "first_name", "last_name", "address_line1", "postcode", "date_of_birth"],
        party_flags_fields=["vulnerable_customer", "fcu_flag"],
        cache_key_template="party:{policy_number}",
    ),
    "policy_basic_details": ToolConfig(
        action_intent="policy_basic_details",
        endpoint="/sor1/policies/{policy_number}",
        method="GET",
        response_fields=["heritage_brand", "product_type", "status", "inception_date", "sor_system"],
        cache_key_template="sor1:{policy_number}",
    ),
    "policy_valuation": ToolConfig(
        action_intent="policy_valuation",
        endpoint="/sor1/policies/{policy_number}/valuation",
        method="GET",
        response_fields=["total_value", "surrender_value", "valuation_date", "currency"],
        response_fields_pension=["total_value", "transfer_value", "GAR", "GMP", "GMR", "charges", "tax", "valuation_date", "currency"],
        response_fields_life=["surrender_value", "bonus_amount", "charges", "valuation_date", "currency"],
        response_fields_annuity=["income_amount", "escalation_rate", "commencement_date", "currency"],
        response_fields_protection=["sum_assured", "premium", "expiry_date", "currency"],
        cache_key_template="valuation:{policy_number}",
    ),
    "adviser_details": ToolConfig(
        action_intent="adviser_details",
        endpoint="/sor1/policies/{policy_number}/parties?role=financial_adviser",
        method="GET",
        response_fields=["party_id", "firm_name", "address_line1", "postcode"],
        cache_key_template="adviser:{policy_number}",
    ),
}

# ── Mock responses for local/test mode ────────────────────────────────────────

_MOCK_RESPONSES: dict[str, dict] = {
    "valuation:ABC/123-45": {
        "total_value": 45230.50,
        "surrender_value": 41200.00,
        "transfer_value": 45230.50,
        "bonus_amount": 5000.0,
        "GAR": "Yes",
        "GMP": 120.0,
        "GMR": 100.0,
        "charges": 25.5,
        "tax": 0.0,
        "unit_price": 1.85,
        "unit_count": 24449,
        "valuation_date": "2024-04-30",
        "currency": "GBP",
    },
    "surrender:ABC/123-45": {
        "surrender_value": 41200.00,
        "surrender_penalty": 4030.50,
        "effective_date": "2024-05-01",
        "currency": "GBP",
    },
    "benefits:ABC/123-45": {
        "benefit_type": "Death benefit",
        "benefit_amount": 100000.00,
        "currency": "GBP",
        "in_trust": False,
    },
    "adviser:ABC/123-45": {
        "party_id": "A001",
        "role": "financial_adviser",
        "firm_name": "Blackwood Financial Services Ltd",
        "address_line1": "22 Market Square",
        "postcode": "M2 3BB",
    },
}


class ToolRegistry:
    """
    Maps action_intent → ToolConfig. Executes tool calls against SOR APIs
    with cache-before-call logic and response field filtering.
    """

    def __init__(self, session_cache, audit_logger) -> None:
        self._registry = self._load_registry()
        self._cache = session_cache
        self._audit = audit_logger
        self._mock = os.getenv("USE_MOCK_POLICY_API", "false").lower() == "true"
        base_url = os.getenv("POLICY_API_BASE_URL", "https://api.example.com/v1")
        if not self._mock:
            self._http = httpx.Client(base_url=base_url, timeout=10.0)

    def _load_registry(self) -> dict[str, ToolConfig]:
        registry_path = Path(os.getenv("TOOL_REGISTRY_PATH", "tool_registry.yaml"))
        if yaml and registry_path.exists():
            try:
                with open(registry_path) as f:
                    raw = yaml.safe_load(f)
                return {k: ToolConfig(action_intent=k, **v) for k, v in raw.items()}
            except Exception as exc:
                logger.warning("Could not load tool registry YAML: %s — using defaults", exc)
        return DEFAULT_REGISTRY

    def has_action(self, action_intent: str) -> bool:
        return action_intent in self._registry

    def call(
        self,
        action_intent: str,
        params: dict,
        session_id: str,
        turn: int,
        product_type: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Execute a tool call for the given action_intent.
        Checks cache first. On miss, calls SOR API, stores full response,
        then returns only the response_fields defined for this action.
        Returns None on error.
        """
        config = self._registry.get(action_intent)
        if config is None:
            logger.error("Unknown action_intent: %r", action_intent)
            return None

        cache_key = config.resolve_cache_key(params)
        cached = self._cache.get(cache_key)

        # Select response fields based on product_type if applicable
        fields = config.response_fields
        if product_type:
            pt = product_type.lower()
            if pt == "pension" and config.response_fields_pension: fields = config.response_fields_pension
            elif pt == "life" and config.response_fields_life: fields = config.response_fields_life
            elif pt in ("annuity", "annuities") and config.response_fields_annuity: fields = config.response_fields_annuity
            elif pt == "protection" and config.response_fields_protection: fields = config.response_fields_protection

        if cached is not None:
            logger.info("[%s] Cache HIT for %r", session_id, cache_key)
            self._audit.log_tool_call(session_id, turn, action_intent, params,
                                      fields, hit=True)
            return self._filter_fields(cached, fields)

        # Cache miss — call SOR API (or mock)
        full_response = self._fetch(config, params, session_id)
        if full_response is None:
            return None

        # Store full response in cache
        self._cache.set(cache_key, full_response)
        self._audit.log_tool_call(session_id, turn, action_intent, params,
                                  fields, hit=False)

        # Return only the allowed fields to the orchestrator (data minimisation)
        return self._filter_fields(full_response, fields)

    def _fetch(self, config: ToolConfig, params: dict, session_id: str) -> Optional[dict]:
        if self._mock:
            cache_key = config.resolve_cache_key(params)
            mock = _MOCK_RESPONSES.get(cache_key)
            if mock:
                return mock
            # Fall through to policy tool (handled by PolicyAPIClient in orchestrator)
            return {"_delegate_to_policy_client": True}

        try:
            url = config.resolve_endpoint(params)
            r = self._http.request(config.method, url)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            logger.error("[%s] SOR API error for %r: %s", session_id, config.action_intent, exc)
            return None

    @staticmethod
    def _filter_fields(response: dict, fields: list[str]) -> dict:
        """Return only the specified fields from a response dict."""
        return {k: response[k] for k in fields if k in response}
