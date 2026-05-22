# tests/test_nato_verification.py
import os, pytest
os.environ.update({"USE_MOCK_ASR": "true", "USE_MOCK_POLICY_API": "true",
                   "USE_MOCK_RAG": "true", "CHANNEL": "voice"})
from unittest.mock import patch, MagicMock
from app.agent import AgentOrchestrator, AgentState
from app.llm_client import AgentTurn, Entities

def make_turn(action, intent="", postcode=None, policy=None, affirm_text=""):
    e = Entities()
    if postcode: e.postcode = postcode
    if policy:   e.policy_number = policy
    return AgentTurn(action=action, intent=intent, entities=e,
                     caller_response="test response", user_text=affirm_text)

class TestNATOPolicyNumber:
    def setup_method(self):
        self.agent = AgentOrchestrator(channel="voice")
        self.ctx = self.agent.new_session()

    def test_policy_captured_triggers_nato_readback(self):
        """When policy is captured on voice, NATO readback should fire."""
        self.ctx.caller_entities.policy_number = "ABC/123-45"
        turn = make_turn("policy_exist_platform_directory_check")
        with patch.object(self.agent, "_speak", side_effect=lambda t, c: t):
            resp = self.agent._dispatch_v2(self.ctx, turn)
        assert "Alpha Bravo Charlie" in resp or "NATO" in resp.lower() or "Is that correct" in resp

    def test_yes_confirms_policy(self):
        """User saying Yes should set policy_confirmed=True."""
        self.ctx.caller_entities.policy_number = "ABC/123-45"
        self.ctx.metadata["last_action"] = "confirm_policy_number"
        turn = make_turn("confirm_policy_number", intent="affirmative", affirm_text="yes")
        with patch.object(self.agent, "_platform_directory_check", return_value="OK"):
            with patch.object(self.agent, "_speak", side_effect=lambda t, c: t):
                self.agent._dispatch_v2(self.ctx, turn)
        assert self.ctx.metadata.get("policy_confirmed") is True


class TestNATOPostcode:
    def setup_method(self):
        self.agent = AgentOrchestrator(channel="voice")
        self.ctx = self.agent.new_session()
        self.ctx.state = AgentState.VERIFY_POLICYHOLDER

    def test_postcode_captured_triggers_nato(self):
        """Postcode on voice should trigger NATO readback before proceeding."""
        self.ctx.caller_entities.postcode = "M1 1AA"
        turn = make_turn("continue_verification")
        with patch.object(self.agent, "_speak", side_effect=lambda t, c: t):
            resp = self.agent._dispatch_v2(self.ctx, turn)
        assert "Is that correct" in resp
        assert self.ctx.metadata.get("last_action") == "confirm_postcode"

    def test_postcode_confirmation_sets_flag(self):
        """Confirming postcode sets postcode_confirmed flag."""
        self.ctx.caller_entities.postcode = "M1 1AA"
        self.ctx.metadata["last_action"] = "confirm_postcode"
        turn = make_turn("confirm_postcode", intent="affirmative", affirm_text="yes that is correct")
        with patch.object(self.agent, "process_turn", return_value="Next question"):
            with patch.object(self.agent, "_speak", side_effect=lambda t, c: t):
                self.agent._dispatch_v2(self.ctx, turn)
        assert self.ctx.metadata.get("postcode_confirmed") is True
