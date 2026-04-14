"""
Policy API tools — wraps your System of Record REST endpoints.

Set POLICY_API_BASE_URL in .env. For local development, set USE_MOCK_POLICY_API=true
and the client returns realistic stub data.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("POLICY_API_BASE_URL", "https://api.example.com/v1")
TIMEOUT = 10.0  # seconds


@dataclass
class PartyRecord:
    party_id: str
    role: str  # life_insured | policy_owner | financial_adviser | trustee | employer
    first_name: str
    last_name: str
    date_of_birth: str  # YYYY-MM-DD
    address_line1: str
    address_line2: str
    city: str
    postcode: str
    country: str = "GB"
    firm_name: str = ""          # adviser firm name (used for FA verification)
    role_status: str = "active"  # whether this party's role is active on the policy


@dataclass
class PolicyRecord:
    policy_number: str
    product_name: str
    status: str  # active | lapsed | cancelled
    commencement_date: str
    heritage_brand: str = "Brand_A"
    product_type: str = "life"
    parties: list[PartyRecord] = field(default_factory=list)


# ── Stub data for local/test mode ─────────────────────────────────────────────

_MOCK_POLICIES = {
    "ABC/123-45": PolicyRecord(
        policy_number="ABC/123-45",
        product_name="Whole of Life Assurance",
        status="active",
        commencement_date="2015-03-01",
        heritage_brand="Brand_A",
        product_type="life",
        parties=[
            PartyRecord(
                party_id="P001",
                role="policy_owner",
                first_name="Jonathan",
                last_name="Smith",
                date_of_birth="1975-08-22",
                address_line1="14 High Street",
                address_line2="",
                city="Manchester",
                postcode="M1 1AA",
            ),
            PartyRecord(
                party_id="P002",
                role="life_insured",
                first_name="Jonathan",
                last_name="Smith",
                date_of_birth="1975-08-22",
                address_line1="14 High Street",
                address_line2="",
                city="Manchester",
                postcode="M1 1AA",
            ),
            PartyRecord(
                party_id="P003",
                role="financial_adviser",
                first_name="Sarah",
                last_name="Jones",
                firm_name="Apex Wealth Management",
                date_of_birth="",
                address_line1="25 Baker Street",
                address_line2="",
                city="London",
                postcode="W1U 7AB",
                role_status="active",
            ),
        ],
    ),
    "AV-LIFE-789": PolicyRecord(
        policy_number="AV-LIFE-789",
        product_name="Life Insurance",
        status="active",
        commencement_date="2020-05-10",
        heritage_brand="Aviva",
        product_type="life",
        parties=[
            PartyRecord(
                party_id="AV001",
                role="policy_owner",
                first_name="Alice",
                last_name="Henderson",
                date_of_birth="1982-11-12",
                address_line1="7 Aviva Close",
                address_line2="",
                city="Norwich",
                postcode="NR1 1AA",
            ),
            PartyRecord(
                party_id="FA001",
                role="financial_adviser",
                first_name="Alex",
                last_name="Adviser",
                firm_name="Aviva Premier Advice",
                date_of_birth="",
                address_line1="1 Financial Way",
                address_line2="",
                city="London",
                postcode="EC2 2BB",
                role_status="active",
            ),
        ],
    ),
    "SW-PEN-456": PolicyRecord(
        policy_number="SW-PEN-456",
        product_name="Personal Pension Plan",
        status="active",
        commencement_date="2012-08-20",
        heritage_brand="Scottish Widows",
        product_type="pension",
        parties=[
            PartyRecord(
                party_id="SW001",
                role="policy_owner",
                first_name="Samuel",
                last_name="Miller",
                date_of_birth="1968-03-25",
                address_line1="Widows Walk",
                address_line2="",
                city="Edinburgh",
                postcode="EH1 2BB",
            ),
            PartyRecord(
                party_id="FA002",
                role="financial_adviser",
                first_name="Sarah",
                last_name="Sweety",
                firm_name="Blackwood Financial",
                date_of_birth="",
                address_line1="22 Market Square",
                address_line2="",
                city="Manchester",
                postcode="M2 3BB",
                role_status="active",
            ),
        ],
    ),
    "PH-ANN-123": PolicyRecord(
        policy_number="PH-ANN-123",
        product_name="Lifetime Annuity",
        status="active",
        commencement_date="2022-01-15",
        heritage_brand="Phoenix Life",
        product_type="annuity",
        parties=[
            PartyRecord(
                party_id="PH001",
                role="policy_owner",
                first_name="Phyllis",
                last_name="Thompson",
                date_of_birth="1953-06-30",
                address_line1="Rise Road",
                address_line2="",
                city="Birmingham",
                postcode="B1 1AA",
            ),
            PartyRecord(
                party_id="FA003",
                role="financial_adviser",
                first_name="Peter",
                last_name="Plan",
                firm_name="Heritage Planning Ltd",
                date_of_birth="",
                address_line1="8 Phoenix Plaza",
                address_line2="",
                city="Wythall",
                postcode="B47 6WG",
                role_status="active",
            ),
        ],
    )
}


class PolicyAPIClient:
    def __init__(self) -> None:
        self._mock = os.getenv("USE_MOCK_POLICY_API", "false").lower() == "true"
        if not self._mock:
            self._http = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)

    # ── Public tool methods ────────────────────────────────────────────────────

    def policy_exists(self, policy_number: str) -> bool:
        """Return True if the policy number exists in the SOR."""
        if self._mock:
            # Mock: any policy number is considered valid
            return bool(policy_number and policy_number.strip())
        try:
            r = self._http.get(f"/policies/{policy_number}/exists")
            return r.status_code == 200 and r.json().get("exists", False)
        except httpx.HTTPError as exc:
            logger.error("policyExists API error: %s", exc)
            return False

    def get_policy_details(self, policy_number: str) -> Optional[PolicyRecord]:
        """Return full policy record including all parties, or None if not found."""
        if self._mock:
            # First check exact hard-coded records
            normalised = policy_number.upper().replace(" ", "")
            for k, v in _MOCK_POLICIES.items():
                if k.upper().replace(" ", "") == normalised:
                    return v
            # Fallback: generate a realistic stub for any other policy number
            return PolicyRecord(
                policy_number=policy_number,
                product_name="Personal Pension Plan",
                status="active",
                commencement_date="2018-06-15",
                parties=[
                    PartyRecord(
                        party_id="P001",
                        role="policy_owner",
                        first_name="John",
                        last_name="Doe",
                        date_of_birth="1980-01-15",
                        address_line1="10 Example Street",
                        address_line2="",
                        city="London",
                        postcode="EC1A 1BB",
                    ),
                    PartyRecord(
                        party_id="P002",
                        role="financial_adviser",
                        first_name="David",
                        last_name="Brown",
                        firm_name="Premier Financial Services",
                        date_of_birth="",
                        address_line1="5 City Road",
                        address_line2="",
                        city="London",
                        postcode="EC1V 1AA",
                        role_status="active",
                    ),
                ],
            )
        try:
            r = self._http.get(f"/policies/{policy_number}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            parties = [PartyRecord(**p) for p in data.pop("parties", [])]
            return PolicyRecord(**data, parties=parties)
        except httpx.HTTPError as exc:
            logger.error("policyDetails API error: %s", exc)
            return None

    def get_party_for_role(self, policy: PolicyRecord, role: str) -> Optional[PartyRecord]:
        """Return the first party matching the given role on a policy."""
        return next((p for p in policy.parties if p.role == role), None)

    def format_policy_summary(self, policy: PolicyRecord) -> str:
        """Build a human-readable summary to read back to a verified caller."""
        lines = [
            f"Policy number {policy.policy_number}.",
            f"Product: {policy.product_name}.",
            f"Status: {policy.status.capitalize()}.",
            f"Commencement date: {policy.commencement_date}.",
            "",
            "Parties on this policy:",
        ]
        for party in policy.parties:
            lines.append(
                f"  {party.role.replace('_', ' ').title()}: "
                f"{party.first_name} {party.last_name}, "
                f"DOB {party.date_of_birth}, "
                f"{party.address_line1}, {party.city}, {party.postcode}."
            )
        return "\n".join(lines)
