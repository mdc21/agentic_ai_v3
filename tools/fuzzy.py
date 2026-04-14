"""
Fuzzy match verification — compares caller-supplied identity data against SOR values.

Uses rapidfuzz for Jaro-Winkler string similarity plus a normalisation layer that:
  - expands common address abbreviations (St → Street, Rd → Road, etc.)
  - handles different word orderings via token_sort_ratio
  - does exact match (after normalisation) for postcodes and DOBs
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

# ── Abbreviation expansion map ────────────────────────────────────────────────

ADDR_ABBREVS: dict[str, str] = {
    # Street types
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "av": "avenue", "av.": "avenue",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "blvd": "boulevard",
    "ct": "court", "ct.": "court",
    "pl": "place", "pl.": "place",
    "terr": "terrace", "ter": "terrace", "tce": "terrace",
    "cres": "crescent", "crs": "crescent",
    "gdns": "gardens", "gdn": "garden",
    "sq": "square",
    "cl": "close",
    "pk": "park",
    "gr": "grove",
    # Directions
    "n": "north", "s": "south", "e": "east", "w": "west",
    "n.": "north", "s.": "south", "e.": "east", "w.": "west",
    # Numbers/ordinals common in UK addresses
    "1st": "first", "2nd": "second", "3rd": "third",
}

# ── Field-level thresholds ─────────────────────────────────────────────────────

THRESHOLDS: dict[str, float] = {
    # Policyholder fields
    "first_name": 85.0,
    "last_name":  85.0,
    "address_line1": 78.0,   # lower because abbreviations are common
    "postcode": 95.0,         # near-exact; only tolerate spacing differences
    "date_of_birth": 100.0,  # exact after normalisation
    # Financial adviser fields (compared against SoR)
    "adviser_firm_name": 80.0,    # firm names may have Ltd/Limited variations
    "adviser_address_line1": 75.0, # slightly lower due to abbreviations
    "adviser_postcode": 95.0,      # near-exact
}


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _expand_abbreviations(text: str) -> str:
    tokens = re.split(r"[\s,]+", text)
    return " ".join(ADDR_ABBREVS.get(t.lower().rstrip("."), t.lower().rstrip(".")) for t in tokens)


def _normalise_name(name: str) -> str:
    return name.lower().strip()


def _normalise_address(addr: str) -> str:
    return _expand_abbreviations(addr.lower().strip())


def _normalise_postcode(postcode: str) -> str:
    return re.sub(r"\s+", "", postcode).upper()


def _normalise_dob(dob_str: str) -> Optional[str]:
    """
    Parse a DOB string in any common format and return YYYY-MM-DD.
    Returns None on failure.
    """
    dob_str = dob_str.strip()
    # Direct ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", dob_str):
        return dob_str
    try:
        dt = dateutil_parser.parse(dob_str, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


# ── Per-field comparison ───────────────────────────────────────────────────────

@dataclass
class FieldResult:
    field: str
    caller_value: str
    sor_value: str
    normalised_caller: str
    normalised_sor: str
    score: float
    threshold: float
    passed: bool


def _compare_field(field: str, caller_val: str, sor_val: str) -> FieldResult:
    threshold = THRESHOLDS.get(field, 85.0)

    if field == "date_of_birth":
        nc = _normalise_dob(caller_val) or caller_val
        ns = _normalise_dob(sor_val) or sor_val
        passed = nc == ns
        score = 100.0 if passed else 0.0

    elif field == "postcode":
        nc = _normalise_postcode(caller_val)
        ns = _normalise_postcode(sor_val)
        score = fuzz.ratio(nc, ns)
        passed = score >= threshold

    elif field == "address_line1":
        nc = _normalise_address(caller_val)
        ns = _normalise_address(sor_val)
        # token_sort handles "14 High Street" vs "High Street 14"
        score = fuzz.token_sort_ratio(nc, ns)
        passed = score >= threshold

    else:  # first_name, last_name
        nc = _normalise_name(caller_val)
        ns = _normalise_name(sor_val)
        # Use WRatio which combines multiple fuzzy strategies
        score = fuzz.WRatio(nc, ns)
        passed = score >= threshold

    return FieldResult(
        field=field,
        caller_value=caller_val,
        sor_value=sor_val,
        normalised_caller=nc,
        normalised_sor=ns,
        score=score,
        threshold=threshold,
        passed=passed,
    )


# ── Full verification ──────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    verified: bool
    results: dict[str, FieldResult]
    failed_fields: list[str]

    def summary(self) -> str:
        lines = [f"Verification {'PASSED' if self.verified else 'FAILED'}"]
        for name, r in self.results.items():
            status = "✓" if r.passed else "✗"
            lines.append(
                f"  {status} {name}: score={r.score:.0f}% (threshold={r.threshold:.0f}%)"
                f"  caller={r.normalised_caller!r}  sor={r.normalised_sor!r}"
            )
        return "\n".join(lines)


def verify_caller(caller_data: dict, sor_data: dict) -> VerificationResult:
    """
    Compare caller-provided identity fields against SOR values.

    caller_data / sor_data keys:
        first_name, last_name, address_line1, postcode, date_of_birth
    """
    fields = list(THRESHOLDS.keys())
    results: dict[str, FieldResult] = {}

    for f in fields:
        caller_val = caller_data.get(f)
        sor_val = sor_data.get(f)
        if not caller_val or not sor_val:
            logger.debug("Skipping field %r — missing value (caller=%r, sor=%r)", f, caller_val, sor_val)
            continue
        results[f] = _compare_field(f, str(caller_val), str(sor_val))

    failed = [f for f, r in results.items() if not r.passed]
    verified = len(results) > 0 and len(failed) == 0

    result = VerificationResult(verified=verified, results=results, failed_fields=failed)
    logger.info(result.summary())
    return result
