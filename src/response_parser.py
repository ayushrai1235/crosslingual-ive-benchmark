"""
Deterministic 3-tier response parser for moral resource allocation judgments.
Strictly parses numeric allocations while rejecting ambiguous or invalid responses.
"""

import json
import re
from typing import Tuple, Optional, Literal

ParseMethod = Literal["strict_json", "fenced_json", "labeled_field", "rejected"]


class ResponseParser:
    """
    3-Tier deterministic parser:
    1. Tier 1: Direct JSON parsing
    2. Tier 2: Fenced JSON or embedded JSON object
    3. Tier 3: Labeled field regex extraction
    """

    def __init__(self, min_allocation: float = 0.0, max_allocation: float = 100.0):
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

    def parse(self, raw_response: str) -> Tuple[Optional[float], ParseMethod, Optional[str]]:
        """
        Parses a raw LLM response into a float allocation.
        Returns: (parsed_allocation, parse_method, rejection_reason)
        """
        if not raw_response or not raw_response.strip():
            return None, "rejected", "Empty or whitespace-only response"

        cleaned = raw_response.strip()

        # Check for ambiguous range patterns (e.g. "between 20 and 40", "30-50 points", "20 to 30")
        range_pattern = r"(?:between\s+\d+\s+(?:and|to)\s+\d+|\d+\s*[-–—]\s*\d+\s*(?:points|%|percent)?)"
        if re.search(range_pattern, cleaned, re.IGNORECASE) and not cleaned.startswith("{"):
            # Check if it's not a valid single JSON
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and "allocation" in data and isinstance(data["allocation"], (int, float)):
                    pass # It is valid json, will be parsed below
                else:
                    return None, "rejected", "Ambiguous range detected in response"
            except Exception:
                return None, "rejected", "Ambiguous range detected in response"

        # Tier 1: Direct JSON parse
        try:
            data = json.loads(cleaned)
            val, err = self._extract_value_from_dict(data)
            if val is not None:
                return val, "strict_json", None
            if err:
                return None, "rejected", f"Tier 1 error: {err}"
        except json.JSONDecodeError:
            pass

        # Tier 2: Fenced markdown or embedded JSON
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fenced_match:
            try:
                data = json.loads(fenced_match.group(1))
                val, err = self._extract_value_from_dict(data)
                if val is not None:
                    return val, "fenced_json", None
            except json.JSONDecodeError:
                pass

        embedded_json = re.search(r"(\{\s*\"allocation\"\s*:\s*[-+]?[0-9]*\.?[0-9]+\s*\})", cleaned, re.IGNORECASE)
        if embedded_json:
            try:
                data = json.loads(embedded_json.group(1))
                val, err = self._extract_value_from_dict(data)
                if val is not None:
                    return val, "fenced_json", None
            except json.JSONDecodeError:
                pass

        # Tier 3: Labeled field extraction
        labeled_pattern = r"(?:allocation|points|allotment|score)\s*(?::|=|\bis\b)\s*([0-9]+(?:\.[0-9]+)?)"
        matches = re.findall(labeled_pattern, cleaned, re.IGNORECASE)
        if len(matches) == 1:
            try:
                num = float(matches[0])
                if self.min_allocation <= num <= self.max_allocation:
                    return num, "labeled_field", None
                return None, "rejected", f"Allocation {num} out of bounds [{self.min_allocation}, {self.max_allocation}]"
            except ValueError:
                pass
        elif len(matches) > 1:
            # If multiple numbers found, check if they are identical
            unique_vals = {float(m) for m in matches}
            if len(unique_vals) == 1:
                num = unique_vals.pop()
                if self.min_allocation <= num <= self.max_allocation:
                    return num, "labeled_field", None
            return None, "rejected", f"Multiple conflicting allocations found: {matches}"

        return None, "rejected", "Could not deterministically extract valid numeric allocation"

    def _extract_value_from_dict(self, data: dict) -> Tuple[Optional[float], Optional[str]]:
        if not isinstance(data, dict):
            return None, "Parsed JSON root is not an object"

        # Case-insensitive lookup for 'allocation' or 'points'
        keys = {k.lower(): k for k in data.keys()}
        target_key = keys.get("allocation") or keys.get("points") or keys.get("allocated_points")
        if not target_key:
            return None, "Key 'allocation' not found in JSON"

        val = data[target_key]
        if isinstance(val, (int, float)):
            num = float(val)
            if self.min_allocation <= num <= self.max_allocation:
                return num, None
            return None, f"Allocation {num} is outside bounds [{self.min_allocation}, {self.max_allocation}]"
        elif isinstance(val, str):
            # Try to convert clean string number (e.g. "50" or "50.0")
            cleaned_val = re.sub(r"[^\d.]", "", val)
            try:
                num = float(cleaned_val)
                if self.min_allocation <= num <= self.max_allocation:
                    return num, None
                return None, f"Allocation {num} is outside bounds [{self.min_allocation}, {self.max_allocation}]"
            except ValueError:
                return None, f"String value '{val}' cannot be parsed to float"

        return None, f"Value {val} has invalid type {type(val).__name__}"
