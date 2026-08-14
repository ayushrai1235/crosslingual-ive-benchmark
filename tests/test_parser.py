"""
Unit tests for the 3-tier deterministic response parser.
Tests strict JSON, fenced JSON, labeled fields, ambiguity rejection, and boundary checks.
"""

from src.response_parser import ResponseParser


def test_tier1_strict_json():
    parser = ResponseParser(max_allocation=100.0)
    raw = '{"allocation": 65.5, "reasoning": "Standard allocation"}'
    alloc, method, notes = parser.parse(raw)
    assert alloc == 65.5
    assert method == "strict_json"
    assert notes is None


def test_tier2_fenced_markdown_json():
    parser = ResponseParser(max_allocation=100.0)
    raw = (
        "Here is my decision on the moral dilemma:\n\n"
        "```json\n"
        "{\n"
        '  "allocation": 80.0,\n'
        '  "confidence": 5,\n'
        '  "reasoning": "Urgent life saving need"\n'
        "}\n"
        "```\n"
        "I hope this helps."
    )
    alloc, method, notes = parser.parse(raw)
    assert alloc == 80.0
    assert method == "fenced_json"
    assert notes is None


def test_tier3_labeled_field():
    parser = ResponseParser(max_allocation=100.0)
    raw = (
        "Decision Report:\n"
        "Allocation: 55 points\n"
        "Reasoning: We must prioritize direct intervention."
    )
    alloc, method, notes = parser.parse(raw)
    assert alloc == 55.0
    assert method == "labeled_field"
    assert notes is None


def test_out_of_bounds_rejection():
    parser = ResponseParser(max_allocation=100.0)

    # Exceeds max budget
    raw_high = '{"allocation": 125.0}'
    alloc, method, notes = parser.parse(raw_high)
    assert alloc is None
    assert method == "rejected"
    assert "outside bounds" in notes or "error" in notes

    # Below 0
    raw_neg = '{"allocation": -10.0}'
    alloc, method, notes = parser.parse(raw_neg)
    assert alloc is None
    assert method == "rejected"


def test_ambiguous_range_rejection():
    parser = ResponseParser(max_allocation=100.0)
    # Range output must be rejected, not averaged
    raw_range = "I recommend an allocation between 40 and 60 points depending on severity."
    alloc, method, notes = parser.parse(raw_range)
    assert alloc is None
    assert method == "rejected"
    assert "Ambiguous" in notes or "range" in notes.lower()


def test_multiple_conflicting_numbers_rejection():
    parser = ResponseParser(max_allocation=100.0)
    raw_mult = "You could give allocation: 30 or alternatively allocation: 70."
    alloc, method, notes = parser.parse(raw_mult)
    assert alloc is None
    assert method == "rejected"
    assert "Multiple" in notes or "conflicting" in notes.lower()
