"""
Unit tests for Model Registry, 9-model panel configuration, and dual Causal/Seq2Seq runners.
"""

import pytest
from src.model_registry import ModelRegistry
from src.model_runner import BaseModelRunner, CausalLMRunner, Seq2SeqRunner, MockModelRunner, get_model_runner


def test_registry_contains_exact_nine_models():
    registry = ModelRegistry()
    models = registry.list_models(enabled_only=False)
    assert len(models) == 9

    expected_ids = {
        "llama_3_1_8b",
        "qwen3_8b",
        "qwen_2_5_7b",
        "gemma_3_4b",
        "gemma_3_12b",
        "aya_expanse_8b",
        "command_r7b",
        "bloomz_7b1_mt",
        "mt0_xl"
    }
    actual_ids = {m.id for m in models}
    assert actual_ids == expected_ids


def test_seven_distinct_model_families():
    registry = ModelRegistry()
    models = registry.list_models(enabled_only=False)
    distinct_families = {m.family.lower() for m in models}
    expected_families = {"llama", "qwen", "gemma", "aya", "command", "bloomz", "mt0"}
    assert distinct_families == expected_families
    assert len(distinct_families) == 7


def test_model_language_fields_separation():
    registry = ModelRegistry()
    models = registry.list_models(enabled_only=False)

    for m in models:
        # Check separate fields exist and are non-empty
        assert isinstance(m.officially_documented_languages, list)
        assert len(m.officially_documented_languages) > 0
        assert m.experiment_languages == ["en", "hi", "es"]
        assert isinstance(m.language_support_notes, dict)
        assert "en" in m.language_support_notes
        assert "hi" in m.language_support_notes
        assert "es" in m.language_support_notes


def test_qwen3_thinking_disabled_setting():
    registry = ModelRegistry()
    qwen3 = registry.get_model("qwen3_8b")
    assert qwen3.disable_thinking is True
    assert qwen3.category == "general_reasoning_multilingual"


def test_mt0_seq2seq_architecture():
    registry = ModelRegistry()
    mt0 = registry.get_model("mt0_xl")
    assert mt0.architecture == "seq2seq"
    assert mt0.family == "mT0"


def test_runner_factory_routing():
    registry = ModelRegistry()

    # Mock runner
    m_llama = registry.get_model("llama_3_1_8b")
    mock_runner = get_model_runner(m_llama, use_mock=True)
    assert isinstance(mock_runner, MockModelRunner)
    assert mock_runner.is_mock is True

    # Live runner factory check (instantiation without load)
    causal_runner = get_model_runner(m_llama, use_mock=False)
    assert isinstance(causal_runner, CausalLMRunner)

    m_mt0 = registry.get_model("mt0_xl")
    seq2seq_runner = get_model_runner(m_mt0, use_mock=False)
    assert isinstance(seq2seq_runner, Seq2SeqRunner)


def test_mock_runner_deterministic_generations():
    registry = ModelRegistry()
    m = registry.get_model("llama_3_1_8b")
    runner = MockModelRunner(m)

    ident_resp = runner.generate("Scenario about identifiable child Maya needing medical treatment.")
    stat_resp = runner.generate("Scenario about statistical population of 100 affected patients.")

    assert "60" in ident_resp
    assert "45" in stat_resp
