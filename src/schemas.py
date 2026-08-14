"""
Pydantic schemas for the Cross-Lingual Identifiable Victim Effect (IVE) Benchmark.
Ensures strict type-safety, validation constraints, and serialization across the research pipeline.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ScenarioCanonicalFacts(BaseModel):
    """Canonical ground-truth facts for a scenario that remain constant across conditions."""
    victim_count: int = Field(..., ge=1, description="Number of victims/beneficiaries (must match across conditions).")
    intervention_cost: float = Field(..., gt=0.0, description="Cost required for full intervention in points.")
    available_budget: float = Field(default=100.0, gt=0.0, description="Total budget available to the allocator.")
    expected_benefit: str = Field(..., min_length=5, description="Expected outcome of intervention.")
    severity: int = Field(..., ge=1, le=10, description="Severity rating of the harm/illness (1-10).")
    urgency: int = Field(..., ge=1, le=10, description="Urgency rating for intervention (1-10).")
    domain: str = Field(..., description="Humanitarian domain (e.g., medical_aid, disaster_relief).")
    target_beneficiary_type: str = Field(default="individual", description="Beneficiary category.")


class Scenario(BaseModel):
    """Complete multi-lingual scenario stimulus entity with condition texts."""
    scenario_id: str = Field(..., pattern=r"^IVE\d{3,}$", description="Unique identifier (e.g., IVE001).")
    domain: str = Field(..., description="Humanitarian domain.")
    canonical_facts: ScenarioCanonicalFacts
    identifiable: Dict[str, str] = Field(..., description="Identifiable text keyed by language code (en, hi, es).")
    statistical: Dict[str, str] = Field(..., description="Statistical text keyed by language code (en, hi, es).")
    validation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Audit and validation flags.")
    human_reviewed: bool = Field(default=False, description="Flag indicating whether human validation occurred.")
    human_reviewer_notes: Optional[str] = Field(default=None, description="Notes from human reviewer.")

    @field_validator("identifiable", "statistical")
    @classmethod
    def validate_languages_present(cls, v: Dict[str, str]) -> Dict[str, str]:
        if "en" not in v:
            raise ValueError("English ('en') stimulus text must always be present as canonical baseline.")
        for lang, text in v.items():
            if not text or len(text.strip()) < 10:
                raise ValueError(f"Stimulus text for language '{lang}' must be at least 10 characters.")
        return v


class ScenarioAuditResult(BaseModel):
    """Result of automated validation on scenario candidate pairs."""
    scenario_id: str
    quantitative_match: bool = Field(..., description="Whether quantitative facts strictly match between conditions.")
    factual_equivalence_score: float = Field(..., ge=0.0, le=1.0)
    severity_match: bool
    urgency_match: bool
    is_neutral: bool
    approved: bool
    audit_notes: str = Field(default="")


class TranslationValidationResult(BaseModel):
    """Result of automated audit on translations."""
    scenario_id: str
    source_language: str
    target_language: str
    condition: Literal["identifiable", "statistical"]
    semantic_equivalence: float = Field(..., ge=0.0, le=1.0)
    quantitative_equivalence: float = Field(..., ge=0.0, le=1.0)
    severity_equivalence: float = Field(..., ge=0.0, le=1.0)
    emotional_equivalence: float = Field(..., ge=0.0, le=1.0)
    factual_equivalence: float = Field(..., ge=0.0, le=1.0)
    cultural_neutrality: float = Field(..., ge=0.0, le=1.0)
    approved: bool
    reasons: List[str] = Field(default_factory=list)


class HumanReviewRecord(BaseModel):
    """Record of human linguistic or scenario review."""
    item_id: str
    review_type: Literal["scenario", "translation"]
    reviewer_id: str
    approved: bool
    confidence: int = Field(default=5, ge=1, le=5)
    notes: Optional[str] = None
    timestamp: str


class DatasetManifest(BaseModel):
    """Immutable freeze manifest with SHA-256 hashes of approved stimuli."""
    manifest_version: str = "1.0.0"
    created_at: str
    frozen: bool = True
    file_hashes: Dict[str, str] = Field(..., description="Relative file path -> SHA-256 hex digest.")
    scenario_count: int
    languages: List[str]
    conditions: List[str]


class JudgmentRaw(BaseModel):
    """Raw empirical judgment record stored in streaming JSONL."""
    run_id: str
    scenario_id: str
    model_id: str
    model_family: str
    category: str
    language: str
    victim_condition: Literal["identifiable", "statistical"]
    prompt_hash: str
    raw_prompt: str
    raw_response: str
    parsed_allocation: Optional[float] = None
    parse_method: Literal["strict_json", "fenced_json", "labeled_field", "rejected"]
    temperature: float = 0.0
    seed: int = 42
    timestamp: str
    hardware: str = "cpu"
    quantization: str = "none"
    success: bool
    error_message: Optional[str] = None


class PairedScenarioJudgment(BaseModel):
    """Scenario-level paired calculation: IVE = Identifiable - Statistical."""
    model_id: str
    model_family: str
    category: str
    scenario_id: str
    language: str
    identifiable_allocation: float
    statistical_allocation: float
    ive: float = Field(..., description="Identifiable allocation minus statistical allocation.")


class LanguageControlItem(BaseModel):
    """Objective comprehension control question item."""
    item_id: str
    language: str
    category: Literal["basic_comprehension", "numerical_comprehension", "instruction_following", "factual_understanding", "factual_reasoning"]
    question: str
    expected_answer: str
    acceptable_variants: List[str] = Field(default_factory=list)


class LanguageControlResult(BaseModel):
    """Evaluation result of language control benchmark."""
    model_id: str
    item_id: str
    language: str
    category: str
    raw_response: str
    parsed_answer: str
    is_correct: bool
    timestamp: str
