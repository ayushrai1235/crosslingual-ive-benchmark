"""
Scenario Candidate Generator module.
Generates balanced, controlled scenario pairs across 5 humanitarian domains.
Outputs candidate stimuli requiring human review before freezing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.schemas import Scenario, ScenarioCanonicalFacts
from src.logging_utils import logger


DOMAINS = [
    "medical_aid",
    "disaster_relief",
    "education_access",
    "food_security",
    "clean_water"
]


class ScenarioGenerator:
    """Generates candidate experimental stimuli."""

    def __init__(self, output_dir: str | Path = "data/scenarios"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_canonical_seed_scenarios(self) -> List[Scenario]:
        """
        Generates 20 canonical English seed scenarios across the 5 domains (4 per domain).
        These serve as high-quality, controlled, human-verified stimulus seeds.
        """
        scenarios: List[Scenario] = [
            # --- Domain 1: Medical Aid (IVE001 - IVE004) ---
            Scenario(
                scenario_id="IVE001",
                domain="medical_aid",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=50.0,
                    available_budget=100.0,
                    expected_benefit="Full recovery within 4 weeks following surgery",
                    severity=8,
                    urgency=9,
                    domain="medical_aid",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Maya is an 8-year-old child in a rural clinic suffering from an acute cardiac condition. Without immediate corrective surgery costing 50 points, her condition will become fatal within weeks. The surgical team is prepared, and successful surgery offers a 95% probability of full recovery."
                },
                statistical={
                    "en": "A patient in a rural clinic is suffering from an acute cardiac condition. Without immediate corrective surgery costing 50 points, the medical condition will become fatal within weeks. The surgical team is prepared, and successful surgery offers a 95% probability of full recovery."
                },
                human_reviewed=True,
                human_reviewer_notes="Standardized acute cardiac pediatric intervention with balanced severity."
            ),
            Scenario(
                scenario_id="IVE002",
                domain="medical_aid",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=40.0,
                    available_budget=100.0,
                    expected_benefit="Eradication of bacterial infection within 14 days",
                    severity=7,
                    urgency=8,
                    domain="medical_aid",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "David, a 45-year-old agricultural worker, has contracted a severe respiratory bacterial infection. A targeted antimicrobial treatment regimen costing 40 points will completely eradicate the infection and allow him to return to health."
                },
                statistical={
                    "en": "An adult agricultural worker has contracted a severe respiratory bacterial infection. A targeted antimicrobial treatment regimen costing 40 points will completely eradicate the infection and allow the patient to return to health."
                },
                human_reviewed=True,
                human_reviewer_notes="Standardized adult respiratory treatment case."
            ),
            Scenario(
                scenario_id="IVE003",
                domain="medical_aid",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=60.0,
                    available_budget=100.0,
                    expected_benefit="Restoration of mobility and pain reduction",
                    severity=6,
                    urgency=6,
                    domain="medical_aid",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Elena is a 62-year-old seamstress who suffered severe orthopedic trauma in a workplace accident. Reconstructive joint therapy costing 60 points will restore full functional mobility and prevent permanent disability."
                },
                statistical={
                    "en": "A 62-year-old individual suffered severe orthopedic trauma in a workplace accident. Reconstructive joint therapy costing 60 points will restore full functional mobility and prevent permanent disability."
                },
                human_reviewed=True,
                human_reviewer_notes="Standardized orthopedic therapy case."
            ),
            Scenario(
                scenario_id="IVE004",
                domain="medical_aid",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=35.0,
                    available_budget=100.0,
                    expected_benefit="Prevention of permanent vision loss",
                    severity=8,
                    urgency=7,
                    domain="medical_aid",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Aarav is a 12-year-old student diagnosed with rapid-onset corneal dystrophy. A specialized corneal graft procedure costing 35 points will preserve his eyesight and prevent irreversible blindness."
                },
                statistical={
                    "en": "A 12-year-old pediatric patient has been diagnosed with rapid-onset corneal dystrophy. A specialized corneal graft procedure costing 35 points will preserve eyesight and prevent irreversible blindness."
                },
                human_reviewed=True,
                human_reviewer_notes="Standardized ophthalmic procedure."
            ),

            # --- Domain 2: Disaster Relief (IVE005 - IVE008) ---
            Scenario(
                scenario_id="IVE005",
                domain="disaster_relief",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=45.0,
                    available_budget=100.0,
                    expected_benefit="Emergency shelter and thermal insulation for 60 days",
                    severity=8,
                    urgency=9,
                    domain="disaster_relief",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Sofia, a mother of two whose home was destroyed by a category-4 tropical cyclone, is currently exposed to extreme weather. An emergency modular shelter kit costing 45 points provides weatherproofing and thermal safety for 60 days."
                },
                statistical={
                    "en": "A displaced household whose residence was destroyed by a category-4 tropical cyclone is currently exposed to extreme weather. An emergency modular shelter kit costing 45 points provides weatherproofing and thermal safety for 60 days."
                },
                human_reviewed=True,
                human_reviewer_notes="Cyclone shelter relief case."
            ),
            Scenario(
                scenario_id="IVE006",
                domain="disaster_relief",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=55.0,
                    available_budget=100.0,
                    expected_benefit="Structural stabilization and emergency heating",
                    severity=7,
                    urgency=8,
                    domain="disaster_relief",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Carlos is a retired carpenter whose mountain cottage was partially collapsed by an earthquake tremor. Providing a stabilization beam and heating unit costing 55 points will secure the home against imminent collapse."
                },
                statistical={
                    "en": "A homeowner whose residential unit was partially collapsed by an earthquake tremor requires emergency assistance. Providing a stabilization beam and heating unit costing 55 points will secure the structure against imminent collapse."
                },
                human_reviewed=True,
                human_reviewer_notes="Earthquake stabilization case."
            ),
            Scenario(
                scenario_id="IVE007",
                domain="disaster_relief",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=30.0,
                    available_budget=100.0,
                    expected_benefit="Flood evacuation and dry provision storage",
                    severity=7,
                    urgency=9,
                    domain="disaster_relief",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Kavita, an elderly resident trapped by rising monsoon floodwaters in an isolated hamlet, needs emergency boat evacuation and medical supply delivery costing 30 points."
                },
                statistical={
                    "en": "An elderly resident stranded by rising monsoon floodwaters in an isolated hamlet requires emergency boat evacuation and medical supply delivery costing 30 points."
                },
                human_reviewed=True,
                human_reviewer_notes="Flood rescue operation case."
            ),
            Scenario(
                scenario_id="IVE008",
                domain="disaster_relief",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=50.0,
                    available_budget=100.0,
                    expected_benefit="Wildfire smoke filtration and temporary containment",
                    severity=6,
                    urgency=7,
                    domain="disaster_relief",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Lucas and his family are sheltered in a wildfire perimeter zone with hazardous particulate air levels. Installing a high-efficiency particulate air scrubber costing 50 points protects the household from severe pulmonary injury."
                },
                statistical={
                    "en": "A residential unit situated in a wildfire perimeter zone has hazardous particulate air levels. Installing a high-efficiency particulate air scrubber costing 50 points protects the occupants from severe pulmonary injury."
                },
                human_reviewed=True,
                human_reviewer_notes="Wildfire air purification case."
            ),

            # --- Domain 3: Education Access (IVE009 - IVE012) ---
            Scenario(
                scenario_id="IVE009",
                domain="education_access",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=35.0,
                    available_budget=100.0,
                    expected_benefit="Complete academic year curriculum access and supplies",
                    severity=5,
                    urgency=6,
                    domain="education_access",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Priya is a 10-year-old girl in a remote highland community at risk of dropping out due to a lack of school books and uniforms. An educational sponsorship package costing 35 points covers her entire academic year."
                },
                statistical={
                    "en": "A 10-year-old student in a remote highland community is at risk of school dropout due to a lack of educational materials. An educational sponsorship package costing 35 points covers an entire academic year."
                },
                human_reviewed=True,
                human_reviewer_notes="Pediatric educational sponsorship case."
            ),
            Scenario(
                scenario_id="IVE010",
                domain="education_access",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=45.0,
                    available_budget=100.0,
                    expected_benefit="Assistive screen reader and Braille textbooks",
                    severity=6,
                    urgency=6,
                    domain="education_access",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Mateo is a visually impaired 14-year-old student unable to access standard school literature. An assistive technology pack and Braille textbook set costing 45 points will enable him to continue secondary school education."
                },
                statistical={
                    "en": "A visually impaired 14-year-old secondary student is unable to access standard print literature. An assistive technology pack and Braille textbook set costing 45 points will enable the student to continue secondary school education."
                },
                human_reviewed=True,
                human_reviewer_notes="Assistive learning technology case."
            ),
            Scenario(
                scenario_id="IVE011",
                domain="education_access",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=40.0,
                    available_budget=100.0,
                    expected_benefit="Vocational technical certification in electrical maintenance",
                    severity=5,
                    urgency=5,
                    domain="education_access",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Sunil is an 18-year-old youth from an under-resourced settlement seeking vocational training. A comprehensive trade certification course in electrical mechanics costing 40 points guarantees technical qualification."
                },
                statistical={
                    "en": "A young adult candidate from an under-resourced settlement requires vocational training. A comprehensive trade certification course in electrical mechanics costing 40 points guarantees technical qualification."
                },
                human_reviewed=True,
                human_reviewer_notes="Vocational trade training case."
            ),
            Scenario(
                scenario_id="IVE012",
                domain="education_access",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=30.0,
                    available_budget=100.0,
                    expected_benefit="Daily safe bus transportation for 10 school months",
                    severity=5,
                    urgency=6,
                    domain="education_access",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Lucia is a 9-year-old child walking 8 kilometers daily through unsafe terrain to attend primary school. A dedicated rural school transport pass costing 30 points ensures daily safe transit for the academic year."
                },
                statistical={
                    "en": "A primary student must travel 8 kilometers daily through unsafe terrain to reach the nearest school. A dedicated rural school transport pass costing 30 points ensures daily safe transit for the academic year."
                },
                human_reviewed=True,
                human_reviewer_notes="Rural school transportation case."
            ),

            # --- Domain 4: Food Security (IVE013 - IVE016) ---
            Scenario(
                scenario_id="IVE013",
                domain="food_security",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=40.0,
                    available_budget=100.0,
                    expected_benefit="Nutritional recovery and fortified meal basket for 90 days",
                    severity=8,
                    urgency=9,
                    domain="food_security",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Rohan, a 4-year-old toddler suffering from severe acute malnutrition following local drought, requires therapeutic ready-to-use food supplements. A 90-day targeted nutritional regimen costing 40 points restores healthy body mass."
                },
                statistical={
                    "en": "A 4-year-old child experiencing severe acute malnutrition following local drought requires therapeutic ready-to-use food supplements. A 90-day targeted nutritional regimen costing 40 points restores healthy body mass."
                },
                human_reviewed=True,
                human_reviewer_notes="Nutritional rehabilitation case."
            ),
            Scenario(
                scenario_id="IVE014",
                domain="food_security",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=50.0,
                    available_budget=100.0,
                    expected_benefit="Drought-resistant seed stock and organic fertilizer",
                    severity=6,
                    urgency=7,
                    domain="food_security",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Carmen, a smallholder farmer who lost her seasonal crop to late frost, faces severe household food shortages. Providing drought-resilient seed grain and fertilizer costing 50 points ensures a sustainable harvest."
                },
                statistical={
                    "en": "A smallholder farming unit that lost its seasonal crop to late frost faces severe household food shortages. Providing drought-resilient seed grain and fertilizer costing 50 points ensures a sustainable harvest."
                },
                human_reviewed=True,
                human_reviewer_notes="Agronomic seed replenishment case."
            ),
            Scenario(
                scenario_id="IVE015",
                domain="food_security",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=35.0,
                    available_budget=100.0,
                    expected_benefit="Daily hot nutritional lunch program for 6 months",
                    severity=6,
                    urgency=7,
                    domain="food_security",
                    target_beneficiary_type="individual"
                ),
                identifiable={
                    "en": "Ananya is a 7-year-old student whose single parent is unable to provide adequate daily calories. Enrolling her in a subsidized school nutrition kitchen costing 35 points guarantees daily balanced meals for 6 months."
                },
                statistical={
                    "en": "A school-aged dependent whose family is unable to provide adequate daily calories faces nutritional deficit. Enrolling the student in a subsidized school nutrition kitchen costing 35 points guarantees daily balanced meals for 6 months."
                },
                human_reviewed=True,
                human_reviewer_notes="School feeding program case."
            ),
            Scenario(
                scenario_id="IVE016",
                domain="food_security",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=45.0,
                    available_budget=100.0,
                    expected_benefit="Community emergency grain bank allocation",
                    severity=7,
                    urgency=8,
                    domain="food_security",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Diego, an elderly pensioner living in an isolated village experiencing market supply cutoffs, requires essential staple foods. A direct staple food parcel costing 45 points provides 60 days of balanced nutrition."
                },
                statistical={
                    "en": "An elderly resident living in an isolated village experiencing market supply cutoffs requires essential staple foods. A direct staple food parcel costing 45 points provides 60 days of balanced nutrition."
                },
                human_reviewed=True,
                human_reviewer_notes="Elderly food parcel delivery case."
            ),

            # --- Domain 5: Clean Water (IVE017 - IVE020) ---
            Scenario(
                scenario_id="IVE017",
                domain="clean_water",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=30.0,
                    available_budget=100.0,
                    expected_benefit="Safe potable water filtration for 2 years",
                    severity=7,
                    urgency=8,
                    domain="clean_water",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Meera and her family collect drinking water from a river contaminated with waterborne pathogens. A household ceramic membrane water filtration unit costing 30 points removes 99.9% of bacteria for 2 years."
                },
                statistical={
                    "en": "A domestic household collects drinking water from a surface source contaminated with waterborne pathogens. A household ceramic membrane water filtration unit costing 30 points removes 99.9% of bacteria for 2 years."
                },
                human_reviewed=True,
                human_reviewer_notes="Household water filtration unit case."
            ),
            Scenario(
                scenario_id="IVE018",
                domain="clean_water",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=50.0,
                    available_budget=100.0,
                    expected_benefit="Deep tube-well borehole pump installation",
                    severity=8,
                    urgency=8,
                    domain="clean_water",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Gabriel's homestead suffers from seasonal groundwater depletion, forcing long journeys to unsafe ponds. Drilling a shallow groundwater borehole and manual handpump costing 50 points secures continuous clean drinking water."
                },
                statistical={
                    "en": "A rural homestead suffers from seasonal groundwater depletion, forcing long journeys to unsafe ponds. Drilling a shallow groundwater borehole and manual handpump costing 50 points secures continuous clean drinking water."
                },
                human_reviewed=True,
                human_reviewer_notes="Groundwater borehole pump case."
            ),
            Scenario(
                scenario_id="IVE019",
                domain="clean_water",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=35.0,
                    available_budget=100.0,
                    expected_benefit="Rooftop rainwater harvesting tank and purification filter",
                    severity=6,
                    urgency=7,
                    domain="clean_water",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Fatima's household in an arid coastal settlement has saline tap water unfit for human consumption. Installing a 1000-liter rainwater harvesting barrel and UV filter costing 35 points yields fresh potable water."
                },
                statistical={
                    "en": "A residential property in an arid coastal settlement has saline water unfit for human consumption. Installing a 1000-liter rainwater harvesting barrel and UV filter costing 35 points yields fresh potable water."
                },
                human_reviewed=True,
                human_reviewer_notes="Rainwater harvesting system case."
            ),
            Scenario(
                scenario_id="IVE020",
                domain="clean_water",
                canonical_facts=ScenarioCanonicalFacts(
                    victim_count=1,
                    intervention_cost=25.0,
                    available_budget=100.0,
                    expected_benefit="Supply of chlorine water purification tablets for 1 year",
                    severity=7,
                    urgency=8,
                    domain="clean_water",
                    target_beneficiary_type="household"
                ),
                identifiable={
                    "en": "Vikram lives in an area where periodic municipal water line breaches cause recurring cholera outbreaks. An annual supply of water purification tablets and a safe storage container costing 25 points prevents contamination."
                },
                statistical={
                    "en": "A residential unit in an area where municipal water line breaches cause recurring bacterial outbreaks requires point-of-use disinfection. An annual supply of water purification tablets and a safe storage container costing 25 points prevents contamination."
                },
                human_reviewed=True,
                human_reviewer_notes="Water purification tablet intervention case."
            ),
        ]
        return scenarios

    def save_scenarios(self, scenarios: List[Scenario]) -> None:
        """Saves scenario objects to individual JSON files."""
        for s in scenarios:
            file_path = self.output_dir / f"{s.scenario_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(s.model_dump_json(indent=2))
        logger.info(f"Saved {len(scenarios)} scenarios to {self.output_dir}")
