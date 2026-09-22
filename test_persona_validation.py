import asyncio
import logging
import re
import sys
from typing import Dict, Any

from models.experiment import CreateExperimentRequest
from agents.persona_generation_agent import persona_generation_agent
from services.llm_service import llm_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PersonaValidationTest")

def test_placeholder_name_rejection():
    logger.info("=== Testing Placeholder Name Rejection ===")
    seen = set()
    placeholders = [
        "Customer 1", "Customer 2", "User 1", "User 2", "Persona 1", "Persona 001",
        "Customer", "User", "Persona", "Unknown", "Anonymous", "Test"
    ]
    for p in placeholders:
        payload = {"name": p, "age": 20, "occupation": "College Student"}
        valid, reason = persona_generation_agent.validate_persona(payload, "College students", seen)
        assert not valid, f"Expected placeholder '{p}' to be rejected, but passed: {reason}"
        logger.info(f"Successfully rejected placeholder name '{p}': {reason}")

def test_duplicate_name_rejection():
    logger.info("=== Testing Duplicate Name Rejection ===")
    seen = {"aarav mehta"}
    payload = {"name": "Aarav Mehta", "age": 20, "occupation": "College Student"}
    valid, reason = persona_generation_agent.validate_persona(payload, "College students", seen)
    assert not valid, "Expected duplicate name 'Aarav Mehta' to be rejected."
    logger.info(f"Successfully rejected duplicate name: {reason}")

def test_target_audience_student_enforcement():
    logger.info("=== Testing Target Audience Enforcement (College Students) ===")
    target_audience = "College students in India"

    # Valid college student
    student_payload = {
        "name": "Kavya Sharma",
        "age": 20,
        "occupation": "B.Tech Computer Science Student",
        "background": "Third-year engineering student at IIT Delhi."
    }
    v1, r1 = persona_generation_agent.validate_persona(student_payload, target_audience, set())
    assert v1, f"Valid student failed validation: {r1}"
    logger.info(f"Valid student passed: {student_payload['name']}")

    # Invalid: 45 year old senior software manager
    manager_payload = {
        "name": "Rajesh Kumar",
        "age": 45,
        "occupation": "Senior Software Manager",
        "background": "Leads engineering team with 15 years experience."
    }
    v2, r2 = persona_generation_agent.validate_persona(manager_payload, target_audience, set())
    assert not v2, "Expected 45yo Senior Manager to fail target audience validation for college students."
    logger.info(f"Successfully rejected invalid 45yo manager: {r2}")

    # Invalid: Retired person
    retiree_payload = {
        "name": "Suresh Patel",
        "age": 68,
        "occupation": "Retired Banker",
        "background": "Spent 40 years in retail banking."
    }
    v3, r3 = persona_generation_agent.validate_persona(retiree_payload, target_audience, set())
    assert not v3, "Expected retiree to fail target audience validation for college students."
    logger.info(f"Successfully rejected retiree for student prompt: {r3}")

def test_target_audience_age_bracket_enforcement():
    logger.info("=== Testing Target Audience Age Bracket Enforcement ===")
    ta_prof = "Software professionals aged 22–35 in India"
    ta_retired = "Retired people aged 60–75"

    # Out of range for 22-35
    old_prof = {"name": "Deepak Verma", "age": 48, "occupation": "Software Architect"}
    v1, r1 = persona_generation_agent.validate_persona(old_prof, ta_prof, set())
    assert not v1, f"Expected age 48 to fail for 22-35 bracket: {r1}"
    logger.info(f"Successfully rejected age 48 for 22-35 bracket: {r1}")

    # Out of range for 60-75
    young_retiree = {"name": "Ananya Joshi", "age": 30, "occupation": "Retired"}
    v2, r2 = persona_generation_agent.validate_persona(young_retiree, ta_retired, set())
    assert not v2, f"Expected age 30 to fail for 60-75 bracket: {r2}"
    logger.info(f"Successfully rejected age 30 for 60-75 bracket: {r2}")

def test_20_college_students_generation():
    logger.info("=== Testing 20 College Students Generation (Mocked Gemini Response Validation) ===")
    
    # 20 distinct Indian college student candidates
    names = [
        "Aaditya Roy", "Ishaan Verma", "Diya Kapoor", "Kabir Sengupta", "Tanya Reddy",
        "Arjun Deshmukh", "Sneha Kulkarni", "Devansh Pillai", "Riya Chatterji", "Yash Malhotra",
        "Kriti Agarwal", "Manav Shah", "Anika Choudhury", "Dhruv Bhatia", "Pooja Hegde",
        "Siddharth Menon", "Bhavya Trivedi", "Gaurav Tripathi", "Nidhi Saxena", "Varun Iyer"
    ]
    student_branches = [
        "Computer Science Student", "Mechanical Engineering Student", "Medical Student (MBBS)",
        "Economics Student", "Arts & Literature Student", "Commerce Student", "Biotechnology Student",
        "Civil Engineering Student", "Law Student", "Psychology Student", "Physics Honours Student",
        "Architecture Student", "Design Student", "Mathematics Student", "Journalism Student",
        "Management Student (BBA)", "Data Science Student", "Chemistry Student", "Electrical Engineering Student",
        "Political Science Student"
    ]

    mock_personas_list = []
    for i in range(20):
        mock_personas_list.append({
            "persona_id": f"persona_{i+1:03d}",
            "name": names[i],
            "age": 19 + (i % 5),
            "occupation": student_branches[i],
            "location": "Delhi" if i % 2 == 0 else "Bengaluru",
            "background": f"{names[i]} is a {student_branches[i]} passionate about study optimization.",
            "personality_traits": ["Studious", "Curious"],
            "goals": ["Ace exams", "Manage time"],
            "preferences": ["AI study tools", "Mobile planner"],
            "pain_points": ["Procrastination", "Information overload"],
            "behavioral_patterns": ["Schedules daily study sessions"],
            "psychological_profile": "Goal-driven student.",
            "technology_adoption": "High" if i % 2 == 0 else "Medium",
            "price_sensitivity": "High" if i % 3 == 0 else "Medium",
            "initial_opinions": ["Excited for AI study planner"]
        })

    # Validate all 20 personas
    seen = set()
    validated_personas = []
    for p_raw in mock_personas_list:
        v, r = persona_generation_agent.validate_persona(p_raw, "College students in India", seen)
        assert v, f"Candidate {p_raw['name']} failed: {r}"
        seen.add(p_raw["name"].lower())
        validated_personas.append(p_raw)

    assert len(validated_personas) == 20
    assert len(seen) == 20
    
    # Verify no invalid careers or placeholder names exist
    for p in validated_personas:
        name_lower = p["name"].lower()
        occ_lower = p["occupation"].lower()
        assert "customer" not in name_lower
        assert "professional" not in name_lower
        assert any(k in occ_lower for k in ["student", "mbbs", "bba"])
        assert p["age"] <= 35

    logger.info("Successfully validated 20/20 college student personas with 20 unique realistic human names!")

def run_all_unit_tests():
    test_placeholder_name_rejection()
    test_duplicate_name_rejection()
    test_target_audience_student_enforcement()
    test_target_audience_age_bracket_enforcement()
    test_20_college_students_generation()
    print("\n[SUCCESS] ALL LOCAL VALIDATION UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_unit_tests()

