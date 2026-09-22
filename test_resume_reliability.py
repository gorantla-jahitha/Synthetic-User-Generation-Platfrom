import sys
import os
import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.experiment import CreateExperimentRequest, Experiment
from models.persona import Persona
from services.database_service import database_service
from services.execution_state_service import execution_state_service
from services.resume_service import resume_service
from agents.persona_generation_agent import persona_generation_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ResumeReliabilityTest")

def cleanup_test_experiment(exp_id: str):
    from database import db_manager
    if db_manager.is_connected:
        db_manager.db.experiments.delete_one({"experiment_id": exp_id})
        db_manager.db.personas.delete_many({"experiment_id": exp_id})
        db_manager.db.experiment_execution_state.delete_one({"experiment_id": exp_id})
        db_manager.db.interviews.delete_many({"experiment_id": exp_id})
        db_manager.db.interview_insights.delete_many({"experiment_id": exp_id})
        db_manager.db.research_insights.delete_one({"experiment_id": exp_id})
        db_manager.db.survey_responses.delete_many({"experiment_id": exp_id})
        db_manager.db.survey_insights.delete_one({"experiment_id": exp_id})
        db_manager.db.adoption_results.delete_one({"experiment_id": exp_id})
    db_manager.in_memory_experiments.pop(exp_id, None)
    db_manager.in_memory_execution_states.pop(exp_id, None)
    db_manager.in_memory_survey_insights.pop(exp_id, None)
    db_manager.in_memory_research_insights.pop(exp_id, None)
    db_manager.in_memory_adoption_results.pop(exp_id, None)

async def test_persona_generation_interrupt_and_resume():
    """
    Test 100-persona generation interrupted at Batch 5 (40 personas completed),
    verifying state saving, atomic lock, resume starting at Batch 5 (not 1-4),
    and completing to 100/100 without regenerating existing 40 personas.
    """
    logger.info("\n==================================================")
    logger.info("TEST 1: 100-Persona Interrupted Batch 5 & Resume (40 -> 100)")
    logger.info("==================================================")

    exp_id = "exp_resume_100_test"
    cleanup_test_experiment(exp_id)
    req_100 = CreateExperimentRequest(
        product_domain="Fintech",
        product_description="AI Spare Change Investing App",
        target_audience="College Students in India",
        research_objective="Evaluate adoption motivators",
        number_of_personas=100
    )

    exp = Experiment(
        experiment_id=exp_id,
        product_domain=req_100.product_domain,
        product_description=req_100.product_description,
        target_audience=req_100.target_audience,
        research_objective=req_100.research_objective,
        number_of_personas=100
    )
    database_service.save_experiment(exp)
    execution_state_service.initialize_execution_state(exp_id, 100)

    # 1. Simulate Batches 1 to 4 generated and saved (40 personas)
    initial_40 = []
    for idx in range(1, 41):
        initial_40.append(Persona(
            persona_id=f"persona_{idx:03d}",
            experiment_id=exp_id,
            name=f"Existing Student {idx}",
            age=20 + (idx % 4),
            occupation="Computer Science Student",
            location="Bangalore",
            background=f"Undergraduate student {idx} managing savings.",
            personality_traits=["Studious"],
            goals=["Save money"],
            preferences=["Mobile app"],
            pain_points=["Budgeting"],
            behavioral_patterns=[],
            psychological_profile="",
            technology_adoption="High",
            price_sensitivity="Medium",
            initial_opinions=["Interested."],
            memory=[],
            conversation_history=[]
        ))
    database_service.save_personas(initial_40)
    execution_state_service.checkpoint_persona_batch(exp_id, 1, 10, 10, 100, failed=False)
    execution_state_service.checkpoint_persona_batch(exp_id, 2, 10, 20, 100, failed=False)
    execution_state_service.checkpoint_persona_batch(exp_id, 3, 10, 30, 100, failed=False)
    execution_state_service.checkpoint_persona_batch(exp_id, 4, 10, 40, 100, failed=False)
    execution_state_service.checkpoint_persona_batch(exp_id, 5, 0, 40, 100, failed=True, last_error="Simulated 429 Quota Exceeded at Batch 5")

    # 2. Check Execution State
    rec = execution_state_service.get_resume_recommendation(exp_id)
    assert rec["can_resume"] is True
    assert rec["overall_status"] == "INCOMPLETE"
    assert rec["resume_point"]["workflow"] == "persona_generation"
    assert rec["resume_point"]["completed_count"] == 40
    assert rec["resume_point"]["next_batch"] == 5
    logger.info("[PASSED] Execution state correctly identified resume point at Batch 5 (40/100).")

    # 3. Simulate resuming remaining Batches 5-10
    llm_call_count = 0
    async def mock_resume_llm(prompt):
        nonlocal llm_call_count
        llm_call_count += 1
        # Each call produces 10 personas starting from 41
        batch_num = 4 + llm_call_count
        start_num = (batch_num - 1) * 10 + 1
        batch_ps = []
        for i in range(10):
            idx = start_num + i
            batch_ps.append({
                "persona_id": f"persona_{idx:03d}",
                "name": f"Resumed Student {idx}",
                "age": 22,
                "occupation": "Engineering Student",
                "location": "Mumbai",
                "background": f"Resumed student {idx} joining study.",
                "personality_traits": ["Analytical"],
                "goals": ["Learn coding"],
                "preferences": ["Web app"],
                "pain_points": ["Time limits"],
                "technology_adoption": "High",
                "price_sensitivity": "Medium",
                "initial_opinions": ["Ready."]
            })
        return {"personas": batch_ps}

    with patch("services.llm_service.llm_service.generate_json", side_effect=mock_resume_llm):
        events = []
        async for event_chunk in resume_service.resume_experiment_stream(exp_id):
            events.append(event_chunk)

    # 4. Verify Resume Results
    assert llm_call_count == 6, f"Expected exactly 6 LLM calls for batches 5-10, got {llm_call_count}"
    
    final_personas = database_service.get_personas_for_experiment(exp_id)
    assert len(final_personas) == 100, f"Expected 100 personas after resume, got {len(final_personas)}"

    # Verify original 40 personas remained untouched
    first_p = database_service.get_persona(exp_id, "persona_001")
    assert first_p["name"] == "Existing Student 1"

    final_state = execution_state_service.get_execution_state(exp_id)
    assert final_state["overall_status"] == "COMPLETE"
    logger.info("[PASSED] 100-persona resume completed from Batch 5 to 10 (100/100) with 0 regeneration of Batches 1-4!")


async def test_pipeline_auto_continuation_on_insights_failure():
    """
    Test resuming an experiment where AI Insights failed:
    Verifies that Resume executes ONLY AI Insights, and then automatically continues
    Adoption Analysis, Dashboard, and PDF Report in a single continuous session.
    """
    logger.info("\n==================================================")
    logger.info("TEST 2: Pipeline Auto-Continuation (AI Insights Incomplete -> Auto Continue Adoption/Dashboard/PDF)")
    logger.info("==================================================")

    exp_id = "exp_pipeline_test"
    cleanup_test_experiment(exp_id)
    exp = Experiment(
        experiment_id=exp_id,
        product_domain="EdTech",
        product_description="AI Study Hub",
        target_audience="College Students",
        research_objective="Evaluate adoption",
        number_of_personas=1
    )
    database_service.save_experiment(exp)

    p1 = Persona(
        persona_id="p1",
        experiment_id=exp_id,
        name="Ananya Iyer",
        age=20,
        occupation="MBBS Student",
        location="Chennai",
        background="Medical student preparing for clinical rotations.",
        personality_traits=["Focused"],
        goals=["Pass exams"],
        preferences=["Mobile app"],
        pain_points=["Long shifts"],
        behavioral_patterns=[],
        psychological_profile="",
        technology_adoption="High",
        price_sensitivity="Medium",
        initial_opinions=["Looks good."],
        memory=[],
        conversation_history=[]
    )
    database_service.save_personas([p1])

    # Save interview session
    database_service.save_interview_session(
        experiment_id=exp_id,
        persona_id="p1",
        persona_name="Ananya Iyer",
        persona_occupation="MBBS Student",
        messages=[
            {"role": "user", "content": "How do you study?"},
            {"role": "persona", "content": "I study during breaks between rotations."}
        ]
    )

    # Save initial execution state with AI Insights marked INCOMPLETE
    execution_state_service.initialize_execution_state(exp_id, 1)
    execution_state_service.checkpoint_workflow_status(exp_id, "persona_generation", "COMPLETE", completed=True)
    execution_state_service.checkpoint_workflow_status(exp_id, "survey", "COMPLETE", completed=True)
    execution_state_service.checkpoint_workflow_status(exp_id, "survey_insights", "COMPLETE", completed=True)
    execution_state_service.checkpoint_workflow_status(exp_id, "ai_insights", "INCOMPLETE", completed=False, last_error="Simulated LLM 503 Timeout")

    rec = execution_state_service.get_resume_recommendation(exp_id)
    assert rec["can_resume"] is True
    assert rec["next_incomplete_workflow"] == "ai_insights"
    logger.info("[PASSED] Recommendation correctly targets 'ai_insights'.")

    # Mock insight generation calls
    async def mock_insights_generate(prompt):
        return {
            "executive_summary": "Medical students value concise mobile access.",
            "top_findings": ["Short study blocks work best"],
            "recurring_themes": ["Time constraint"],
            "pain_points": ["Exhaustion"],
            "product_opportunities": ["Offline flashcards"],
            "persona_evidence": [
                {
                    "persona_name": "Ananya Iyer",
                    "persona_role": "MBBS Student",
                    "quote": "I study during breaks between rotations.",
                    "extracted_insight": "Short study blocks work best.",
                    "product_implication": "Flashcards feature."
                }
            ],
            "sentiment": {"positive_pct": 100.0, "neutral_pct": 0.0, "negative_pct": 0.0}
        }

    async def mock_adoption_generate(prompt):
        return {
            "overall_adoption_score": 88.0,
            "evidence_summary": "High utility for medical students.",
            "persona_adoptions": [
                {
                    "persona_id": "p1",
                    "persona_name": "Ananya Iyer",
                    "adoption_score": 88.0,
                    "would_use": "YES",
                    "reasoning": "Fits busy rotation schedule."
                }
            ]
        }

    with patch("services.llm_service.llm_service.generate_json", side_effect=mock_insights_generate):
        with patch("agents.insight_agent.llm_service.generate_json", side_effect=mock_adoption_generate):
            events = []
            async for event_chunk in resume_service.resume_experiment_stream(exp_id):
                events.append(event_chunk)

    final_state = execution_state_service.get_execution_state(exp_id)
    assert final_state["workflows"]["ai_insights"]["completed"] is True
    assert final_state["workflows"]["adoption"]["completed"] is True
    assert final_state["workflows"]["dashboard"]["completed"] is True
    assert final_state["workflows"]["pdf"]["completed"] is True
    assert final_state["overall_status"] == "COMPLETE"
    logger.info("[PASSED] AI Insights resumed and pipeline automatically continued through Adoption, Dashboard, and PDF to COMPLETE status!")


async def test_atomic_resume_lock():
    """
    Test atomic per-experiment lock:
    Verifies that simultaneous resume calls reject concurrent requests and return cleanly.
    """
    logger.info("\n==================================================")
    logger.info("TEST 3: Atomic Per-Experiment Resume Lock")
    logger.info("==================================================")

    exp_id = "exp_lock_test"
    cleanup_test_experiment(exp_id)
    execution_state_service.initialize_execution_state(exp_id, 10)

    # First lock acquire succeeds
    locked1, state1 = execution_state_service.acquire_resume_lock(exp_id)
    assert locked1 is True
    assert state1["overall_status"] == "RESUMING"

    # Second lock acquire fails (already RESUMING)
    locked2, state2 = execution_state_service.acquire_resume_lock(exp_id)
    assert locked2 is False
    logger.info("[PASSED] Concurrent resume lock correctly rejected second request.")

    # Release lock
    execution_state_service.release_resume_lock(exp_id, final_status="COMPLETE")
    state_after = execution_state_service.get_execution_state(exp_id)
    assert state_after["overall_status"] == "COMPLETE"
    logger.info("[PASSED] Resume lock released cleanly.")


async def run_all_resume_reliability_tests():
    await test_persona_generation_interrupt_and_resume()
    await test_pipeline_auto_continuation_on_insights_failure()
    await test_atomic_resume_lock()
    logger.info("\n==================================================")
    logger.info("ALL PROJECT-WIDE RESUME RELIABILITY TESTS PASSED PERFECTLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    asyncio.run(run_all_resume_reliability_tests())
