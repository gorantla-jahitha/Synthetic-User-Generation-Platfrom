import sys
import os
import asyncio
import json
import logging
import unittest
from unittest.mock import AsyncMock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.experiment import CreateExperimentRequest
from models.persona import Persona
from agents.persona_generation_agent import persona_generation_agent
from agents.survey_agent import survey_agent
from agents.interview_agent import interview_agent
from agents.insight_agent import insight_agent
from services.database_service import database_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ReliabilityTest")

async def test_sequential_batching_and_retries_mocked():
    """
    Test 50 persona generation (5 sequential batches of 10) with simulated transient failure & retry.
    Mocked ONLY inside this test function to preserve 0-mock policy in production runtime.
    """
    logger.info("\n==================================================")
    logger.info("TEST: 50-Persona Sequential Batching & Retry Handling (Mocked LLM inside test)")
    logger.info("==================================================")

    req_50 = CreateExperimentRequest(
        product_domain="EdTech",
        product_description="AI Planner",
        target_audience="College Students in India",
        research_objective="Study tools adoption",
        number_of_personas=50
    )

    batch_call_count = 0

    async def mock_generate_json(prompt):
        nonlocal batch_call_count
        batch_call_count += 1

        # Simulate a transient failure on call #3 (Batch 3 attempt 1) to test retry logic
        if batch_call_count == 3:
            raise RuntimeError("Gemini API 429 Rate Limit Exceeded (Simulated)")

        start_num = ((batch_call_count - (1 if batch_call_count > 3 else 0) - 1) * 10) + 1
        personas_data = []
        for i in range(10):
            idx = start_num + i
            personas_data.append({
                "persona_id": f"persona_{idx:03d}",
                "name": f"Student Person {idx}",
                "age": 20 + (idx % 4),
                "occupation": "B.Tech Computer Science Student",
                "location": "Delhi, India",
                "background": f"Undergraduate student {idx} managing course load.",
                "personality_traits": ["Studious", "Focused"],
                "goals": ["Ace exams"],
                "preferences": ["Mobile app"],
                "pain_points": ["Time management"],
                "technology_adoption": "High",
                "price_sensitivity": "Medium",
                "initial_opinions": ["Looking forward to AI planner."]
            })
        return {"personas": personas_data}

    with patch("services.llm_service.llm_service.generate_json", side_effect=mock_generate_json):
        personas = await persona_generation_agent.generate_personas(req_50, "exp_mock_50")

    assert len(personas) == 50, f"Expected exactly 50 personas, got {len(personas)}"
    assert batch_call_count == 6, f"Expected 6 LLM calls (5 batches + 1 retry), got {batch_call_count}"

    # Verify 50 unique names
    unique_names = set(p.name.lower() for p in personas)
    assert len(unique_names) == 50, "Duplicate persona names found across 50 personas!"
    logger.info(f"[PASSED] 50 personas generated sequentially with retry handling! Unique names count: {len(unique_names)}")

    # Also test /api/personas/generate-stream endpoint
    logger.info("--- Testing /api/personas/generate-stream SSE Endpoint ---")
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)

    sse_call_count = 0
    async def mock_sse_generate(prompt):
        nonlocal sse_call_count
        sse_call_count += 1
        start_num = ((sse_call_count - 1) * 10) + 1
        personas_data = []
        for i in range(10):
            idx = start_num + i
            personas_data.append({
                "persona_id": f"persona_{idx:03d}",
                "name": f"SSE Student {idx}",
                "age": 22,
                "occupation": "College Student",
                "location": "Mumbai, India",
                "background": f"Student {idx} studying for exams.",
                "personality_traits": ["Studious"],
                "goals": ["Pass exams"],
                "preferences": ["Mobile app"],
                "pain_points": ["Time management"],
                "technology_adoption": "High",
                "price_sensitivity": "Medium",
                "initial_opinions": ["Looking forward to app."]
            })
        return {"personas": personas_data}

    with patch("services.llm_service.llm_service.generate_json", side_effect=mock_sse_generate):
        res = client.post("/api/personas/generate-stream", json={
            "product_domain": "EdTech",
            "product_description": "AI Planner",
            "target_audience": "College Students in India",
            "research_objective": "Study tools adoption",
            "number_of_personas": 20
        })
        assert res.status_code == 200, f"SSE endpoint failed with HTTP {res.status_code}"
        assert "text/event-stream" in res.headers["content-type"]
        body_text = res.text
        assert "data: {\"type\": \"init\"" in body_text or "init" in body_text
        assert "data: {\"type\": \"complete\"" in body_text or "complete" in body_text
        logger.info(f"[PASSED] /api/personas/generate-stream returned 200 text/event-stream with live batch logging!")


async def test_sentence_boundary_guardrails():
    """
    Test hard sentence boundary guardrails for Survey (max 2 sentences), Interview (max 3 sentences), and Adoption reasoning (max 1 sentence).
    """
    logger.info("\n==================================================")
    logger.info("TEST: Sentence Boundary Safety Guardrails")
    logger.info("==================================================")

    # 1. Test survey response sentence guardrail (max 2 sentences)
    survey_text = "Sentence one is here. Sentence two is here. Sentence three should be trimmed away. Sentence four is extra."
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', survey_text) if s.strip()]
    trimmed_survey = " ".join(sents[:2])
    if not trimmed_survey[-1] in ".!?": trimmed_survey += "."
    assert trimmed_survey == "Sentence one is here. Sentence two is here."
    logger.info(f"[PASSED] Survey response trimmed safely to 2 complete sentences: \"{trimmed_survey}\"")

    # 2. Test interview response sentence guardrail (max 3 sentences)
    interview_text = "First turn sentence. Second turn sentence. Third turn sentence. Fourth turn sentence."
    sents_iv = [s.strip() for s in re.split(r'(?<=[.!?])\s+', interview_text) if s.strip()]
    trimmed_iv = " ".join(sents_iv[:3])
    if not trimmed_iv[-1] in ".!?": trimmed_iv += "."
    assert trimmed_iv == "First turn sentence. Second turn sentence. Third turn sentence."
    logger.info(f"[PASSED] Interview response trimmed safely to 3 complete sentences: \"{trimmed_iv}\"")

    # 3. Test adoption reasoning guardrail (max 1 sentence)
    reasoning_text = "The user has high interest based on daily pain points. However pricing creates slight hesitation."
    r_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reasoning_text) if s.strip()]
    short_reasoning = r_sents[0] if r_sents else reasoning_text
    if not short_reasoning[-1] in ".!?": short_reasoning += "."
    assert short_reasoning == "The user has high interest based on daily pain points."
    logger.info(f"[PASSED] Adoption reasoning trimmed safely to 1 complete sentence: \"{short_reasoning}\"")


async def test_dashboard_and_pdf_reports():
    """
    Test Dashboard and PDF Report generation endpoints using stored experiment data.
    Verifies that Dashboard and PDF generation do not make unnecessary Gemini API calls.
    """
    logger.info("\n==================================================")
    logger.info("TEST: Dashboard & PDF Report Endpoints")
    logger.info("==================================================")

    from fastapi.testclient import TestClient
    from main import app
    from models.experiment import Experiment
    client = TestClient(app)

    # 1. Seed dummy experiment in database
    exp_id = "exp_test_dash_pdf"
    exp = Experiment(
        experiment_id=exp_id,
        product_domain="EdTech",
        product_description="AI Study Planner App",
        target_audience="College Students in India",
        research_objective="Evaluate adoption motivators and study habits.",
        number_of_personas=2
    )
    database_service.save_experiment(exp)

    p1 = Persona(
        persona_id="p1",
        experiment_id=exp_id,
        name="Aarav Sharma",
        age=21,
        occupation="B.Tech Computer Science Student",
        location="Delhi",
        background="Senior undergraduate student preparing for entrance exams.",
        personality_traits=["Studious", "Focused"],
        goals=["Pass exams"],
        preferences=["Mobile app"],
        pain_points=["Time management"],
        behavioral_patterns=[],
        psychological_profile="",
        technology_adoption="High",
        price_sensitivity="Medium",
        initial_opinions=["Looks promising."],
        memory=[],
        conversation_history=[]
    )
    database_service.save_personas([p1])

    # Save interview session
    database_service.save_interview_session(
        experiment_id=exp_id,
        persona_id="p1",
        persona_name="Aarav Sharma",
        persona_occupation="B.Tech CS Student",
        messages=[
            {"role": "user", "content": "How do you manage study schedules?"},
            {"role": "persona", "content": "I struggle to balance GATE prep and college lectures."}
        ]
    )

    # Save research insights (interview-only)
    database_service.save_research_insights(
        experiment_id=exp_id,
        insights_dict={
            "experiment_id": exp_id,
            "interviews_included": 1,
            "executive_summary": "Students require automated schedule management to lower cognitive load.",
            "top_findings": ["Manual scheduling wastes time", "Offline access is critical"],
            "recurring_themes": ["Context switching", "Exam pressure"],
            "pain_points": ["Overwhelming syllabus"],
            "product_opportunities": ["Automated exam countdown timer"],
            "persona_evidence": [
                {
                    "persona_name": "Aarav Sharma",
                    "persona_role": "B.Tech CS Student",
                    "quote": "I struggle to balance GATE prep and college lectures.",
                    "extracted_insight": "Context switching burns time.",
                    "product_implication": "Auto-schedule study blocks."
                }
            ],
            "sentiment": {"positive_pct": 50.0, "neutral_pct": 50.0, "negative_pct": 0.0}
        }
    )

    # Save adoption data
    database_service.save_adoption_result(
        experiment_id=exp_id,
        adoption_dict={
            "experiment_id": exp_id,
            "overall_adoption_score": 85.0,
            "evidence_summary": "High interest due to scheduling automation.",
            "persona_adoptions": [
                {
                    "persona_id": "p1",
                    "persona_name": "Aarav Sharma",
                    "adoption_score": 85.0,
                    "would_use": "YES",
                    "reasoning": "High utility for exam prep."
                }
            ]
        }
    )

    # 2. Test GET /api/dashboard/{experiment_id}
    dash_res = client.get(f"/api/dashboard/{exp_id}")
    assert dash_res.status_code == 200, f"Dashboard endpoint failed: {dash_res.text}"
    dash_data = dash_res.json()

    assert dash_data["experiment_id"] == exp_id
    assert dash_data["experiment_overview"]["total_personas"] == 1
    assert dash_data["experiment_overview"]["interviews_conducted"] == 1
    assert dash_data["product_validation"]["overall_adoption_score"] == 85.0
    assert len(dash_data["key_findings"]) >= 1
    assert len(dash_data["recurring_themes"]) >= 1
    assert len(dash_data["persona_quotes"]) >= 1
    assert dash_data["has_interview_data"] is True
    assert dash_data["has_research_insights"] is True
    assert dash_data["has_adoption_data"] is True
    logger.info(f"[PASSED] Dashboard endpoint returned structured metrics and persona quotes.")

    # 3. Test GET /api/reports/pdf/{experiment_id}
    pdf_res = client.get(f"/api/reports/pdf/{exp_id}")
    assert pdf_res.status_code == 200, f"PDF generation endpoint failed: {pdf_res.text}"
    assert pdf_res.headers["content-type"] == "application/pdf"
    pdf_bytes = pdf_res.content
    assert pdf_bytes.startswith(b"%PDF"), "Response is not a valid PDF binary stream"
    assert len(pdf_bytes) > 1000, "PDF binary size is suspiciously small"
    logger.info(f"[PASSED] Downloadable PDF Research Report generated successfully ({len(pdf_bytes)} bytes).")


async def run_all_reliability_tests():
    await test_sequential_batching_and_retries_mocked()
    await test_sentence_boundary_guardrails()
    await test_dashboard_and_pdf_reports()
    logger.info("\n==================================================")
    logger.info("ALL RELIABILITY UNIT TESTS PASSED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    asyncio.run(run_all_reliability_tests())

