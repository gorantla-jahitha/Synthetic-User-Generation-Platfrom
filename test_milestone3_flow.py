import sys
import os
import asyncio
import json
import logging

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db_manager
from models.experiment import CreateExperimentRequest, Experiment
from models.survey import SurveyRequest
from services.database_service import database_service
from agents.persona_generation_agent import persona_generation_agent
from agents.survey_agent import survey_agent
from agents.interview_agent import interview_agent
from agents.insight_agent import insight_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Milestone3Test")

async def run_milestone3_verification():
    logger.info("==================================================")
    logger.info("STARTING MILESTONE 3 END-TO-END VERIFICATION TEST")
    logger.info("==================================================")

    # 1. Milestone 1: Create Research Experiment
    exp_req = CreateExperimentRequest(
        product_domain="Education Technology",
        product_description="An AI-powered smart study planner that automatically schedules study sessions and tracks learning progress.",
        target_audience="College students in India aged 18 to 24",
        research_objective="Understand what motivates students to adopt AI study planning tools and identifying major adoption barriers.",
        number_of_personas=5
    )

    exp = Experiment(
        product_domain=exp_req.product_domain,
        product_description=exp_req.product_description,
        target_audience=exp_req.target_audience,
        research_objective=exp_req.research_objective,
        number_of_personas=exp_req.number_of_personas
    )
    database_service.save_experiment(exp)
    exp_id = exp.experiment_id

    logger.info(f"1. Generating Synthetic Personas for Experiment {exp_id} (Milestone 2 baseline)...")
    personas = await persona_generation_agent.generate_personas(exp_req, exp_id)
    database_service.save_personas(personas)

    assert len(personas) == 5, f"Expected 5 personas, got {len(personas)}"
    logger.info(f"-> Generated {len(personas)} personas for Experiment ID: {exp_id}")
    for p in personas:
        logger.info(f"   - Persona ID: {p.persona_id} | Name: {p.name} | Occupation: {p.occupation} | Age: {p.age}")

    # Verify personas persisted in DB
    db_personas = database_service.get_personas_for_experiment(exp_id)
    assert len(db_personas) == 5, "Personas failed to persist in database!"
    logger.info("[SUCCESS] Milestone 1 & 2 Persona Generation Verified.")

    await asyncio.sleep(2)

    # 2. Milestone 2: Batch Survey Execution
    logger.info("\n2. Executing Batch Survey Mode (Milestone 2 baseline)...")
    survey_q = "What is the biggest feature that would convince you to use this study planner app?"
    survey_req = SurveyRequest(experiment_id=exp_id, question=survey_q)
    survey_resp = await survey_agent.run_survey(survey_req)
    assert len(survey_resp.responses) == 5, "Survey response count mismatch!"
    logger.info(f"-> Survey completed for question: '{survey_q}' with 3 persona responses.")
    for r in survey_resp.responses:
        logger.info(f"   - {r.persona_name}: '{r.response[:80]}...'")
    logger.info("[SUCCESS] Milestone 2 Batch Survey Verified.")

    await asyncio.sleep(2)

    # 3. Milestone 3: AI Person Interview (Persona Isolation & Multi-turn History)
    logger.info("\n3. Testing AI Person Interview Mode (Milestone 3)...")
    persona_1 = personas[0]
    persona_2 = personas[1]

    logger.info(f"-> Interviewing Persona 1: {persona_1.name} ({persona_1.occupation})...")
    turn1 = await interview_agent.process_user_message(
        experiment_id=exp_id,
        persona_id=persona_1.persona_id,
        user_message="Hello! What challenges do you face when organizing your daily studies?"
    )
    logger.info(f"   Response from {turn1['persona_name']}: '{turn1['persona_response'][:90]}...'")

    await asyncio.sleep(2)

    turn2 = await interview_agent.process_user_message(
        experiment_id=exp_id,
        persona_id=persona_1.persona_id,
        user_message="Why is that specific challenge so difficult for you?"
    )
    logger.info(f"   Follow-up Response from {turn1['persona_name']}: '{turn2['persona_response'][:90]}...'")

    # Verify session history for Persona 1
    session_p1 = database_service.get_interview_session(exp_id, persona_1.persona_id)
    assert len(session_p1["messages"]) == 4, f"Expected 4 messages for Persona 1, got {len(session_p1['messages'])}"

    await asyncio.sleep(2)

    # Interview Persona 2 to verify separate history isolation
    logger.info(f"-> Interviewing Persona 2: {persona_2.name} ({persona_2.occupation})...")
    turn_p2 = await interview_agent.process_user_message(
        experiment_id=exp_id,
        persona_id=persona_2.persona_id,
        user_message="Would you be willing to pay a monthly subscription for this study app?"
    )
    logger.info(f"   Response from {turn_p2['persona_name']}: '{turn_p2['persona_response'][:90]}...'")

    session_p2 = database_service.get_interview_session(exp_id, persona_2.persona_id)
    assert len(session_p2["messages"]) == 2, f"Persona 2 should have 2 messages, got {len(session_p2['messages'])}"
    assert session_p1["messages"] != session_p2["messages"], "Interview histories must be isolated!"
    logger.info("[SUCCESS] Persona Isolation & Multi-turn Chat Memory Verified.")

    await asyncio.sleep(2)

    # 4. Milestone 3: Interview Insights Generation
    logger.info("\n4. Generating Interview Research Insights for Persona 1...")
    iv_insights = await interview_agent.generate_insights(exp_id, persona_1.persona_id)
    assert "key_insights" in iv_insights, "Missing key_insights in interview insights"
    logger.info(f"-> Insights generated for {persona_1.name}: Intent={iv_insights.get('purchase_intent')}, Sentiment={iv_insights.get('sentiment_breakdown')}")
    logger.info("[SUCCESS] Interview Research Insights Verified.")

    await asyncio.sleep(2)

    # 5. Milestone 3: Survey Research Insights
    logger.info("\n5. Generating Survey Research Insights...")
    surv_insights = await insight_agent.generate_survey_insights(exp_id)
    assert surv_insights["total_responses_analyzed"] == 5
    assert "sentiment" in surv_insights
    logger.info(f"-> Survey Insights: Analyzed {surv_insights['total_responses_analyzed']} responses. Sentiment: Pos={surv_insights['sentiment']['positive_pct']}%, Neu={surv_insights['sentiment']['neutral_pct']}%, Neg={surv_insights['sentiment']['negative_pct']}%")
    logger.info("[SUCCESS] Survey Research Insights Verified.")

    await asyncio.sleep(2)

    # 6. Milestone 3: Interview-Only AI Research Insights
    logger.info("\n6. Generating Interview-Only AI Research Insights...")
    res_insights = await insight_agent.generate_integrated_research_insights(exp_id)
    assert res_insights["interviews_included"] is True
    assert res_insights["surveys_included"] is False, "Survey data must be 100% excluded from AI Research Insights!"
    assert res_insights["interviewed_personas_count"] == 2, f"Expected 2 interviewed personas, got {res_insights['interviewed_personas_count']}"
    assert len(res_insights["persona_findings"]) > 0, "Missing persona-specific findings!"
    
    logger.info(f"-> AI Research Insights Executive Summary: '{res_insights['executive_summary'][:100]}...'")
    logger.info(f"-> Interviewed Personas Analyzed ({res_insights['interviewed_personas_count']}): {res_insights['interviewed_persona_names']}")
    for pf in res_insights["persona_findings"]:
        logger.info(f"   - Persona: {pf.get('persona_name')} ({pf.get('persona_occupation')})")
        logger.info(f"     Key Response: '{pf.get('key_response')[:60]}...'")
        logger.info(f"     Insight: '{pf.get('extracted_insight')}'")
    logger.info("[SUCCESS] Interview-Only AI Research Insights Verified.")

    await asyncio.sleep(2)

    # 7. Milestone 3: "Would You Use This Product?" / Product Adoption Scoring
    logger.info("\n7. Analyzing Product Adoption Scoring...")
    adoption_res = await insight_agent.analyze_product_adoption(exp_id)
    assert 0.0 <= adoption_res["overall_adoption_score"] <= 100.0, f"Invalid overall adoption score: {adoption_res['overall_adoption_score']}"
    assert len(adoption_res["persona_adoptions"]) == 5

    logger.info(f"-> Overall Adoption Score: {adoption_res['overall_adoption_score']}%")
    for pa in adoption_res["persona_adoptions"]:
        logger.info(f"   - {pa['persona_name']}: Would Use = {pa['decision']} | Score = {pa['adoption_score']}% | Reasoning = '{pa['reasoning'][:60]}...'")

    logger.info("[SUCCESS] Product Adoption Scoring Verified.")
    logger.info("\n==================================================")
    logger.info("ALL MILESTONE 3 VERIFICATION CHECKS PASSED PERFECTLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    asyncio.run(run_milestone3_verification())
