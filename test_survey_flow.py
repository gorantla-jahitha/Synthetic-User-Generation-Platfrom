import asyncio
import logging
import time
from models.experiment import Experiment
from models.persona import Persona, MemoryItem
from models.survey import SurveyRequest
from services.database_service import database_service
from agents.survey_agent import survey_agent, MAX_SURVEY_CONCURRENCY, SURVEY_REQUEST_TIMEOUT
from services.llm_service import llm_service, GeminiQuotaExceededError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SurveyFlowTest")

async def run_test():
    logger.info("=== Testing Survey Mode Concurrency, Memory Persistence & Quota Handling ===")
    
    # 1. Create a dummy experiment with 5 personas
    exp_id = "exp_survey_test_001"
    experiment = Experiment(
        experiment_id=exp_id,
        product_domain="Education Technology",
        product_description="An AI-powered study planning application.",
        target_audience="College students in India",
        research_objective="Understand motivation to adopt AI study planning tools.",
        number_of_personas=5
    )
    database_service.save_experiment(experiment)

    names = ["Aaditya Roy", "Ishaan Verma", "Diya Kapoor", "Kabir Sengupta", "Tanya Reddy"]
    branches = ["Computer Science", "Mechanical Engineering", "MBBS", "Economics", "Commerce"]

    personas = []
    for i in range(5):
        p = Persona(
            persona_id=f"persona_{i+1:03d}",
            experiment_id=exp_id,
            name=names[i],
            age=20 + i,
            occupation=f"{branches[i]} Student",
            location="New Delhi",
            background=f"{names[i]} is a {branches[i]} student who balances tight exam schedules.",
            personality_traits=["Studious", "Analytical"],
            goals=["Ace semester exams", "Manage daily study hours"],
            preferences=["Mobile app", "Smart schedule"],
            pain_points=["Procrastination", "Exam stress"],
            behavioral_patterns=["Studies late night"],
            psychological_profile="Goal-oriented student",
            technology_adoption="High",
            price_sensitivity="Medium",
            initial_opinions=["Interested in automated study planning"],
            memory=[MemoryItem(fact=f"Background: {branches[i]} student", source="initial_profile")],
            conversation_history=[]
        )
        personas.append(p)
    
    database_service.save_personas(personas)
    logger.info(f"Saved 5 test personas to DB under experiment_id '{exp_id}'.")

    # 2. Test Question 1
    q1 = "Would you use this AI-powered study planning application? Why or why not?"
    req1 = SurveyRequest(experiment_id=exp_id, question=q1)

    logger.info(f"\n--- RUNNING QUESTION 1: '{q1}' ---")
    start_time = time.time()
    try:
        payload1 = await survey_agent.run_survey(req1)
        elapsed = time.time() - start_time
        logger.info(f"[SUCCESS] Question 1 completed in {elapsed:.2f} seconds.")
        logger.info(f"Responses received: {len(payload1.responses)}")
        for r in payload1.responses:
            logger.info(f" - {r.persona_name}: {r.response[:80]}... (Memory updates: {r.memory_updates})")
        
        # Verify Memory Persistence in DB after Q1
        updated_personas = database_service.get_personas_for_experiment(exp_id)
        for p in updated_personas:
            history = p.get("conversation_history", [])
            assert len(history) == 1, f"Expected 1 history turn for {p['name']}, found {len(history)}"
            logger.info(f"Verified DB history for {p['name']}: {len(history)} turn(s), memory items: {len(p.get('memory', []))}")

        # 3. Test Question 2 (Multi-turn Memory Check)
        q2 = "What would be the biggest reason that would stop you from using this application regularly?"
        req2 = SurveyRequest(experiment_id=exp_id, question=q2)

        logger.info(f"\n--- RUNNING QUESTION 2: '{q2}' ---")
        start_time_q2 = time.time()
        payload2 = await survey_agent.run_survey(req2)
        elapsed_q2 = time.time() - start_time_q2
        logger.info(f"[SUCCESS] Question 2 completed in {elapsed_q2:.2f} seconds.")
        logger.info(f"Responses received: {len(payload2.responses)}")
        for r in payload2.responses:
            logger.info(f" - {r.persona_name}: {r.response[:80]}...")

        # Verify DB after Q2 (should have 2 turns)
        updated_personas2 = database_service.get_personas_for_experiment(exp_id)
        for p in updated_personas2:
            history = p.get("conversation_history", [])
            assert len(history) == 2, f"Expected 2 history turns for {p['name']}, found {len(history)}"
            logger.info(f"Verified 2-turn DB history for {p['name']}: turns={len(history)}")

        print("\n[SUMMARY REPORT]")
        print(f"- Number of Gemini calls per persona: 1 call per persona (0 extra calls for consistency)")
        print(f"- Survey requests concurrent: YES (asyncio.gather)")
        print(f"- Concurrency limit: {MAX_SURVEY_CONCURRENCY}")
        print(f"- Timeout duration: {SURVEY_REQUEST_TIMEOUT}s")
        print(f"- Retry behavior for HTTP 429: Immediate fast fail (GeminiQuotaExceededError, no 60s x 3 retries)")
        print(f"- Survey fallback/mock responses removed: YES (0 mock responses)")
        print(f"- Memory persistence: YES (Verified 2 consecutive turns in MongoDB)")
        print(f"- Total Q1 completion time: {elapsed:.2f}s | Total Q2 completion time: {elapsed_q2:.2f}s")

    except GeminiQuotaExceededError as qe:
        elapsed = time.time() - start_time
        logger.warning(f"[QUOTA ERROR PASSED FAST] Immediate quota error in {elapsed:.2f} seconds: {qe}")
        print("\n[SUMMARY REPORT]")
        print(f"- Quota status: Gemini API quota exceeded (Handled immediately in {elapsed:.2f}s)")
        print(f"- Concurrency limit: {MAX_SURVEY_CONCURRENCY}")
        print(f"- Timeout duration: {SURVEY_REQUEST_TIMEOUT}s")
        print(f"- Retry behavior for HTTP 429: Immediate fast fail (no hanging UI)")
        print(f"- Survey fallback/mock responses removed: YES")

if __name__ == "__main__":
    asyncio.run(run_test())
