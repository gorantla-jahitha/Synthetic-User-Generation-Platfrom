import asyncio
import logging
from main import app
from fastapi.testclient import TestClient
from config import settings
from services.llm_service import llm_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("E2E_Test")

client = TestClient(app)

def test_full_workflow():
    logger.info("==================================================")
    logger.info("=== 1. Testing Health Check Endpoint ===")
    logger.info("==================================================")
    health_res = client.get("/api/health")
    assert health_res.status_code == 200, f"Health check failed: {health_res.text}"
    health_data = health_res.json()
    logger.info(f"Health check response: {health_data}")
    
    assert health_data["status"] == "online"
    assert "database" in health_data
    assert health_data["database"]["connected"] is True
    assert health_data["database"]["mode"] == "mongodb_persistent"

    llm_info = health_data.get("llm", {})
    is_gemini_real = llm_info.get("configured") is True and llm_info.get("mode") == "REAL"
    
    if is_gemini_real:
        logger.info(f"🚀 REAL GEMINI MODE ACTIVE! Model: {llm_info.get('model')}")
    else:
        logger.info("⚠️ FALLBACK / DEMO MODE ACTIVE (GEMINI_API_KEY not configured)")

    logger.info("==================================================")
    logger.info("=== 2. Testing Persona Generation Workflow ===")
    logger.info("==================================================")
    gen_payload = {
        "product_domain": "Health and Fitness Technology",
        "product_description": "A smart water bottle that tracks hydration, sends reminders through a mobile application, and provides personalized hydration recommendations.",
        "target_audience": "Urban Indian customers aged 18 to 40, including college students, software professionals, fitness enthusiasts, and people with sedentary lifestyles.",
        "research_objective": "Understand what factors influence customers to purchase the product, particularly price, health benefits, convenience, technology features, and willingness to use a companion mobile application.",
        "number_of_personas": 10
    }

    gen_res = client.post("/api/personas/generate", json=gen_payload)
    if gen_res.status_code == 503:
        err_msg = gen_res.json().get("detail")
        assert err_msg == "Gemini LLM is unavailable. Please configure or verify the Gemini API key.", f"Unexpected error detail: {err_msg}"
        logger.info(f"[CONFIRMED] API returned 503 as expected when Gemini is unavailable: '{err_msg}'")
        logger.info("[CONFIRMED] System NEVER falls back to hardcoded personas.")
        return
    
    assert gen_res.status_code == 200, f"Persona generation failed: {gen_res.text}"
    gen_data = gen_res.json()
    
    experiment_id = gen_data["experiment_id"]
    personas = gen_data["personas"]
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Successfully generated {len(personas)} personas.")
    assert len(personas) >= 5, f"Expected at least 5 personas, got {len(personas)}"

    # Print first 3 personas to inspect LLM variation
    for idx, p in enumerate(personas[:3], start=1):
        logger.info(f"Persona {idx}: Name={p['name']}, Age={p['age']}, Occupation={p['occupation']}, Location={p['location']}, Price Sensitivity={p['price_sensitivity']}, Tech={p['technology_adoption']}")
        logger.info(f"   Background: {p['background']}")
        logger.info(f"   Traits: {p['personality_traits']}")
        logger.info(f"   Goals: {p['goals']}")
        logger.info(f"   Pain Points: {p['pain_points']}")

    logger.info("==================================================")
    logger.info("=== 3. Testing Single Persona & Memory Endpoint ===")
    logger.info("==================================================")
    p0 = personas[0]
    p_res = client.get(f"/api/personas/{experiment_id}/{p0['persona_id']}")
    assert p_res.status_code == 200
    
    mem_res = client.get(f"/api/personas/{experiment_id}/{p0['persona_id']}/memory")
    assert mem_res.status_code == 200
    logger.info(f"Persona {p0['name']} initial memory count: {len(mem_res.json()['memory'])}")

    logger.info("==================================================")
    logger.info("=== 4. Testing REAL Survey Question 1 ===")
    logger.info("==================================================")
    survey_q1 = {
        "experiment_id": experiment_id,
        "question": "Would you pay ₹4,000 for this smart water bottle if it included six months of access to the mobile application?"
    }

    s1_res = client.post("/api/survey", json=survey_q1)
    assert s1_res.status_code == 200, f"Survey Q1 failed: {s1_res.text}"
    s1_data = s1_res.json()
    responses_1 = s1_data["responses"]
    logger.info(f"Received {len(responses_1)} side-by-side responses.")
    
    for r in responses_1[:3]:
        logger.info(f"[{r['persona_name']}] Answer: \"{r['response']}\" | Consistency: {r['consistency']['is_consistent']} ({r['consistency']['confidence']} Confidence)")

    logger.info("==================================================")
    logger.info("=== 5. Testing Multi-Turn Memory Question 2 ===")
    logger.info("==================================================")
    survey_q2 = {
        "experiment_id": experiment_id,
        "question": "What would stop you from purchasing this product?"
    }

    s2_res = client.post("/api/survey", json=survey_q2)
    assert s2_res.status_code == 200, f"Survey Q2 failed: {s2_res.text}"
    s2_data = s2_res.json()
    responses_2 = s2_data["responses"]
    
    for r in responses_2[:3]:
        logger.info(f"[{r['persona_name']}] Turn 2 Answer: \"{r['response']}\" | Memory Updated: {r.get('memory_updates')}")

    logger.info("==================================================")
    logger.info("=== 6. Testing Survey History Endpoint ===")
    logger.info("==================================================")
    hist_res = client.get(f"/api/survey/{experiment_id}/history")
    assert hist_res.status_code == 200
    hist_turns = hist_res.json()["history"]
    assert len(hist_turns) == 2
    logger.info(f"Retrieved {len(hist_turns)} turns from survey history in MongoDB.")

    logger.info("==================================================")
    logger.info("✅ E2E VERIFICATION COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    test_full_workflow()
