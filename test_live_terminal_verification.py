import sys
import os
import logging
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def run_verification():
    print("\n==================================================")
    print("=== LIVE TERMINAL BATCH LOGGING TEST: 5 PERSONAS ===")
    print("==================================================")

    res_5 = client.post("/api/personas/generate-stream", json={
        "product_domain": "Fitness Tech",
        "product_description": "Smart Hydration Bottle",
        "target_audience": "Gym goers and athletes in India",
        "research_objective": "Evaluate price and mobile app interest",
        "number_of_personas": 5
    })

    assert res_5.status_code == 200, f"5-persona generation failed: {res_5.text}"

    print("\n==================================================")
    print("=== LIVE TERMINAL BATCH LOGGING TEST: 50 PERSONAS ===")
    print("==================================================")

    res_50 = client.post("/api/personas/generate-stream", json={
        "product_domain": "EdTech",
        "product_description": "AI Study Planner",
        "target_audience": "College Students in India aged 18 to 25",
        "research_objective": "Study habits and exam preparation tools adoption",
        "number_of_personas": 50
    })

    assert res_50.status_code == 200, f"50-persona generation failed: {res_50.text}"
    print("\n[VERIFICATION COMPLETE] Test client completed successfully.")

if __name__ == "__main__":
    run_verification()
