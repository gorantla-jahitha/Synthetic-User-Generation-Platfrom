import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from database import db_manager
from routes.experiment_routes import router as experiment_router
from routes.persona_routes import router as persona_router
from routes.survey_routes import router as survey_router
from routes.interview_routes import router as interview_router
from routes.insight_routes import router as insight_router
from routes.dashboard_routes import router as dashboard_router
from services.llm_service import llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend REST API for Prompt-Based Synthetic Customer Research & Behavioral Simulation Platform"
)

# Standard FastAPI CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(experiment_router)
app.include_router(persona_router)
app.include_router(survey_router)
app.include_router(interview_router)
app.include_router(insight_router)
app.include_router(dashboard_router)

# Safe startup log — API key is NEVER logged
_llm_status = llm_service.get_status()
logger.info("=" * 60)
logger.info(f"LLM Provider: {_llm_status['provider'].upper()}")
logger.info(f"LLM Model:    {_llm_status['model']}")
logger.info(f"LLM Mode:     {_llm_status['mode']}")
logger.info(f"LLM Configured: {_llm_status['configured']}")
logger.info("=" * 60)

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint returning DB connectivity status and LLM readiness.
    """
    db_status = db_manager.get_status()
    llm_status = llm_service.get_status()
    
    return {
        "status": "online",
        "version": settings.VERSION,
        "database": db_status,
        "llm": llm_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
