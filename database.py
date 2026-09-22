import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.is_connected = False
        self.status_message = "Initializing..."
        # In-memory stores when MongoDB is offline
        self.in_memory_experiments = {}
        self.in_memory_personas = {}
        self.in_memory_memories = {}
        self.in_memory_survey_sessions = {}
        self.in_memory_survey_responses = {}
        self.in_memory_interviews = {}
        self.in_memory_survey_insights = {}
        self.in_memory_research_insights = {}
        self.in_memory_adoption_results = {}
        self.in_memory_execution_states = {}
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=2000
            )
            # Force a call to verify connection
            self.client.admin.command('ping')
            self.db = self.client[settings.DATABASE_NAME]
            self.is_connected = True
            self.status_message = f"Connected to MongoDB at {settings.MONGODB_URI} [{settings.DATABASE_NAME}]"
            logger.info(self.status_message)
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            self.is_connected = False
            self.status_message = f"MongoDB connection unavailable ({str(e)}). Running in transparent temporary in-memory mode."
            logger.warning(self.status_message)

    def get_status(self):
        # Re-check on query if previously failed
        if not self.is_connected:
            try:
                self.client.admin.command('ping')
                self.is_connected = True
                self.db = self.client[settings.DATABASE_NAME]
                self.status_message = f"Connected to MongoDB at {settings.MONGODB_URI}"
            except Exception:
                pass

        return {
            "connected": self.is_connected,
            "status_message": self.status_message,
            "mode": "mongodb_persistent" if self.is_connected else "offline_in_memory"
        }

db_manager = DatabaseManager()
