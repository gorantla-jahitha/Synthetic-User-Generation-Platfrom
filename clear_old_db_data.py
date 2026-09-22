import sys
sys.path.append('backend')
from database import db_manager

def clear_old_data():
    if db_manager.is_connected:
        db_manager.db.personas.delete_many({})
        db_manager.db.experiments.delete_many({})
        db_manager.db.survey_responses.delete_many({})
        print("[SUCCESS] Cleared old test personas and experiments from MongoDB!")
    else:
        print("[INFO] MongoDB is not connected; in-memory store is clean on startup.")

if __name__ == "__main__":
    clear_old_data()
