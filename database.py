
import os
import pymongo
from datetime import datetime
from bson.objectid import ObjectId

# Config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/mental_health_db")

class Database:
    def __init__(self):
        self.memory_db = {
            'users': [],
            'journals': [],
            'streaks': {}
        }
        self.db = None
        self.local_file = "data/local_db.json"
        
        try:
            self.client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            self.client.admin.command('ping')
            self.db = self.client.get_database()
            self.users = self.db.users
            self.journals = self.db.journals
            self.streaks = self.db.streaks
            print(f"✓ Connected to MongoDB at {MONGO_URI}")
        except Exception as e:
            print(f"✗ Failed to connect to MongoDB ({e}). Using Persistent JSON Fallback.")
            self.db = None
            self._load_local_db()

    def _load_local_db(self):
        if os.path.exists(self.local_file):
            try:
                import json
                with open(self.local_file, 'r') as f:
                    data = json.load(f)
                    # Convert isoformat str back to datetime where needed might be complex, 
                    # but for now we load raw. Dates might need handling.
                    self.memory_db = data
                    
                    # Deserialize dates
                    for user in self.memory_db.get('users', []):
                        if isinstance(user.get('created_at'), str):
                            user['created_at'] = datetime.fromisoformat(user['created_at'])
                        for assessment in user.get('assessments', []):
                            if isinstance(assessment.get('date'), str):
                                assessment['date'] = datetime.fromisoformat(assessment['date'])
                                
                    for journal in self.memory_db.get('journals', []):
                        if isinstance(journal.get('date'), str):
                            journal['date'] = datetime.fromisoformat(journal['date'])
                print(f"✓ Loaded data from {self.local_file}")
            except Exception as e:
                print(f"Error loading local db: {e}")
                
    def _save_local_db(self):
        if self.db is None:
            import json
            # Helper to serialize datetime
            def json_serial(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError ("Type not serializable")
                
            try:
                with open(self.local_file, 'w') as f:
                    json.dump(self.memory_db, f, default=json_serial, indent=2)
            except Exception as e:
                print(f"Error saving local db: {e}")

    # User CRUD
    def create_user(self, name, email, password_hash):
        if self.db is None:
            # Fallback
            self.memory_db['users'].append({
                "name": name, 
                "email": email, 
                "password_hash": password_hash,
                "created_at": datetime.now(),
                "created_at": datetime.now(),
                "assessments": []
            })
            self._save_local_db()
            return True
            
        try:
            user = {
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "created_at": datetime.now(),
                "assessments": []
            }
            self.users.insert_one(user)
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def get_user(self, email):
        if self.db is None:
            return next((u for u in self.memory_db['users'] if u['email'] == email), None)
        return self.users.find_one({"email": email})

    def update_user(self, email, data):
        if self.db is None: return False # Simple fallback doesn't support advanced update yet
        self.users.update_one({"email": email}, {"$set": data})
        return True

    # Assessment
    def save_assessment(self, email, assessment_data):
        if self.db is None:
            user = self.get_user(email)
            if user: 
                user['assessments'].append(assessment_data)
                self._save_local_db()
            return True
            
        assessment_data['date'] = datetime.now()
        self.users.update_one(
            {"email": email},
            {"$push": {"assessments": assessment_data}}
        )
        return True

    def get_history(self, email):
        user = self.get_user(email)
        return user.get('assessments', []) if user else []

    # Journal
    def save_journal(self, email, content, mood):
        entry = {
            "email": email,
            "content": content,
            "mood": mood,
            "date": datetime.now()
        }
        
        if self.db is None:
            self.memory_db['journals'].append(entry)
            self.update_streak(email)
            self._save_local_db()
            return True
            
        self.journals.insert_one(entry)
        self.update_streak(email)
        return True
        
    def get_journals(self, email):
        if self.db is None:
            return sorted([j for j in self.memory_db['journals'] if j['email'] == email], key=lambda x: x['date'], reverse=True)
            
        return list(self.journals.find({"email": email}).sort("date", -1))

    # Gamification
    def update_streak(self, email):
        today = datetime.now().date()
        date_str = today.isoformat()
        
        if self.db is None:
            streak_data = self.memory_db['streaks'].get(email, {"current_streak": 0, "last_active": "", "max_streak": 0})
            
            if streak_data['last_active']:
                last_active = datetime.fromisoformat(streak_data['last_active']).date()
                if (today - last_active).days == 1:
                    streak_data['current_streak'] += 1
                elif (today - last_active).days > 1:
                    streak_data['current_streak'] = 1
            else:
                streak_data['current_streak'] = 1
                
            streak_data['last_active'] = date_str
            streak_data['max_streak'] = max(streak_data['current_streak'], streak_data['max_streak'])
            self.memory_db['streaks'][email] = streak_data
            return

        streak_doc = self.streaks.find_one({"email": email})
        
        if not streak_doc:
            self.streaks.insert_one({
                "email": email,
                "current_streak": 1,
                "last_active": date_str,
                "max_streak": 1
            })
        else:
            last_active = datetime.fromisoformat(streak_doc['last_active']).date()
            if (today - last_active).days == 1:
                # Consecutive day
                new_streak = streak_doc['current_streak'] + 1
                self.streaks.update_one(
                    {"email": email},
                    {
                        "$set": {
                            "current_streak": new_streak,
                            "last_active": date_str,
                            "max_streak": max(new_streak, streak_doc['max_streak'])
                        }
                    }
                )
            elif (today - last_active).days > 1:
                # Streak broken
                self.streaks.update_one(
                    {"email": email},
                    {"$set": {"current_streak": 1, "last_active": date_str}}
                )

    def get_streak(self, email):
        if self.db is None:
            return self.memory_db['streaks'].get(email, {}).get('current_streak', 0)
            
        doc = self.streaks.find_one({"email": email})
        return doc['current_streak'] if doc else 0

# Global Instance
db = Database()
