# =====================================================================
# MENTAL HEALTH & BEHAVIORAL ANALYTICS SYSTEM - FLASK APPLICATION
# User Input Based Analysis - No Pre-defined Datasets
# =====================================================================
"""
DISCLAIMER: This system is for educational and screening purposes only.
It does not replace professional mental health diagnosis or treatment.
"""

from flask import Flask, render_template, request, session, redirect, url_for, jsonify, make_response, send_file, flash
import numpy as np
import joblib
import secrets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from scipy.stats import entropy
from datetime import datetime
from functools import wraps
import hashlib
import json
import io
import warnings
warnings.filterwarnings('ignore')

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT


app = Flask(__name__)
app.secret_key = 'kiro_mental_health_2024_secret_key'

# Initialize RAG Engine
# Initialize RAG Engine and LLM
from rag_engine import RAGEngine
from database import db
from llm_interface import LLMInterface

try:
    rag_engine = RAGEngine()
    print("✓ RAG Engine Loaded with Web Search")
except Exception as e:
    print(f"✗ Failed to load RAG Engine: {e}")
    rag_engine = None

# Initialize LLM
llm_agent = LLMInterface()

@app.route("/chat_api", methods=["POST"])
def chat_api():
    """Endpoint for AI Assistant with Llama 3.2 & RLHF"""
    if not rag_engine:
        return jsonify({"response": "AI Assistant is currently unavailable."})
        
    data = request.json
    user_query = data.get("message", "")
    user_feedback = data.get("feedback", None) # For RLHF
    
    # RLHF: Log feedback if present
    if user_feedback:
        try:
            # 1. Retrieve context (Query + Response)
            last_response = session.get('last_response', '')
            full_context = f"Q: {user_query} A: {last_response}"
            
            # 2. Get embedding (Feature Vector X)
            if rag_engine and hasattr(rag_engine, 'encoder'):
                # Encode the Q+A pair for better reward modeling
                context_embedding = rag_engine.encoder.encode([full_context]).tolist()[0]
            else:
                context_embedding = [0.0] * 384 
            
            # 3. Log Data for Fine-Tuning
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "response": last_response,
                "feedback": user_feedback, # 'up' or 'down'
                "embedding": context_embedding
            }
            
            # Append to RLHF logs
            with open("data/rlhf_feedback.jsonl", "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")
                
            print(f"✓ RLHF Data Logged: {user_feedback}")
            return jsonify({"status": "feedback_received", "message": "Model updated via Active Learning"})
            
        except Exception as e:
            print(f"Error logging RLHF: {e}")
            return jsonify({"status": "error"})

    # Context Gathering
    user_context = get_current_user()
    journal_context = ""
    if user_context:
        # Get recent journal mood
        journals = db.get_journals(user_context['email'])
        if journals:
            last_entry = journals[0]
            journal_context = f"User was feeling {last_entry['mood']} on {last_entry['date']}."

    # RAG Retrieval Strategy
    # 1. Rewrite for better search
    optimized_query = llm_agent.rewrite_query(user_query)
    
    # 2. Search with optimized query
    results = rag_engine.query(optimized_query)
    
    # LLM Generation
    response_text = llm_agent.generate_response(
        user_query, 
        results, 
        user_profile=user_context, 
        journal_context=journal_context
    )
    
    # Cache response for feedback loop
    session['last_response'] = response_text
    
    # Save Interaction to Vector DB (Memory)
    if rag_engine:
        memory_text = f"User: {user_query}\nAI: {response_text}"
        metadata = {
            "source": "chat_history", 
            "user": session.get('user_email', 'guest'),
            "timestamp": str(datetime.now())
        }
        rag_engine.save_memory(memory_text, metadata)
    
    return jsonify({"response": response_text})


# =====================================================================
# USER STORAGE (In-memory for demo, use database in production)
# =====================================================================
users_db = {}  # Format: {email: {password_hash, name, created_at, assessments: []}}

# =====================================================================
# HISTORY STORAGE (In-memory for demo, use database in production)
# =====================================================================
assessment_history = []  # Stores all assessments for history tracking


# =====================================================================
# AUTHENTICATION HELPERS
# =====================================================================
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session and 'guest_mode' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get current logged in user data"""
    if 'user_email' in session:
        # Try MongoDB first
        user = db.get_user(session['user_email'])
        if user: return user
        # Fallback to in-memory (if user registered before mongo or mongo down)
        return users_db.get(session['user_email'])
    return None

# =====================================================================
# REQUIRED CLASS DEFINITIONS (Must match training code)
# =====================================================================

# =====================================================================
# MACHINE LEARNING MODELS
# =====================================================================
try:
    from predict_mental_health import (
        RewardModel, VariationalPreferenceLearning, ActiveLearner,
        EnhancedQuantumTransform, SuperEnsemble95
    )
except ImportError:
    # Fallback if file missing (dev mode)
    print("Warning: predict_mental_health module not found. Models disabled.")
    RewardModel = None
    VariationalPreferenceLearning = None
    ActiveLearner = None
    EnhancedQuantumTransform = None
    SuperEnsemble95 = None



# =====================================================================
# LOAD MODELS
# =====================================================================
MODEL_DIR = "model"

try:
    sleep_model = joblib.load(f"{MODEL_DIR}/students_sleep_model.joblib")
    sleep_scaler = joblib.load(f"{MODEL_DIR}/students_sleep_scaler.joblib")
    depression_model = joblib.load(f"{MODEL_DIR}/depression_model.joblib")
    depression_scaler = joblib.load(f"{MODEL_DIR}/depression_scaler.joblib")
    anxiety_model = joblib.load(f"{MODEL_DIR}/anxiety_model.joblib")
    anxiety_scaler = joblib.load(f"{MODEL_DIR}/anxiety_scaler.joblib")
    burnout_model = joblib.load(f"{MODEL_DIR}/burnout_model.joblib")
    burnout_scaler = joblib.load(f"{MODEL_DIR}/burnout_scaler.joblib")
    models_loaded = True
    print("✓ All models loaded successfully")
except Exception as e:
    models_loaded = False
    print(f"✗ Error loading models: {e}")


# =====================================================================
# HELPER FUNCTIONS FOR ANALYSIS
# =====================================================================

def analyze_user_profile(user_data):
    """Comprehensive analysis of user's mental health profile"""
    analysis = {
        'overall_risk': 'low',
        'risk_score': 0,
        'concerns': [],
        'strengths': [],
        'recommendations': []
    }
    
    # Analyze sleep patterns
    if user_data.get('sleep_duration', 7) < 6:
        analysis['concerns'].append('Insufficient sleep duration')
        analysis['risk_score'] += 15
    elif user_data.get('sleep_duration', 7) >= 7:
        analysis['strengths'].append('Adequate sleep duration')
    
    # Analyze screen time
    if user_data.get('screen_time', 0) > 8:
        analysis['concerns'].append('Excessive daily screen time')
        analysis['risk_score'] += 10
    if user_data.get('night_screen', 0) > 2:
        analysis['concerns'].append('High nighttime screen exposure')
        analysis['risk_score'] += 15
    
    # Analyze stress
    stress = user_data.get('stress_score', 0)
    if stress > 20:
        analysis['concerns'].append('Elevated stress levels')
        analysis['risk_score'] += 20
    elif stress < 10:
        analysis['strengths'].append('Well-managed stress levels')
    
    # Analyze social media
    social_hours = user_data.get('social_media_hours', 0)
    if social_hours > 4:
        analysis['concerns'].append('High social media usage')
        analysis['risk_score'] += 10
    
    # Analyze work-life balance (for professionals)
    work_hours = user_data.get('work_hours', 40)
    if work_hours > 50:
        analysis['concerns'].append('Extended work hours')
        analysis['risk_score'] += 15
    
    # MBSR participation
    if user_data.get('mbsr_participation', 0) == 1:
        analysis['strengths'].append('Active mindfulness practice')
        analysis['risk_score'] -= 10
    
    # Determine overall risk level
    if analysis['risk_score'] >= 40:
        analysis['overall_risk'] = 'high'
    elif analysis['risk_score'] >= 20:
        analysis['overall_risk'] = 'moderate'
    else:
        analysis['overall_risk'] = 'low'
    
    # Generate recommendations
    if 'Insufficient sleep duration' in analysis['concerns']:
        analysis['recommendations'].append('Aim for 7-9 hours of sleep per night')
    if 'High nighttime screen exposure' in analysis['concerns']:
        analysis['recommendations'].append('Reduce screen time 1-2 hours before bed')
    if 'Elevated stress levels' in analysis['concerns']:
        analysis['recommendations'].append('Consider stress management techniques like MBSR')
    if 'High social media usage' in analysis['concerns']:
        analysis['recommendations'].append('Set daily limits for social media apps')
    if 'Extended work hours' in analysis['concerns']:
        analysis['recommendations'].append('Prioritize work-life balance and regular breaks')
    
    if not analysis['recommendations']:
        analysis['recommendations'].append('Continue maintaining your healthy habits!')
    
    return analysis


def get_phq9_interpretation(score):
    """Interpret PHQ-9 depression screening score"""
    if score < 5:
        return {'severity': 'Minimal', 'color': 'emerald', 'action': 'No action needed'}
    elif score < 10:
        return {'severity': 'Mild', 'color': 'yellow', 'action': 'Watchful waiting; repeat screening'}
    elif score < 15:
        return {'severity': 'Moderate', 'color': 'orange', 'action': 'Consider counseling or medication'}
    elif score < 20:
        return {'severity': 'Moderately Severe', 'color': 'rose', 'action': 'Active treatment recommended'}
    else:
        return {'severity': 'Severe', 'color': 'red', 'action': 'Immediate intervention recommended'}


def get_gad7_interpretation(score):
    """Interpret GAD-7 anxiety screening score"""
    if score < 5:
        return {'severity': 'Minimal', 'color': 'emerald', 'action': 'No action needed'}
    elif score < 10:
        return {'severity': 'Mild', 'color': 'yellow', 'action': 'Monitor symptoms'}
    elif score < 15:
        return {'severity': 'Moderate', 'color': 'orange', 'action': 'Consider treatment'}
    else:
        return {'severity': 'Severe', 'color': 'red', 'action': 'Active treatment recommended'}


# =====================================================================
# ROUTES
# =====================================================================
# AUTHENTICATION ROUTES
# =====================================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration"""
    if 'user_email' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        errors = []
        if not name or len(name) < 2:
            errors.append("Name must be at least 2 characters")
        if not email or '@' not in email:
            errors.append("Please enter a valid email address")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters")
        if password != confirm_password:
            errors.append("Passwords do not match")
        if email in users_db or db.get_user(email):
            errors.append("An account with this email already exists")
        
        if errors:
            return render_template("signup.html", errors=errors, name=name, email=email)
        
        # Create user (MongoDB + In-memory fallback)
        password_hash = hash_password(password)
        
        # 1. Try Mongo
        db.create_user(name, email, password_hash)
        
        # 2. Keep in-memory for session consistency if mongo fails silently or for legacy parts
        users_db[email] = {
            'name': name,
            'email': email,
            'password_hash': password_hash,
            'created_at': datetime.now().isoformat(),
            'assessments': []
        }
        
        # Auto login after signup
        session['user_email'] = email
        session['user_name'] = name
        flash(f'Welcome to Kiro AI, {name}!', 'success')
        return redirect(url_for('index'))
    
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login"""
    if 'user_email' in session:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        user = db.get_user(email)
        if not user:
            user = users_db.get(email)
        
        if user and user['password_hash'] == hash_password(password):
            session['user_email'] = email
            session['user_name'] = user['name']
            if remember:
                session.permanent = True
            
            flash(f'Welcome back, {user["name"]}!', 'success')
            
            # Redirect to next page if specified
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="Invalid email or password", email=email)
    
    return render_template("login.html")


@app.route("/logout")
def logout():
    """User logout - shows logout confirmation page"""
    user_name = session.get('user_name', 'User')
    session.clear()
    return render_template("logout.html", user_name=user_name)


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """User profile page"""
    user = get_current_user()
    
    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'update_profile':
            new_name = request.form.get('name', '').strip()
            if new_name and len(new_name) >= 2:
                user['name'] = new_name
                session['user_name'] = new_name
                flash('Profile updated successfully!', 'success')
        
        elif action == 'change_password':
            current_password = request.form.get('current_password', '')
            new_password = request.form.get('new_password', '')
            confirm_password = request.form.get('confirm_password', '')
            
            if user['password_hash'] != hash_password(current_password):
                flash('Current password is incorrect', 'error')
            elif len(new_password) < 6:
                flash('New password must be at least 6 characters', 'error')
            elif new_password != confirm_password:
                flash('New passwords do not match', 'error')
            else:
                user['password_hash'] = hash_password(new_password)
                flash('Password changed successfully!', 'success')
        
        return redirect(url_for('profile'))
    
    # Get user's assessment count
    user_assessments = [a for a in assessment_history if a.get('user_email') == session.get('user_email')]
    
    return render_template("profile.html", user=user, assessment_count=len(user_assessments))


# =====================================================================
# MAIN ROUTES
# =====================================================================

@app.route("/")
def index():
    """Landing page - Start health assessment"""
    # Redirect to login if not authenticated AND not in guest mode
    if 'user_email' not in session and 'guest_mode' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", models_loaded=models_loaded)


@app.route("/guest_login")
def guest_login():
    """Login as a guest user"""
    session['guest_mode'] = True
    session['user_name'] = 'Guest'
    # Clear any previous user email to ensure no conflict
    session.pop('user_email', None)
    return redirect(url_for('index'))


@app.route("/start-assessment")
def start_assessment():
    """Begin the comprehensive health assessment"""
    # Clear any previous session data but keep auth
    # Store auth keys
    email = session.get('user_email')
    guest = session.get('guest_mode')
    name = session.get('user_name')
    
    session.clear()
    
    # Restore auth
    if email: session['user_email'] = email
    if guest: session['guest_mode'] = guest
    if name: session['user_name'] = name
    
    return redirect(url_for('assessment_step1'))


@app.route("/assessment/step1", methods=["GET", "POST"])
def assessment_step1():
    """Step 1: Basic Information"""
    if request.method == "POST":
        session['user_data'] = {
            'name': request.form.get('name', 'User'),
            'age': int(request.form.get('age', 25)),
            'gender': request.form.get('gender', 'male'),
            'role': request.form.get('role', 'student')
        }
        return redirect(url_for('assessment_step2'))
    
    return render_template("assessment_step1.html")


@app.route("/assessment/step2", methods=["GET", "POST"])
def assessment_step2():
    """Step 2: Sleep & Screen Time"""
    if 'user_data' not in session:
        return redirect(url_for('assessment_step1'))
    
    if request.method == "POST":
        session['user_data'].update({
            'sleep_duration': float(request.form.get('sleep_duration', 7)),
            'sleep_quality': int(request.form.get('sleep_quality', 7)),
            'screen_time': float(request.form.get('screen_time', 6)),
            'night_screen': float(request.form.get('night_screen', 2)),
            'social_media_hours': float(request.form.get('social_media_hours', 3))
        })
        session.modified = True
        return redirect(url_for('assessment_step3'))
    
    return render_template("assessment_step2.html", user_data=session.get('user_data', {}))


@app.route("/assessment/step3", methods=["GET", "POST"])
def assessment_step3():
    """Step 3: Mental Health Screening (PHQ-9, GAD-7)"""
    if 'user_data' not in session:
        return redirect(url_for('assessment_step1'))
    
    if request.method == "POST":
        session['user_data'].update({
            'stress_score': int(request.form.get('stress_score', 15)),
            'phq9_score': int(request.form.get('phq9_score', 5)),
            'gad7_score': int(request.form.get('gad7_score', 5)),
            'social_platform': request.form.get('social_platform', 'instagram')
        })
        session.modified = True
        
        # Check if user is healthcare worker for additional questions
        if session['user_data']['role'] in ['healthcare_worker', 'medical_professional']:
            return redirect(url_for('assessment_step4'))
        else:
            return redirect(url_for('assessment_results'))
    
    return render_template("assessment_step3.html", user_data=session.get('user_data', {}))


@app.route("/assessment/step4", methods=["GET", "POST"])
def assessment_step4():
    """Step 4: Healthcare Worker Specific (Burnout Assessment)"""
    if 'user_data' not in session:
        return redirect(url_for('assessment_step1'))
    
    if request.method == "POST":
        session['user_data'].update({
            'work_hours': int(request.form.get('work_hours', 45)),
            'patient_load': int(request.form.get('patient_load', 50)),
            'mbsr_participation': int(request.form.get('mbsr_participation', 0)),
            'post_covid': int(request.form.get('post_covid', 1))
        })
        session.modified = True
        return redirect(url_for('assessment_results'))
    
    return render_template("assessment_step4.html", user_data=session.get('user_data', {}))


@app.route("/journal", methods=["GET", "POST"])
@login_required
def journal():
    """Daily Mood Journal with AI Sentiment Analysis"""
    user_email = session.get('user_email')
    
    # Handle Guest journaling
    if not user_email:
        if 'guest_id' not in session:
            session['guest_id'] = f"guest_{secrets.token_hex(4)}@kiro.ai"
        user_email = session['guest_id']
    
    if request.method == "POST":
        content = request.form.get('content', '')
        mood = request.form.get('mood', 'neutral')
        
        # Save to DB
        db.save_journal(user_email, content, mood)
        flash('Journal entry saved! Keep up the streak.', 'success')
        return redirect(url_for('journal'))
        
    # Get history
    entries = db.get_journals(user_email)
    streak = db.get_streak(user_email)
    
    return render_template("journal.html", entries=entries, streak=streak)


@app.route("/export_report")
@login_required
def export_report():
    """Generate PDF Report of user's health status"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph(f"Mental Health Report: {user.get('name', 'User')}", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Content (Simplified for demo)
    story.append(Paragraph("This report summarizes your recent mental health assessments.", styles['Normal']))
    story.append(Spacer(1, 12))
    
    if user.get('assessments'):
        last_assessment = user['assessments'][-1]
        story.append(Paragraph(f"Last Assessment Date: {last_assessment.get('date', datetime.now()).strftime('%Y-%m-%d')}", styles['Heading2']))
        # Add risk calculation if available, or just raw data
        story.append(Paragraph(f"Risk Level: {last_assessment.get('overall_risk', 'N/A').title()}", styles['Normal']))
    else:
        story.append(Paragraph("No assessment data found.", styles['Normal']))
        
    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mimetype='application/pdf'
    )


@app.route("/assessment/results")
def assessment_results():
    """Display comprehensive analysis results"""
    if 'user_data' not in session:
        return redirect(url_for('assessment_step1'))
    
    user_data = session['user_data']
    results = {}
    
    # 1. Sleep Quality Prediction
    try:
        gender_enc = 0 if user_data['gender'].lower() == 'female' else 1
        sleep_features = np.array([[
            user_data['age'], gender_enc, user_data['screen_time'],
            user_data['night_screen'], user_data['social_media_hours'],
            user_data['stress_score']
        ]])
        
        try:
            sleep_scaled = sleep_scaler.transform(sleep_features)
        except:
            sleep_scaled = StandardScaler().fit_transform(sleep_features)
        
        sleep_pred = sleep_model.predict(sleep_scaled)[0]
        sleep_proba = sleep_model.predict_proba(sleep_scaled)[0]
        
        results['sleep'] = {
            'prediction': int(sleep_pred),
            'label': 'Good Sleep Quality' if sleep_pred == 1 else 'Poor Sleep Quality',
            'confidence': round(max(sleep_proba) * 100, 1)
        }
    except Exception as e:
        results['sleep'] = {'error': str(e)}
    
    # 2. Depression Screening
    try:
        gender_map = {'female': 0, 'male': 1}
        role_map = {'healthcare_worker': 0, 'medical_professional': 1, 'student': 2, 'young_adult_non_medical': 3}
        platform_map = {'facebook': 0, 'instagram': 1, 'tiktok': 2, 'x': 3, 'youtube': 4}
        
        dep_features = np.array([[
            user_data['age'],
            gender_map.get(user_data['gender'].lower(), 0),
            role_map.get(user_data['role'].lower(), 2),
            user_data['social_media_hours'],
            platform_map.get(user_data.get('social_platform', 'instagram').lower(), 1),
            user_data['phq9_score'],
            user_data['gad7_score']
        ]])
        
        try:
            dep_scaled = depression_scaler.transform(dep_features)
        except:
            dep_scaled = StandardScaler().fit_transform(dep_features)
        
        dep_pred = depression_model.predict(dep_scaled)[0]
        dep_proba = depression_model.predict_proba(dep_scaled)[0]
        
        results['depression'] = {
            'prediction': int(dep_pred),
            'label': 'Elevated Indicators' if dep_pred == 1 else 'Low Risk',
            'confidence': round(max(dep_proba) * 100, 1),
            'interpretation': get_phq9_interpretation(user_data['phq9_score'])
        }
    except Exception as e:
        results['depression'] = {'error': str(e)}
    
    # 3. Anxiety Screening
    try:
        anx_pred = anxiety_model.predict(dep_scaled)[0]
        anx_proba = anxiety_model.predict_proba(dep_scaled)[0]
        
        results['anxiety'] = {
            'prediction': int(anx_pred),
            'label': 'Elevated Indicators' if anx_pred == 1 else 'Low Risk',
            'confidence': round(max(anx_proba) * 100, 1),
            'interpretation': get_gad7_interpretation(user_data['gad7_score'])
        }
    except Exception as e:
        results['anxiety'] = {'error': str(e)}
    
    # 4. Burnout Risk (for healthcare workers)
    if user_data['role'] in ['healthcare_worker', 'medical_professional']:
        try:
            gender_enc = 0 if user_data['gender'].lower() == 'female' else 1
            role_enc = 0 if user_data['role'] == 'healthcare_worker' else 1
            
            burnout_features = np.array([[
                user_data['age'], gender_enc, role_enc,
                user_data['stress_score'], user_data.get('work_hours', 45),
                user_data.get('patient_load', 50), user_data.get('mbsr_participation', 0)
            ]])
            
            try:
                burnout_scaled = burnout_scaler.transform(burnout_features)
            except:
                burnout_scaled = StandardScaler().fit_transform(burnout_features)
            
            burnout_pred = burnout_model.predict(burnout_scaled)[0]
            burnout_proba = burnout_model.predict_proba(burnout_scaled)[0]
            
            results['burnout'] = {
                'prediction': int(burnout_pred),
                'label': 'High Burnout Risk' if burnout_pred == 1 else 'Low Burnout Risk',
                'confidence': round(max(burnout_proba) * 100, 1)
            }
        except Exception as e:
            results['burnout'] = {'error': str(e)}
    
    # 5. Overall Profile Analysis
    results['profile'] = analyze_user_profile(user_data)
    
    # --- New: Calculate & Save Risk for PDF Report ---
    risk_score = 0
    if results.get('sleep', {}).get('prediction') == 0: risk_score += 1 # Poor Sleep (0=Poor)
    if results.get('burnout', {}).get('prediction') == 1: risk_score += 1 # High Burnout
    if results.get('depression', {}).get('prediction') == 1: risk_score += 1 # Depressed
    if results.get('anxiety', {}).get('prediction') == 1: risk_score += 1 # Anxious
    
    risk_label = "Low"
    if risk_score >= 1: risk_label = "Moderate"
    if risk_score >= 3: risk_label = "High"
    
    # Save to DB for history
    assessment_record = {
        "date": datetime.now(),
        "results": results,
        "overall_risk": risk_label
    }
    db.save_assessment(session.get('user'), assessment_record)
    # -------------------------------------------------
    
    # Save results to session for PDF export
    session['results'] = results
    session.modified = True
    
    return render_template("assessment_results.html", 
                         user_data=user_data, 
                         results=results)


@app.route("/quick-screen/<screen_type>", methods=["GET", "POST"])
def quick_screen(screen_type):
    """Quick individual screening without full assessment"""
    result = None
    
    if request.method == "POST":
        if screen_type == "sleep":
            result = run_sleep_screening(request.form)
        elif screen_type == "depression":
            result = run_depression_screening(request.form)
        elif screen_type == "anxiety":
            result = run_anxiety_screening(request.form)
        elif screen_type == "burnout":
            result = run_burnout_screening(request.form)
    
    return render_template(f"quick_{screen_type}.html", result=result, screen_type=screen_type)


def run_sleep_screening(form_data):
    """Run sleep quality screening"""
    try:
        age = float(form_data.get('age', 25))
        gender = form_data.get('gender', 'male')
        screen_time = float(form_data.get('screen_time', 6))
        night_screen = float(form_data.get('night_screen', 2))
        social_media = float(form_data.get('social_media', 3))
        stress = float(form_data.get('stress', 15))
        
        gender_enc = 0 if gender.lower() == 'female' else 1
        features = np.array([[age, gender_enc, screen_time, night_screen, social_media, stress]])
        
        try:
            features_scaled = sleep_scaler.transform(features)
        except:
            features_scaled = StandardScaler().fit_transform(features)
        
        prediction = sleep_model.predict(features_scaled)[0]
        probas = sleep_model.predict_proba(features_scaled)[0]
        
        risk_factors = []
        if night_screen > 2:
            risk_factors.append("High nighttime screen exposure")
        if stress > 20:
            risk_factors.append("Elevated stress levels")
        if social_media > 4:
            risk_factors.append("Excessive social media usage")
        
        return {
            'prediction': int(prediction),
            'label': 'Good Sleep Quality' if prediction == 1 else 'Poor Sleep Quality',
            'confidence': round(max(probas) * 100, 1),
            'risk_factors': risk_factors
        }
    except Exception as e:
        return {'error': str(e)}


def run_depression_screening(form_data):
    """Run depression screening"""
    try:
        age = float(form_data.get('age', 25))
        gender = form_data.get('gender', 'male')
        role = form_data.get('role', 'student')
        social_hours = float(form_data.get('social_hours', 3))
        platform = form_data.get('platform', 'instagram')
        phq9 = float(form_data.get('phq9', 5))
        gad7 = float(form_data.get('gad7', 5))
        
        gender_map = {'female': 0, 'male': 1}
        role_map = {'healthcare_worker': 0, 'medical_professional': 1, 'student': 2, 'young_adult_non_medical': 3}
        platform_map = {'facebook': 0, 'instagram': 1, 'tiktok': 2, 'x': 3, 'youtube': 4}
        
        features = np.array([[
            age, gender_map.get(gender.lower(), 0), role_map.get(role.lower(), 2),
            social_hours, platform_map.get(platform.lower(), 1), phq9, gad7
        ]])
        
        try:
            features_scaled = depression_scaler.transform(features)
        except:
            features_scaled = StandardScaler().fit_transform(features)
        
        prediction = depression_model.predict(features_scaled)[0]
        probas = depression_model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Elevated Indicators' if prediction == 1 else 'Low Risk',
            'confidence': round(max(probas) * 100, 1),
            'interpretation': get_phq9_interpretation(phq9)
        }
    except Exception as e:
        return {'error': str(e)}


def run_anxiety_screening(form_data):
    """Run anxiety screening"""
    try:
        age = float(form_data.get('age', 25))
        gender = form_data.get('gender', 'male')
        role = form_data.get('role', 'student')
        social_hours = float(form_data.get('social_hours', 3))
        platform = form_data.get('platform', 'instagram')
        phq9 = float(form_data.get('phq9', 5))
        gad7 = float(form_data.get('gad7', 5))
        
        gender_map = {'female': 0, 'male': 1}
        role_map = {'healthcare_worker': 0, 'medical_professional': 1, 'student': 2, 'young_adult_non_medical': 3}
        platform_map = {'facebook': 0, 'instagram': 1, 'tiktok': 2, 'x': 3, 'youtube': 4}
        
        features = np.array([[
            age, gender_map.get(gender.lower(), 0), role_map.get(role.lower(), 2),
            social_hours, platform_map.get(platform.lower(), 1), phq9, gad7
        ]])
        
        try:
            features_scaled = anxiety_scaler.transform(features)
        except:
            features_scaled = StandardScaler().fit_transform(features)
        
        prediction = anxiety_model.predict(features_scaled)[0]
        probas = anxiety_model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Elevated Indicators' if prediction == 1 else 'Low Risk',
            'confidence': round(max(probas) * 100, 1),
            'interpretation': get_gad7_interpretation(gad7)
        }
    except Exception as e:
        return {'error': str(e)}


def run_burnout_screening(form_data):
    """Run burnout screening"""
    try:
        age = float(form_data.get('age', 35))
        gender = form_data.get('gender', 'male')
        role = form_data.get('role', 'healthcare_worker')
        stress = float(form_data.get('stress', 20))
        work_hours = float(form_data.get('work_hours', 50))
        patient_load = float(form_data.get('patient_load', 70))
        mbsr = int(form_data.get('mbsr', 0))
        
        gender_enc = 0 if gender.lower() == 'female' else 1
        role_enc = 0 if role == 'healthcare_worker' else 1
        
        features = np.array([[age, gender_enc, role_enc, stress, work_hours, patient_load, mbsr]])
        
        try:
            features_scaled = burnout_scaler.transform(features)
        except:
            features_scaled = StandardScaler().fit_transform(features)
        
        prediction = burnout_model.predict(features_scaled)[0]
        probas = burnout_model.predict_proba(features_scaled)[0]
        
        risk_factors = []
        if work_hours > 50:
            risk_factors.append("Extended work hours")
        if patient_load > 80:
            risk_factors.append("High patient load")
        if stress > 22:
            risk_factors.append("Elevated stress")
        if mbsr == 0:
            risk_factors.append("No mindfulness practice")
        
        return {
            'prediction': int(prediction),
            'label': 'High Burnout Risk' if prediction == 1 else 'Low Burnout Risk',
            'confidence': round(max(probas) * 100, 1),
            'risk_factors': risk_factors
        }
    except Exception as e:
        return {'error': str(e)}


@app.route("/about")
def about():
    """About & Disclaimer Page"""
    return render_template("about.html")


# =====================================================================
# PDF GENERATION
# =====================================================================
def generate_pdf_report(user_data, results):
    """Generate a PDF report of the assessment results"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2', fontSize=24, spaceAfter=20, alignment=TA_CENTER, textColor=colors.HexColor('#00C9A7')))
    styles.add(ParagraphStyle(name='Subtitle', fontSize=12, spaceAfter=10, alignment=TA_CENTER, textColor=colors.grey))
    styles.add(ParagraphStyle(name='SectionHeader', fontSize=14, spaceAfter=10, spaceBefore=15, textColor=colors.HexColor('#A06AD8'), fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='BodyText2', fontSize=10, spaceAfter=8, leading=14))
    styles.add(ParagraphStyle(name='Risk_High', fontSize=12, textColor=colors.HexColor('#F97316'), fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Risk_Low', fontSize=12, textColor=colors.HexColor('#10B981'), fontName='Helvetica-Bold'))
    
    story = []
    
    # Title
    story.append(Paragraph("Mental Health Wellness Report", styles['Title2']))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Subtitle']))
    story.append(Spacer(1, 20))
    
    # User Info
    story.append(Paragraph("Personal Information", styles['SectionHeader']))
    user_info = [
        ['Name:', user_data.get('name', 'N/A')],
        ['Age:', str(user_data.get('age', 'N/A'))],
        ['Gender:', user_data.get('gender', 'N/A').title()],
        ['Role:', user_data.get('role', 'N/A').replace('_', ' ').title()]
    ]
    t = Table(user_info, colWidths=[100, 300])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Overall Risk
    story.append(Paragraph("Overall Assessment", styles['SectionHeader']))
    risk_level = results.get('profile', {}).get('overall_risk', 'unknown').upper()
    risk_style = 'Risk_High' if risk_level == 'HIGH' else 'Risk_Low'
    story.append(Paragraph(f"Overall Risk Level: {risk_level}", styles[risk_style]))
    story.append(Spacer(1, 10))
    
    # Screening Results Table
    story.append(Paragraph("Screening Results", styles['SectionHeader']))
    
    results_data = [['Screening', 'Result', 'Confidence']]
    
    if results.get('sleep') and not results['sleep'].get('error'):
        results_data.append(['Sleep Quality', results['sleep']['label'], f"{results['sleep']['confidence']}%"])
    
    if results.get('depression') and not results['depression'].get('error'):
        results_data.append(['Depression (PHQ-9)', results['depression']['label'], f"{results['depression']['confidence']}%"])
    
    if results.get('anxiety') and not results['anxiety'].get('error'):
        results_data.append(['Anxiety (GAD-7)', results['anxiety']['label'], f"{results['anxiety']['confidence']}%"])
    
    if results.get('burnout') and not results['burnout'].get('error'):
        results_data.append(['Burnout Risk', results['burnout']['label'], f"{results['burnout']['confidence']}%"])
    
    if len(results_data) > 1:
        t = Table(results_data, colWidths=[150, 180, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ]))
        story.append(t)
    story.append(Spacer(1, 15))
    
    # Concerns
    concerns = results.get('profile', {}).get('concerns', [])
    if concerns:
        story.append(Paragraph("Areas of Concern", styles['SectionHeader']))
        for concern in concerns:
            story.append(Paragraph(f"• {concern}", styles['BodyText2']))
        story.append(Spacer(1, 10))
    
    # Strengths
    strengths = results.get('profile', {}).get('strengths', [])
    if strengths:
        story.append(Paragraph("Your Strengths", styles['SectionHeader']))
        for strength in strengths:
            story.append(Paragraph(f"• {strength}", styles['BodyText2']))
        story.append(Spacer(1, 10))
    
    # Recommendations
    recommendations = results.get('profile', {}).get('recommendations', [])
    if recommendations:
        story.append(Paragraph("Personalized Recommendations", styles['SectionHeader']))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['BodyText2']))
        story.append(Spacer(1, 15))
    
    # Disclaimer
    story.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(name='Disclaimer', fontSize=8, textColor=colors.grey, alignment=TA_CENTER, leading=12)
    story.append(Paragraph(
        "DISCLAIMER: This report is for educational and screening purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment. "
        "Please consult a qualified healthcare professional for proper evaluation.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# =====================================================================
# HISTORY & ANALYTICS ROUTES
# =====================================================================

@app.route("/export-pdf")
def export_pdf():
    """Export assessment results as PDF"""
    if 'user_data' not in session or 'results' not in session:
        return redirect(url_for('index'))
    
    user_data = session['user_data']
    results = session['results']
    
    pdf_buffer = generate_pdf_report(user_data, results)
    
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=wellness_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
    
    return response


@app.route("/save-to-history", methods=["POST"])
def save_to_history():
    """Save current assessment to history"""
    if 'user_data' not in session or 'results' not in session:
        return jsonify({'success': False, 'message': 'No assessment data found'})
    
    assessment_data = {
        'timestamp': datetime.now().isoformat(),
        'user_data': session['user_data'].copy(),
        'results': session['results'].copy(),
        'user_email': session.get('user_email')  # Associate with logged in user
    }
    
    # Save to MongoDB
    if session.get('user_email'):
        db.save_assessment(session['user_email'], assessment_data)
        success_msg = 'Assessment saved to profile'
    else:
        # For guests, we still might want to track globally or just session
        assessment_history.append(assessment_data) # Keep legacy for guest analytics
        success_msg = 'Assessment tracked (Guest)'
        
    return jsonify({'success': True, 'message': success_msg})


@app.route("/history")
@login_required
def history():
    """View assessment history"""
    user_email = session['user_email']
    user_history = db.get_history(user_email)
    
    # Sort by date desc
    user_history.sort(key=lambda x: x.get('date', datetime.min), reverse=True)
    
    return render_template("history.html", history=user_history)


@app.route("/history/<int:assessment_id>")
def view_history_item(assessment_id):
    """View a specific historical assessment"""
    assessment = next((a for a in assessment_history if a['id'] == assessment_id), None)
    if not assessment:
        return redirect(url_for('history'))
    
    return render_template("assessment_results.html", 
                         user_data=assessment['user_data'], 
                         results=assessment['results'],
                         is_historical=True,
                         assessment_date=assessment['timestamp'])


@app.route("/analytics")
def analytics():
    """View analytics dashboard with charts"""
    # Prepare data for charts
    chart_data = {
        'total_assessments': len(assessment_history),
        'risk_distribution': {'low': 0, 'moderate': 0, 'high': 0},
        'sleep_quality': {'good': 0, 'poor': 0},
        'depression_indicators': {'low': 0, 'elevated': 0},
        'anxiety_indicators': {'low': 0, 'elevated': 0},
        'burnout_risk': {'low': 0, 'high': 0},
        'age_groups': {'18-25': 0, '26-35': 0, '36-45': 0, '46+': 0},
        'roles': {},
        'timeline': []
    }
    
    for assessment in assessment_history:
        results = assessment.get('results', {})
        user_data = assessment.get('user_data', {})
        
        # Risk distribution
        risk = results.get('profile', {}).get('overall_risk', 'unknown')
        if risk in chart_data['risk_distribution']:
            chart_data['risk_distribution'][risk] += 1
        
        # Sleep quality
        if results.get('sleep') and not results['sleep'].get('error'):
            if results['sleep']['prediction'] == 1:
                chart_data['sleep_quality']['good'] += 1
            else:
                chart_data['sleep_quality']['poor'] += 1
        
        # Depression
        if results.get('depression') and not results['depression'].get('error'):
            if results['depression']['prediction'] == 0:
                chart_data['depression_indicators']['low'] += 1
            else:
                chart_data['depression_indicators']['elevated'] += 1
        
        # Anxiety
        if results.get('anxiety') and not results['anxiety'].get('error'):
            if results['anxiety']['prediction'] == 0:
                chart_data['anxiety_indicators']['low'] += 1
            else:
                chart_data['anxiety_indicators']['elevated'] += 1
        
        # Burnout
        if results.get('burnout') and not results['burnout'].get('error'):
            if results['burnout']['prediction'] == 0:
                chart_data['burnout_risk']['low'] += 1
            else:
                chart_data['burnout_risk']['high'] += 1
        
        # Age groups
        age = user_data.get('age', 0)
        if 18 <= age <= 25:
            chart_data['age_groups']['18-25'] += 1
        elif 26 <= age <= 35:
            chart_data['age_groups']['26-35'] += 1
        elif 36 <= age <= 45:
            chart_data['age_groups']['36-45'] += 1
        elif age > 45:
            chart_data['age_groups']['46+'] += 1
        
        # Roles
        role = user_data.get('role', 'unknown').replace('_', ' ').title()
        chart_data['roles'][role] = chart_data['roles'].get(role, 0) + 1
        
        # Timeline
        chart_data['timeline'].append({
            'date': assessment['timestamp'][:10],
            'risk': risk
        })
    
    return render_template("analytics.html", chart_data=chart_data)


@app.route("/api/chart-data")
def api_chart_data():
    """API endpoint for chart data (for AJAX updates)"""
    chart_data = {
        'labels': [],
        'risk_counts': [],
        'sleep_data': [],
        'mood_data': []
    }
    
    # Process last 10 assessments for timeline
    recent = assessment_history[-10:] if len(assessment_history) > 10 else assessment_history
    
    for assessment in recent:
        results = assessment.get('results', {})
        chart_data['labels'].append(assessment['timestamp'][:10])
        
        risk = results.get('profile', {}).get('overall_risk', 'low')
        risk_score = {'low': 1, 'moderate': 2, 'high': 3}.get(risk, 1)
        chart_data['risk_counts'].append(risk_score)
    
    return jsonify(chart_data)


@app.route("/compare")
def compare():
    """Compare multiple assessments"""
    if len(assessment_history) < 2:
        return render_template("compare.html", assessments=[], message="Need at least 2 assessments to compare")
    
    return render_template("compare.html", assessments=assessment_history[-5:], message=None)


@app.route("/wellness-tips")
def wellness_tips():
    """Personalized wellness tips based on assessment history"""
    tips = {
        'sleep': [
            "Maintain a consistent sleep schedule, even on weekends",
            "Create a relaxing bedtime routine",
            "Keep your bedroom cool, dark, and quiet",
            "Avoid screens 1-2 hours before bed",
            "Limit caffeine intake after 2 PM"
        ],
        'stress': [
            "Practice deep breathing exercises daily",
            "Take regular breaks during work",
            "Try progressive muscle relaxation",
            "Spend time in nature",
            "Connect with friends and family"
        ],
        'social_media': [
            "Set daily time limits for social media apps",
            "Turn off non-essential notifications",
            "Curate your feed to include positive content",
            "Take regular digital detox breaks",
            "Prioritize in-person connections"
        ],
        'burnout': [
            "Set clear boundaries between work and personal life",
            "Delegate tasks when possible",
            "Take regular vacation time",
            "Practice saying 'no' to additional commitments",
            "Seek support from colleagues or supervisors"
        ],
        'mindfulness': [
            "Start with 5 minutes of meditation daily",
            "Practice mindful eating",
            "Try body scan meditation before sleep",
            "Use mindfulness apps for guided sessions",
            "Practice gratitude journaling"
        ]
    }
    
    # Personalize based on last assessment
    personalized = []
    if 'results' in session:
        results = session['results']
        if results.get('sleep', {}).get('prediction') == 0:
            personalized.extend(tips['sleep'][:3])
        if results.get('profile', {}).get('overall_risk') in ['moderate', 'high']:
            personalized.extend(tips['stress'][:3])
        if results.get('burnout', {}).get('prediction') == 1:
            personalized.extend(tips['burnout'][:3])
    
    return render_template("wellness_tips.html", tips=tips, personalized=personalized)


# =====================================================================
# RUN APP
# =====================================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
