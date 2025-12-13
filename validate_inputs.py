
import joblib
import numpy as np
import sys
import os

# ---------------------------------------------------------
# 1. SETUP - DEFINING CLASSES FOR PICKLE LOADING
# ---------------------------------------------------------

# Try to import real classes first to avoid pickle errors
try:
    from predict_mental_health import (
        RewardModel, VariationalPreferenceLearning, ActiveLearner,
        EnhancedQuantumTransform, SuperEnsemble95
    )
    print("✓ Imported classes from predict_mental_health")
    
    # Alias to __main__ for pickle compatibility
    if __name__ == "__main__":
        import sys
        current_module = sys.modules[__name__]
        setattr(current_module, 'SuperEnsemble95', SuperEnsemble95)
        setattr(current_module, 'EnhancedQuantumTransform', EnhancedQuantumTransform)
        setattr(current_module, 'RewardModel', RewardModel)
        setattr(current_module, 'VariationalPreferenceLearning', VariationalPreferenceLearning)
        setattr(current_module, 'ActiveLearner', ActiveLearner)
except ImportError:
    print("⚠ Could not import classes. Using dummies (This might fail if pickle expects logic).")
    # Dummies if file is missing (unlikely in this context)
    class RewardModel:
        def __init__(self, n_features): pass
    class VariationalPreferenceLearning:
        def __init__(self, n_features, n_latent): pass
    class ActiveLearner:
        def __init__(self, n_queries=20): pass
    class EnhancedQuantumTransform:
        def __init__(self, n_features): pass
        def transform(self, X): return X
    class SuperEnsemble95:
        def __init__(self, n_features, n_classes): pass

# ---------------------------------------------------------
# 2. VALIDATION LOGIC
# ---------------------------------------------------------

def load_model_and_scaler(name, model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"✗ Failed to load {name}: {e}")
        return None, None

def run_test(name, model, scaler, inputs, feature_names):
    print(f"\n--- Testing {name} ---")
    print(f"Input: {dict(zip(feature_names, inputs[0]))}")
    
    try:
        # Scale
        X_scaled = scaler.transform(inputs)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        print(f"Prediction: {pred}")
        print(f"Probabilities: {proba}")
        print(f"Result: {'POSITIVE/HIGH' if pred==1 else 'NEGATIVE/LOW'} (Conf: {max(proba):.2f})")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    MODEL_DIR = "model"
    
    # -----------------------------------------------------
    # SCENARIO 1: SLEEP QUALITY
    # User Input: Age 25, Male, Screen 6h, Night 2h, Social 3h, Stress 15
    # Features: [age, gender_enc, screen_time_hours, screen_time_night_hours, social_media_hours, stress_score]
    # Gender Enc: Female=0, Male=1
    # -----------------------------------------------------
    sleep_model, sleep_scaler = load_model_and_scaler("Sleep", f"{MODEL_DIR}/students_sleep_model.joblib", f"{MODEL_DIR}/students_sleep_scaler.joblib")
    if sleep_model:
        inputs = np.array([[25, 1, 6.0, 2.0, 3.0, 15]])
        run_test("Sleep Quality", sleep_model, sleep_scaler, inputs, 
                 ["Age", "Gender(M=1)", "Screen", "NightScreen", "Social", "Stress"])

    # -----------------------------------------------------
    # SCENARIO 2: DEPRESSION
    # User Input: Age 25, Male, Student, Social 3h, Instagram, PHQ9=5, GAD7=5 (GAD7 from anxiety form context usually)
    # Features: [age, gender_enc, role_enc, social_media_hours, platform_enc, phq9_score, gad7_score]
    # Mappings from app.py:
    # gender: male=1
    # role: student=2
    # platform: instagram=1
    # -----------------------------------------------------
    dep_model, dep_scaler = load_model_and_scaler("Depression", f"{MODEL_DIR}/depression_model.joblib", f"{MODEL_DIR}/depression_scaler.joblib")
    if dep_model:
        # Note: Screenshot shows PHQ9=5. GAD7 input is in next form but likely similar low value? Assuming 5.
        inputs = np.array([[25, 1, 2, 3.0, 1, 5, 5]])
        run_test("Depression Screening", dep_model, dep_scaler, inputs,
                 ["Age", "Gender", "Role", "Social", "Platform", "PHQ9", "GAD7"])

    # -----------------------------------------------------
    # SCENARIO 3: ANXIETY
    # Same inputs as above relative to profile
    # -----------------------------------------------------
    anx_model, anx_scaler = load_model_and_scaler("Anxiety", f"{MODEL_DIR}/anxiety_model.joblib", f"{MODEL_DIR}/anxiety_scaler.joblib")
    if anx_model:
        inputs = np.array([[25, 1, 2, 3.0, 1, 5, 5]])
        run_test("Anxiety Screening", anx_model, anx_scaler, inputs,
                 ["Age", "Gender", "Role", "Social", "Platform", "PHQ9", "GAD7"])

    # -----------------------------------------------------
    # SCENARIO 4: BURNOUT
    # User Input: Age 35, Male, Healthcare Worker, Stress 20, Work 50h, Patients 70, MBSR No
    # Features: [age, gender_enc, role_enc, stress_score, work_hours_per_week, patient_load_per_week, mbsr_participation]
    # Mappings:
    # gender: male=1
    # role: healthcare_worker=0
    # mbsr: no=0
    # -----------------------------------------------------
    burn_model, burn_scaler = load_model_and_scaler("Burnout", f"{MODEL_DIR}/burnout_model.joblib", f"{MODEL_DIR}/burnout_scaler.joblib")
    if burn_model:
        inputs = np.array([[35, 1, 0, 20, 50, 70, 0]])
        run_test("Burnout Risk", burn_model, burn_scaler, inputs,
                 ["Age", "Gender", "Role", "Stress", "WorkHrs", "Patients", "MBSR"])
